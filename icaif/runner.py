"""One live round end to end, and the scheduler that runs a phase's rounds on the clock.

A round, in order (`run_round`):

1. **Book.** The server's portfolio through the kit (`portfolio.parse`); in a dry run,
   a paper book of what the dry runs "submitted". Paper orders of earlier rounds are
   filled first, at the opens they executed at.
2. **Rule.** The 30 names' daily closes, a `live.market` over them, and the backtested
   rule desk, `Desk(RuleBrain())`: at the phase's first round 1 in cash, risk parity at
   the regime-blended exposure; every other round, hold.
3. **Choose.** The fallback chain is the agent's book, then the rule's, then no
   submission. `submit="rule"` (Validation) submits the rule's and the agent shadows;
   `submit="agent"` asks the agent first, inside a budget that leaves time to upload.
4. **Guard.** A hold is written as hold.json, never decision.json. The kit uploads only
   a file named decision.json, so nothing downstream can upload a hold by accident. A
   trade is then checked against the book (`guard`).
5. **Upload.** Only when `live` and the owner's arm file names this phase and submit
   mode (`tools/live_runner.py arm`), through the kit's own `decision()`, which
   re-checks the window and the slot against the server's clock.
6. **Shadow.** With `submit="rule"`, the agent desk runs after the upload, on its own
   paper book and inside the time left before the deadline. Its decision is logged
   beside the submitted one, and its order fills in its paper book.
7. **Record.** Each desk's journal is told what became of its decision (uploaded, on
   paper, held), then one atomic commit of state.json, then a line per round in
   rounds.jsonl, the files in <round_id>/, and each desk's journal in journal/<desk>.json.

**Every file is written whole or not at all** (`write_atomic`: a temp file, fsync, then
a rename). A worker killed mid-write leaves the last good file, and the round, which
never committed, runs again from the state before it: a torn state.json would read as
"no entry yet", and a torn decision.json would be re-used and uploaded as it lay.

Every stage fails toward silence. A missed round holds the book (`docs/rules.md`), which
costs at most a day of a book we had chosen to keep. An upload cannot be taken back: the
first in a window consumes the slot even when invalid, and a re-submitted hold pays the
fee on every name's drift.
"""

import contextlib
import fcntl
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from icaif import calendar, compiler, data, live, sim, watchdog
from icaif import portfolio as P
from icaif.agents import brains
from icaif.agents.desk import Desk, DeskConfig
from icaif.agents.journal import Journal

KIT = data.ROOT / "starter-kit"
KIT_STATE = KIT / ".icaif"
ARM_FILE = KIT_STATE / "ARMED.json"
LIVE_OUT = data.ROOT / "output" / "live"
TOOL = data.ROOT / "tools" / "live_runner.py"
PHASES = ("validation", "official")
# Summed |target - book| below this is drift, not a decision.
MIN_TURNOVER = 0.005
# Entry states that mean the book has (or may have) entered: a second entry over it
# would buy the whole book again. "ambiguous" counts: the kit could not say whether an
# upload committed, and re-entering on a maybe is the expensive guess.
ENTERED = ("dry-run", "uploaded", "executed", "ambiguous")
UPLOAD_TO_ENTRY = {"dry-run": "dry-run", "uploaded": "uploaded", "executed": "executed",
                   "ambiguous": "ambiguous"}


class RunnerError(RuntimeError):
    pass


def et(value) -> pd.Timestamp:
    """A timestamp in ET; a naive one (typed at a prompt) is read as ET already."""
    ts = pd.Timestamp(value)
    return ts.tz_localize(calendar.TZ) if ts.tzinfo is None else ts.tz_convert(calendar.TZ)


@dataclass
class Config:
    phase: str
    submit: str = "rule"          # "rule": the rule's book goes in, the agent shadows
    shadow: str = "claude"        # the agent desk's brain: "claude" | "rule" | "none"
    live: bool = False            # read the server's book; upload when armed
    out: Optional[Path] = None
    window_days: int = 15
    lead_s: float = 12 * 60       # wake this long before each deadline
    upload_margin_s: float = 45   # no upload starts later than this before the deadline
    scoring: bool = True
    # Measured 2026-10-01: 14 s on the EDGAR snapshot, 82 s asking EDGAR for recent
    # filings (live.load_events). A deadlock never finishes, so the limit only has to
    # clear the slow honest run; 4 minutes leaves the shadow 7 of round 1's 12.
    scoring_timeout_s: float = 240
    agent_budget_s: float = 360
    shadow_cost_cap: float = 10.0  # USD per phase
    model: str = brains.DEFAULT_MODEL
    effort: str = "high"

    def __post_init__(self):
        if self.submit not in ("rule", "agent"):
            raise ValueError(f"submit must be 'rule' or 'agent', not {self.submit!r}")
        if self.shadow not in ("claude", "rule", "none"):
            raise ValueError(f"shadow must be 'claude', 'rule' or 'none', not {self.shadow!r}")
        if self.model not in brains.ALLOWED_MODELS:
            raise ValueError(f"model must be one of {brains.ALLOWED_MODELS}, not {self.model!r}")
        if self.live and self.phase not in PHASES:
            raise ValueError(f"only {PHASES} can go live; {self.phase!r} is a dry run's")
        self.out = Path(self.out) if self.out else LIVE_OUT / self.phase


@dataclass
class Round:
    id: str
    phase: str
    day: date
    number: int
    deadline: pd.Timestamp
    execution: pd.Timestamp
    row: dict

    @classmethod
    def from_row(cls, row: dict) -> "Round":
        return cls(row["id"], row["phase"], date.fromisoformat(row["day"]), int(row["number"]),
                   et(row["deadline"]), et(row["execution_time"]), row)


# ----------------------------------------------------------------------------- state

def write_atomic(path: Path, text: str, mode: int = 0o644) -> None:
    """`path` holds `text` whole, or what it held before: never a part.

    A temp file beside it, flushed to disk, then renamed over it (rename is atomic on
    POSIX), then the directory synced so the rename itself survives a power cut. The
    temp's name is fixed per target, so a crash's leftover is overwritten next time and
    never read: readers open only the final name. `mode` applies from creation, so a
    private file (a decision carries the team token) is never briefly world-readable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode), "w") as f:
        os.fchmod(f.fileno(), mode)   # a leftover temp keeps its old mode otherwise
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


class State:
    """The phase's memory between rounds: the entry, the desks and their journals, the
    paper books, spend. One file, so a round commits all of it or none of it: a journal
    saved apart from the paper book it describes could survive a crash the book did not,
    and the next round would reconcile a fill that never happened.
    """

    def __init__(self, out: Path):
        self.path = Path(out) / "state.json"
        self.data = (json.loads(self.path.read_text()) if self.path.exists() else
                     {"entry": None, "rounds": {}, "desks": {}, "paper": {}, "spent_usd": 0.0})

    def save(self) -> None:
        write_atomic(self.path, json.dumps(live._clean(self.data), indent=1))


@contextlib.contextmanager
def phase_lock(out: Path):
    """One worker per phase at a time: two would each read "not entered" and enter."""
    out.mkdir(parents=True, exist_ok=True)
    fd = os.open(out / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RunnerError(f"another runner holds {out}") from None
        yield
    finally:
        os.close(fd)


def write_private(path: Path, text: str) -> None:
    """Mode 0600 from creation, and whole: a live decision.json carries the team token,
    and a retry re-uploads the file a dead worker left, so half of one must never exist."""
    write_atomic(path, text, mode=0o600)


def index_round(out: Path, st: "State", rec: dict) -> None:
    """rounds.jsonl one line longer, rewritten whole.

    A torn last line (an append cut short before writes were atomic) is dropped: it was
    never a whole record. A committed round with no line, because its worker died
    between the commit and this write, gets one back from its round.json, so the index
    a report reads never skips a round the state has.
    """
    path = out / "rounds.jsonl"
    rows = read_lines(path)
    listed = {x.get("round_id") for x in rows}
    for rid in st.data["rounds"]:
        f = out / rid / "round.json"
        if rid not in listed and rid != rec["round_id"] and f.exists():
            rows.append({k: v for k, v in json.loads(f.read_text()).items() if k != "traceback"})
    rows.append({k: v for k, v in rec.items() if k != "traceback"})
    write_atomic(path, "".join(json.dumps(x) + "\n" for x in rows))


def read_lines(path: Path) -> list[dict]:
    """The JSON lines of `path` that parse; a torn one is skipped, not fatal."""
    if not path.exists():
        return []
    rows = []
    for x in path.read_text().splitlines():
        try:
            rows.append(json.loads(x))
        except ValueError:
            continue
    return rows


# ----------------------------------------------------------------------------- the owner's switch

def arm_status(phase: str, submit: str, now: pd.Timestamp) -> tuple[bool, str]:
    """Whether the owner has approved uploads for this phase and submit mode, now.

    The file is written only by `tools/live_runner.py arm`, at a terminal, after the
    server's portfolio parses. It names one phase, one submit mode and an expiry, so
    an approval for Validation's rule book is not one for Official or for the agent.
    """
    if not ARM_FILE.exists():
        return False, "no arm file (the owner runs tools/live_runner.py arm)"
    try:
        a = json.loads(ARM_FILE.read_text())
        expires = et(a["expires"])
    except (OSError, ValueError, KeyError):
        return False, "arm file unreadable; re-arm"
    if a.get("phase") != phase:
        return False, f"armed for {a.get('phase')!r}, not {phase!r}"
    if a.get("submit") != submit:
        return False, f"armed to submit the {a.get('submit')}'s book, not the {submit}'s"
    if now >= expires:
        return False, f"the arm expired at {expires}"
    return True, f"armed at {a.get('armed_at')} until {expires}"


# ----------------------------------------------------------------------------- the kit

def kit_session(readonly: bool = False):
    """The kit's own client. `readonly` uses a separate checkpoint: the kit locks a
    checkpoint per process, and a schedule read must never block a round's upload."""
    from kit.config import load_environment
    from kit.original_client import OriginalSession, load_profile

    load_environment(KIT / ".env")
    load_environment(data.ROOT / ".env")
    token = os.environ.get("CODABENCH_TOKEN", "")
    if not token.strip():
        raise RunnerError("CODABENCH_TOKEN is not set (starter-kit/.env or the environment)")
    profile = Path(os.environ.get("ICAIF_PROFILE") or "profiles/profile99-production.json")
    profile = profile if profile.is_absolute() else KIT / profile
    name = "readonly-checkpoint.json" if readonly else "checkpoint.json"
    # 90 s of receipt polling, not the kit's 180: the submission is created before the
    # poll starts, and a pending receipt is re-read at the next round 1 (refresh_entry).
    return OriginalSession(profile=load_profile(profile), token=token, result_timeout=90,
                           checkpoint=KIT_STATE / name, credentials=KIT_STATE / "credentials.json")


def receipt_outcome(result: dict) -> str:
    """'executed' | 'uploaded' (accepted, not yet executed) | 'ambiguous' | 'failed'.

    Read conservatively: anything that is not clearly a failure keeps the book counted
    as entered. A failure read as success costs a day in cash; a success read as a
    failure buys the whole entry a second time.
    """
    s = {k: str(result.get(k, "")).upper() for k in
         ("status", "validation_status", "selection_status", "execution_status")}
    if s["status"] == "MISSED_DEADLINE":
        return "failed"
    if s["status"] in ("SLOT_CONSUMED", "DUPLICATE") or s["selection_status"] == "DUPLICATE":
        return "ambiguous"  # an attempt already holds the slot: ours from a crashed worker?
    if (s["validation_status"] in ("INVALID", "LATE", "MISSING")
            or s["selection_status"] == "NOT_ELIGIBLE"
            or s["execution_status"] in ("HELD", "EXECUTION_FAILED")):
        return "failed"
    if s["execution_status"] == "EXECUTED":
        return "executed"
    return "uploaded"


def upload(cfg: Config, session, path: Path, r: Round, now: pd.Timestamp) -> dict:
    """The kit uploads decision.json, or nothing does; the reason is always recorded."""
    if not cfg.live:
        return {"status": "dry-run"}
    ok, why = arm_status(cfg.phase, cfg.submit, now)
    if not ok:
        return {"status": "not-armed", "why": why}
    if now > r.deadline - pd.Timedelta(seconds=cfg.upload_margin_s):
        return {"status": "too-late", "why": f"{(r.deadline - now).total_seconds():.0f}s before the deadline"}
    from kit.original_client import AmbiguousSubmission, AutomationError

    try:
        result = session.decision(str(path))
    except AmbiguousSubmission as err:
        return {"status": "ambiguous", "why": str(err)}
    except AutomationError as err:
        # An error after the submission was created (its receipt still pending when the
        # kit stopped polling) is not "nothing was uploaded": the kit's checkpoint says
        # which. Read as refused, the entry would count as failed while it executes.
        op = (getattr(session, "state", None) or {}).get("operations", {}).get(
            f"decision:{r.phase}:{r.id}", {})
        if op.get("submission_id"):
            return {"status": "uploaded", "submission_id": op["submission_id"],
                    "why": f"created; receipt not read: {err}"}
        maybe = "already has a different immutable file" in str(err)
        return {"status": "ambiguous" if maybe else "refused", "why": str(err)}
    return {"status": receipt_outcome(result), "receipt": result, "armed": why,
            "submission_id": result.get("platform_submission_id")}


def refresh_entry(st: State, session, rec: dict) -> None:
    """Re-read an entry whose receipt was pending; an INVALID one frees the next round 1.

    Counted as entered while pending (re-entering on a maybe is the expensive guess),
    an entry the backend later rejected would otherwise keep the book in cash for the
    rest of the phase.
    """
    e = st.data.get("entry") or {}
    sid = e.get("submission_id")
    if e.get("status") != "uploaded" or not sid:
        return
    try:
        result = session.fetch(int(sid))
    except Exception as err:  # noqa: BLE001 - still pending: stays entered
        rec["warnings"].append(f"entry receipt {sid} not readable yet: {type(err).__name__}: {err}")
        return
    new = receipt_outcome(result)
    if new != e["status"]:
        rec["warnings"].append(f"entry {e['round_id']} receipt: {e['status']} -> {new}")
        e.update(status=new, receipt=result)


# ----------------------------------------------------------------------------- deciding

def entered(st: State, book: Optional[P.Book]) -> tuple[bool, str]:
    """Has the phase's book entered? The entry record first, the book second.

    A held share means entered whatever the record says: a lost state file must not
    read as a fresh phase over a book that holds 30 names.
    """
    e = st.data.get("entry")
    if e and e.get("status") in ENTERED:
        return True, f"entry {e['status']} at {e['round_id']}"
    if book is not None and not book.all_cash:
        return True, "the book holds shares"
    return False, "no entry yet" + (f" (last attempt {e['status']} at {e['round_id']})" if e else "")


def guard(target: Optional[dict], current: Optional[pd.Series], *, source: str,
          entered: bool, book_all_cash: bool, round_no: int) -> tuple[bool, str]:
    """(upload?, why): the last check between a desk's answer and decision.json.

    The kit re-sizes every name to its target at the fill, so a "hold" uploaded as
    weights is a trade of every name's drift, fee and turnover rank included. The rule
    trades once a phase, at round 1 from cash; anything else from it is a bug.
    """
    if target is None:
        return False, "hold: the desk returned no trade"
    if current is None or current.isna().any():
        return False, "the book cannot be valued, so a trade cannot be checked against it"
    t = pd.Series(target, dtype=float).reindex(current.index).fillna(0.0)
    moved = (t - current).abs()
    if moved.max() <= compiler.HELD:
        return False, "hold: the target is the book as it stands"
    if source == "rule":
        if entered:
            return False, "the rule trades once, at entry, and this phase's book has entered"
        if not book_all_cash:
            return False, "an entry over a book that holds shares"
        if round_no != 1:
            return False, "the rule enters only at round 1"
    if moved.sum() < MIN_TURNOVER:
        return False, f"hold: turnover {moved.sum():.4f} is drift, not a decision"
    return True, "trade"


class Recording:
    """A brain wrapper that keeps what each role was shown and answered.

    The rules require prompts disclosed in full: the system prompts are versioned in
    `agents/prompts.py`, and the observation and answer of every call are kept here.
    """

    def __init__(self, inner):
        self.inner, self.name, self.calls = inner, inner.name, []

    def decide(self, role, system, payload, schema, timeout):
        entry = {"role": role, "payload": payload, "timeout_s": round(timeout, 1)}
        t0 = time.perf_counter()
        try:
            answer = self.inner.decide(role, system, payload, schema, timeout)
            entry["answer"] = answer.model_dump()
            return answer
        except Exception as err:
            entry["error"] = f"{type(err).__name__}: {err}"
            raise
        finally:
            entry["latency_s"] = round(time.perf_counter() - t0, 2)
            self.calls.append(entry)


class _Spent:
    name = "spent"

    def __init__(self, cap):
        self.cap = cap

    def decide(self, *a, **k):
        raise brains.BrainError(f"the phase's ${self.cap:.2f} shadow budget is spent")


def make_brain(cfg: Config, st: State):
    if cfg.shadow == "rule":
        return brains.RuleBrain()
    left = cfg.shadow_cost_cap - float(st.data.get("spent_usd", 0.0))
    if left <= 0:
        return _Spent(cfg.shadow_cost_cap)
    return brains.make(cfg.model, cfg.effort, max_cost=left)


def run_desk(name: str, brain, cfg: Config, st: State, mkt: sim.Market, r: Round,
             book: P.Book, *, entered_now: Optional[bool] = None, inputs: Optional[dict] = None,
             budget_s: Optional[float] = None) -> tuple[Optional[dict], dict]:
    """One desk's answer this round, its state and journal carried over from the last.

    The desk's journal reconciles against `book` (the server's live) before the desk
    decides; what it found that disagrees is returned as `journal_issues`. A journal
    that fails is recorded and the desk decides without memory (`journal_strict` off):
    the submitted book's entry must not wait on a bug in what the agent is shown.
    """
    dcfg = DeskConfig(window_days=cfg.window_days, anonymize=False, journal_strict=False,
                      round_budget_s=budget_s if budget_s is not None else cfg.agent_budget_s)
    desk = Desk(brain, dcfg, **(inputs or {}))
    desk.restore(st.data["desks"].get(name), mkt.tickers)
    if entered_now is not None:
        desk.book.entered = entered_now
    n, i0 = len(desk.log), len(desk.journal.issues)
    ctx = sim.RoundContext(r.day, r.number, r.deadline, r.execution,
                           {t: book.shares.get(t, 0.0) for t in mkt.tickers}, book.cash, mkt,
                           round_id=r.id, book_source=book.source)
    w = desk(ctx)
    st.data["desks"][name] = desk.state()
    info = {"roles": [{k: e[k] for k in ("role", "source", "reason", "decision", "latency_s")}
                      for e in desk.log[n:]],
            "day_no": desk.day_no, "entered": desk.book.entered}
    if desk.journal_errors:
        info["journal_error"] = desk.journal_errors[-1]
    issues = desk.journal.issues[i0:] if not desk.journal_errors else []
    if issues:
        info["journal_issues"] = [f"{i['kind']}: {i['text']}" for i in issues]
    return w, info


def journal_order(st: State, name: str, r: Round, status: str, *, why: Optional[str] = None,
                  target: Optional[dict] = None, source: Optional[str] = None,
                  ran: bool = True, skip_why: Optional[str] = None) -> None:
    """Tell desk `name`'s journal what became of this round: the status of what was
    submitted for its book (uploaded, dry-run, paper, held by the guard), and whose book
    it was. Without it, every guard hold would read next round as a fill that never came,
    and an upload as a decision nobody made. A desk that did not run gets the round on
    record as skipped, with the order its book was given anyway."""
    d = st.data["desks"].get(name)
    if d is None:
        return
    j = Journal.from_json(d.get("journal"))
    if not ran:
        j.skipped(r.id, r.number, str(r.day), skip_why or "the desk did not run")
    if ran or target is not None:
        j.set_order(r.id, status, why=why, target=target, source=source)
    d["journal"] = j.to_json()


def journal_files(cfg: Config, st: State) -> None:
    """Each desk's journal on its own, journal/<desk>.json, after the commit.

    A copy for reading (`tools/live_runner.py journal`), never read back: the journal of
    record is the one inside state.json, committed with the books it describes. Only
    the desks' own JSON goes in, never a decision file, which carries the team token.
    """
    books = ({"rule": "submitted", "agent": "shadow (paper)"} if cfg.submit == "rule"
             else {"rule": "submitted (fallback)", "agent": "submitted"})
    for name, d in st.data["desks"].items():
        j = d.get("journal")
        if not isinstance(j, dict):
            continue
        write_atomic(cfg.out / "journal" / f"{name}.json", json.dumps(live._clean({
            "phase": cfg.phase, "desk": name, "book": books.get(name, name), "submit": cfg.submit,
            "rounds_on_record": len(j.get("rounds", [])), "pnl_since": Journal.from_json(j).since(),
            "journal": j}), indent=1))


def _describe(w: Optional[dict], info: dict, current: Optional[pd.Series]) -> dict:
    out = {"action": "hold" if w is None else "trade", **info}
    if w is not None:
        t = pd.Series(w, dtype=float)
        out.update({"gross": float(t.sum()), "names": int((t > 0).sum()),
                    "weights": {k: v for k, v in w.items() if v > 0}})
        if current is not None and not current.isna().any():
            out["turnover"] = float((t.reindex(current.index).fillna(0) - current).abs().sum())
    return out


# ----------------------------------------------------------------------------- doors

def score_in_child(cfg: Config, r: Round, now: pd.Timestamp, left_s: float) -> dict:
    """The daily model's scores for `r.day`, from a child the watchdog can kill.

    Once a day: a later round reuses the files. One retry, in a fresh process, if
    the first attempt timed out and there is time: the deadlock was a property of a
    process, not of the data.
    """
    target = cfg.out / "scores" / str(r.day)
    if (target / "scores.parquet").exists():
        return {"status": "cached", "dir": str(target)}
    tries = []
    for _ in range(2):
        if left_s - 30 < 20:
            tries.append({"skipped": f"{left_s:.0f}s left before the deadline"})
            break
        timeout = min(cfg.scoring_timeout_s, left_s - 30)
        t0 = time.monotonic()
        res = watchdog.run([sys.executable, str(TOOL), "score", "--as-of", now.isoformat(),
                            "--out", str(target)], timeout, env={"OMP_NUM_THREADS": "1"})
        tries.append(res.summary())
        left_s -= time.monotonic() - t0
        if res.ok or not res.timed_out:
            break
    ok = (target / "scores.parquet").exists()
    return {"status": "ok" if ok else ("timeout" if tries and tries[-1].get("timed_out") else "failed"),
            "dir": str(target), "tries": tries}


# A round this close to its deadline on the wall clock is happening now: it archives the
# feeds itself before the shadow decides. A rehearsal of a past day never does.
NEWS_NOW_S = 30 * 60


def agent_inputs(cfg: Config, mkt: sim.Market, r: Round) -> tuple[dict, dict]:
    """What the agent desk reads besides prices, each loaded alone: a missing input is
    recorded and left out, never a reason to drop the others or the round."""
    from icaif import macro, news, universe

    kw, meta, tickers = {}, {}, mkt.tickers
    sdir = cfg.out / "scores" / str(r.day)
    filings_meta = {}

    def load_filings():
        events, m = live.load_filings(tickers, r.deadline, cfg.out / "filings" / "text")
        filings_meta.update(m)
        return events

    loaders = {
        "scores": lambda: compiler.DailyPanel(
            pd.read_parquet(sdir / "scores.parquet").pivot(index="date", columns="ticker",
                                                           values="pred"), tickers),
        "universe_scores": lambda: live.universe_scores(sdir, tickers),
        "context": lambda: macro.wide(pd.read_parquet(sdir / "prices_context.parquet")),
        "earnings": lambda: live.CalendarEarnings(
            pd.read_parquet(universe.latest("earnings_calendar_*.parquet")), mkt.days),
        "fomc": macro.FomcCalendar.load,
        "filings": load_filings,
        "vol": lambda: live.vol_forecasts(r.deadline, cfg.out / "vol" / str(r.day), tickers),
    }
    for key, load in loaders.items():
        try:
            value = load()
            if value is not None:
                kw[key] = value
            meta[key] = "ok" if value is not None else "none"
        except Exception as err:  # noqa: BLE001 - one missing input, not a missing round
            meta[key] = f"{type(err).__name__}: {err}"
    if "earnings" in kw:
        meta["earnings_snapshot"] = universe.latest("earnings_calendar_*.parquet").name
    if filings_meta:
        meta["filings_source"] = filings_meta
    if news.ARCHIVE.exists():
        # The scheduled archiver runs 5 minutes before each deadline, after this round's
        # shadow has decided (the runner wakes 12 minutes before): read alone, the archive
        # would show the shadow headlines an hour old, missing whatever moved the name.
        left = (r.deadline - pd.Timestamp.now(tz=calendar.TZ)).total_seconds()
        if 0 < left <= NEWS_NOW_S:
            try:
                path, frame, issues = news.snapshot(tickers, budget_s=40)
                meta["news_snapshot"] = {"file": path.name, "headlines": len(frame), **issues}
            except Exception as err:  # noqa: BLE001 - the archive as it stands still serves
                meta["news_snapshot"] = f"{type(err).__name__}: {err}"
        kw["news_dir"] = news.ARCHIVE
        meta["news"] = "archive"
    return kw, meta


@dataclass
class Doors:
    """Everything a round touches outside this process; tests replace them."""

    now: Callable[[], pd.Timestamp]
    closes: Callable = live.fetch_closes
    intraday: Callable = live.fetch_intraday
    session: Optional[Callable] = None
    score: Callable = score_in_child
    inputs: Callable = agent_inputs
    brain: Callable = make_brain
    book: Optional[Callable] = None   # a dry run's stand-in for the server's book


# ----------------------------------------------------------------------------- the round

def run_round(cfg: Config, row: dict, doors: Doors) -> dict:
    """Run one round and return its record (also appended to rounds.jsonl)."""
    r = Round.from_row(row)
    t0 = time.perf_counter()
    rec = {"round_id": r.id, "phase": cfg.phase, "submit": cfg.submit, "shadow": cfg.shadow,
           "live": cfg.live, "host": socket.gethostname(), "deadline": str(r.deadline),
           "execution": str(r.execution), "warnings": [], "errors": [], "timings": {}}
    with phase_lock(cfg.out):
        st = State(cfg.out)
        now = doors.now()
        rec["started"] = str(now)
        prior = st.data["rounds"].get(r.id, {})
        if prior.get("upload") in ("uploaded", "executed", "ambiguous"):
            rec["outcome"] = f"skipped: this round already has an upload ({prior['upload']})"
        elif now >= r.deadline:
            rec["outcome"] = "missed: started after the deadline"
        else:
            try:
                cm = doors.session() if cfg.live else contextlib.nullcontext(None)
                with cm as session:
                    _round(cfg, r, st, session, doors, rec, t0)
            except Exception as err:  # noqa: BLE001 - recorded; the round holds
                rec["errors"].append(f"round: {type(err).__name__}: {err}")
                rec["traceback"] = traceback.format_exc()[-3000:]
                rec.setdefault("outcome", "no submission: the round failed")
        rec["timings"]["total"] = round(time.perf_counter() - t0, 3)
        st.data["rounds"][r.id] = {"outcome": rec.get("outcome"), "at": str(doors.now()),
                                   "upload": rec.get("submitted", {}).get("upload", {}).get("status")}
        st.save()   # the commit: everything below is a record of it, rewritten whole
        rec = live._clean(rec)
        record = cfg.out / r.id / "round.json"
        if record.exists() and str(rec.get("outcome", "")).startswith("skipped"):
            record = record.with_name("round.rerun.json")   # the round's own record stays
        write_atomic(record, json.dumps(rec, indent=2))
        index_round(cfg.out, st, rec)
        journal_files(cfg, st)
    return rec


def _round(cfg, r: Round, st: State, session, doors: Doors, rec: dict, t0: float) -> None:
    tickers = sorted(data.load_universe())
    rdir = cfg.out / r.id
    rdir.mkdir(parents=True, exist_ok=True)

    def lap(stage):
        rec["timings"][stage] = round(time.perf_counter() - t0, 3)

    # 1. data, and the paper books' fills
    daily = bars30 = None
    try:
        daily, rec["closes"] = doors.closes(doors.now())
    except Exception as err:  # noqa: BLE001
        rec["errors"].append(f"closes: {type(err).__name__}: {err}")
    lap("closes")
    try:
        bars30, rec["intraday"] = doors.intraday(doors.now())
    except Exception as err:  # noqa: BLE001 - rounds 2-7 lose their trigger, not their book
        rec["warnings"].append(f"intraday bars unavailable: {type(err).__name__}: {err}")
    lap("intraday")
    opens = live.fills(bars30) if bars30 is not None and len(bars30) else None
    papers = {k: P.PaperBook.from_json(st.data["paper"].get(k)) for k in ("submitted", "agent")}
    for k, pb in papers.items():
        for issue in pb.settle(opens, doors.now()):
            rec["warnings"].append(f"paper {k}: {issue}")

    # 2. the book
    book = None
    try:
        if doors.book is not None:
            book = doors.book()
        elif cfg.live:
            raw = session.portfolio(cfg.phase)
            write_private(rdir / "portfolio_raw.json", json.dumps(raw, default=str, indent=1))
            book = P.parse(raw, tickers)
        else:
            book = papers["submitted"].book(tickers)
    except Exception as err:  # noqa: BLE001 - without a book there is nothing to check a trade against
        rec["errors"].append(f"book: {type(err).__name__}: {err}")
    lap("book")

    mkt = None
    if daily is not None:
        latest = pd.Timestamp(rec["closes"]["latest_session"])
        today = live.today_60m(bars30, r.day) if bars30 is not None else None
        # The 30m opens are the paper books' fills, and the journals price fills at them.
        mkt = live.market(daily, today, live.market_days(daily, latest), fill_opens=opens)
        if r.day not in mkt.days:
            rec["errors"].append(f"{r.day} is not a session on the NYSE calendar")
            mkt = None
    current = None
    if mkt is not None and book is not None:
        last = mkt.recent_closes(r.deadline, 1)
        current = book.weights(last.iloc[-1]) if len(last) else None
    last = mkt.recent_closes(r.deadline, 1) if mkt is not None else None
    rec["book"] = None if book is None else book.summary(
        last.iloc[-1] if last is not None and len(last) else None)
    if cfg.live and session is not None and r.number == 1:
        refresh_entry(st, session, rec)
    was, why = entered(st, book)
    rec["entered"] = {"value": was, "why": why}

    # 2-3. the rule, and the agent first when it is the one submitted
    rule_w, rule_ok = None, False
    if mkt is not None and book is not None:
        rb = Recording(brains.RuleBrain())
        try:
            rule_w, info = run_desk("rule", rb, cfg, st, mkt, r, book, entered_now=was)
            rule_ok = True
            if rb.calls:
                info["p_turbulent_next"] = rb.calls[-1]["payload"]["market"]["p_turbulent_next_session"]
            rec["rule"] = _describe(rule_w, info, current)
        except Exception as err:  # noqa: BLE001
            rec["errors"].append(f"rule desk: {type(err).__name__}: {err}")
            rec["traceback"] = traceback.format_exc()[-3000:]
    lap("rule")

    agent_w, agent_ok = None, False
    if cfg.submit == "agent" and mkt is not None and book is not None:
        agent_w, agent_ok = _agent(cfg, r, st, doors, rec, mkt, book, current,
                                   reserve_s=cfg.upload_margin_s + 60, entered_now=was, label="agent")
    lap("agent")

    if agent_ok:
        chosen, source = agent_w, "agent"
    elif rule_ok:
        chosen, source = rule_w, "rule"
        if cfg.submit == "agent":
            rec["warnings"].append("the agent desk failed; the rule's answer stands")
    else:
        chosen, source = None, "none"

    # 4. the guard, and the file
    ok, why = (False, "no desk produced an answer") if source == "none" else guard(
        chosen, current, source=source, entered=was,
        book_all_cash=bool(book is not None and book.all_cash), round_no=r.number)
    if ok and cfg.live and not getattr(session, "creds", None):
        ok, why = False, "no team credentials in starter-kit/.icaif; nothing can be uploaded"
    sub = {"source": source, "action": "trade" if ok else "hold", "guard": why}
    path = rdir / "decision.json"
    if ok and cfg.live and path.exists():
        # A worker that died after writing (perhaps after uploading) left this file. The
        # kit recovers an uncertain upload by the file's hash, so the retry resumes with
        # the same bytes; a recomputed file would read as a second, different upload.
        sub["reused_file"] = True
    elif ok:
        decision = live.envelope(chosen, r.row, team=session.creds if cfg.live else None)
        write_private(path, live.check_decision(decision))
    else:
        path = rdir / "hold.json"
        write_atomic(path, json.dumps(live._clean({"round_id": r.id, "source": source, "reason": why,
                                                   "target": chosen}), indent=2))
    sub["file"] = str(path)
    lap("decision_ready")
    rec["ready_before_deadline_s"] = round((r.deadline - doors.now()).total_seconds(), 1)

    # 5. upload
    up = upload(cfg, session, path, r, doors.now()) if ok else {"status": "hold"}
    sub["upload"] = up
    rec["submitted"] = sub
    lap("upload")
    if ok and source in ("rule", "agent") and not was:
        st.data["entry"] = {"round_id": r.id, "status": UPLOAD_TO_ENTRY.get(up["status"], "failed"),
                            "upload": up["status"], "source": source,
                            "submission_id": up.get("submission_id")}
    if ok and up["status"] == "dry-run":
        papers["submitted"].order(r.id, r.execution, chosen)
    rec["outcome"] = (f"trade ({source}) -> upload {up['status']}" if ok else
                      why if why.startswith("hold") else f"no submission: {why}")
    status = up["status"] if ok else "hold"
    sent = chosen if ok else None
    if cfg.submit == "rule":
        if rule_w is not None or not rule_ok:
            journal_order(st, "rule", r, status, why=why, ran=rule_ok,
                          skip_why="the rule desk did not run: " + "; ".join(rec["errors"])[:200])
    else:   # both desks ran on the submitted book: tell each whose book went in
        for name, ran in (("rule", rule_ok), ("agent", agent_ok)):
            journal_order(st, name, r, status, why=why, target=sent, source=source, ran=ran,
                          skip_why=f"the {name} desk did not run")

    # 6. the shadow
    if cfg.submit == "rule" and cfg.shadow != "none" and mkt is not None:
        shadow_book = papers["agent"].book(tickers)
        last = mkt.recent_closes(r.deadline, 1)
        shadow_current = shadow_book.weights(last.iloc[-1]) if len(last) else None
        w, ok_ = _agent(cfg, r, st, doors, rec, mkt, shadow_book, shadow_current,
                        reserve_s=30, entered_now=None, label="shadow")
        if ok_ and w is not None:
            s_ok, s_why = guard(w, shadow_current, source="agent", entered=True,
                                book_all_cash=shadow_book.all_cash, round_no=r.number)
            rec["shadow"]["guard"] = s_why
            if s_ok:
                papers["agent"].order(r.id, r.execution, w)
            journal_order(st, "agent", r, "paper" if s_ok else "hold", why=s_why)
        elif not ok_:
            sh = rec.get("shadow") or {}
            journal_order(st, "agent", r, "hold", ran=False,
                          skip_why=f"the shadow did not run: {sh.get('why') or sh.get('error') or 'unknown'}")
    lap("shadow")
    for k, pb in papers.items():
        st.data["paper"][k] = pb.to_json()
    for label in ("rule", "agent", "shadow"):
        x = rec.get(label) or {}
        for issue in x.get("journal_issues", []):
            rec["warnings"].append(f"journal ({label}): {issue}")
        if x.get("journal_error"):
            rec["warnings"].append(f"journal ({label}) failed, the desk decided without memory: "
                                   f"{x['journal_error']}")


def _agent(cfg, r, st, doors, rec, mkt, book, current, *, reserve_s, entered_now, label):
    """The agent desk, finished `reserve_s` before the deadline; (weights, ran).

    A crash is recorded, not raised: in shadow mode the submitted book is already
    uploaded, and in agent mode the rule's answer stands in.
    """
    def left():
        return (r.deadline - doors.now()).total_seconds() - reserve_s

    if left() < 30:
        rec["warnings"].append(f"{label}: {left():.0f}s left before the deadline; not run")
        rec[label] = {"action": "not run", "why": "no time left"}
        return None, False
    brain, meta, budget = None, {}, 0.0
    try:
        if cfg.scoring:
            rec[f"{label}_scores"] = doors.score(cfg, r, doors.now(), left())
        inputs, meta = doors.inputs(cfg, mkt, r)
        brain = Recording(doors.brain(cfg, st))
        problem = brains.credentials_problem(cfg.model) if cfg.shadow == "claude" else None
        if problem:
            rec["warnings"].append(f"{label}: {problem}; every role falls back to the rule")
        budget = min(cfg.agent_budget_s, max(left(), 0))
        w, info = run_desk("agent", brain, cfg, st, mkt, r, book, entered_now=entered_now,
                           inputs=inputs, budget_s=budget)
    except Exception as err:  # noqa: BLE001
        rec["errors"].append(f"{label} desk: {type(err).__name__}: {err}")
        rec[label] = {"action": "failed", "error": f"{type(err).__name__}: {err}", "inputs": meta}
        return None, False
    finally:
        if brain is not None:
            if isinstance(brain.inner, brains.PAID):
                st.data["spent_usd"] = float(st.data.get("spent_usd", 0.0)) + brain.inner.cost()
            if brain.calls:
                write_atomic(cfg.out / r.id / f"{label}_calls.json",
                             json.dumps(live._clean(brain.calls), indent=1, default=str))
    rec[label] = {**_describe(w, info, current), "inputs": meta, "brain": brain.name,
                  "budget_s": round(budget, 1)}
    return w, True


# ----------------------------------------------------------------------------- schedules

def phase_rounds(schedule: dict, phase: str) -> list[dict]:
    """The phase's rounds that will run, in deadline order."""
    rows = [r for r in schedule.get("rounds", []) if r.get("phase") == phase
            and r.get("status") != "CANCELLED"]
    return sorted(rows, key=lambda r: et(r["deadline"]))


def rehearsal_schedule(days: list[date], phase: str) -> dict:
    """The standard calendar's rounds on `days`, under a phase no server knows.

    Round ids are `<phase>-<day>-r<n>`, so even a decision written for one cannot
    match a real round, and the dry-run envelope adds `dryrun-` on top.
    """
    rows, prev = [], None
    for d in days:
        for x in calendar.rounds_for(d):
            opens = (prev + pd.Timedelta(minutes=10)) if prev is not None else x["deadline"] - pd.Timedelta(hours=17)
            rows.append({"id": f"{phase}-{d}-r{x['round']}", "phase": phase, "day": str(d),
                         "number": x["round"], "opens_at": opens.isoformat(),
                         "deadline": x["deadline"].isoformat(),
                         "execution_time": x["execution"].isoformat(),
                         "close_time": calendar.at(d, calendar.session_close(d)).isoformat(),
                         "status": "SCHEDULED"})
            prev = x["execution"]
    return {"source": "rehearsal", "rounds": rows, "symbols": sorted(data.load_universe())}


def bundled_schedule() -> dict:
    """The kit's own schedule.json: the published calendar, for dry runs of a real phase."""
    from kit.config import load_schedule

    return load_schedule()


def server_schedule() -> tuple[dict, float]:
    """The organizers' live schedule and the server clock's offset from ours (s)."""
    local = pd.Timestamp.now(tz="UTC")
    with kit_session(readonly=True) as s:
        sched = s.schedule()
    server = sched.get("current_time")
    return sched, (et(server) - local).total_seconds() if server else 0.0


# ----------------------------------------------------------------------------- the clock

def keep_awake() -> Optional[subprocess.Popen]:
    """macOS: hold an idle- and system-sleep assertion for as long as this process lives.
    A sleeping Mac misses rounds in silence; a closed lid still sleeps without power."""
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        return subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])
    return None


def say(msg: str, out: Optional[Path] = None) -> None:
    line = f"{pd.Timestamp.now(tz=calendar.TZ):%Y-%m-%d %H:%M:%S %Z}  {msg}"
    print(line, flush=True)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "runner.log", "a") as f:
            f.write(line + "\n")


def worker_argv(cfg: Config, row: dict, schedule_path: Path, as_of=None) -> list[str]:
    argv = [sys.executable, str(TOOL), "round", "--phase", cfg.phase, "--round-id", row["id"],
            "--schedule", str(schedule_path), "--submit", cfg.submit, "--shadow", cfg.shadow,
            "--model", cfg.model, "--out", str(cfg.out)]
    if cfg.live:
        argv.append("--live")
    if not cfg.scoring:
        argv.append("--no-scores")
    if as_of is not None:
        argv += ["--as-of", pd.Timestamp(as_of).isoformat()]
    return argv


def _now_et() -> pd.Timestamp:
    return pd.Timestamp.now(tz=calendar.TZ)


def run_phase(cfg: Config, read_schedule: Callable[[], tuple[dict, float]], *,
              fast: bool = False, sleep: Callable[[float], None] = time.sleep,
              spawn: Callable = watchdog.run, clock: Callable[[], pd.Timestamp] = _now_et) -> list[dict]:
    """Run every remaining round of the phase, each in a worker the watchdog can kill.

    The schedule is re-read at least every ten minutes while waiting, so an organizer
    cancellation or a moved deadline is seen before the wake, and the wake is computed
    on the server's clock. A worker that dies before recording its round is retried
    once while four minutes remain; after the deadline the round is recorded missed.
    `fast` replays the rounds back to back with each worker's clock set to its wake, and
    retries a dead worker once too, at the same wake: a replay that skipped the retry
    would rehearse a recovery the live runner does not have.
    """
    cfg.out.mkdir(parents=True, exist_ok=True)
    if not fast:
        keep_awake()
    snap = cfg.out / "schedule.json"
    retried: set = set()
    done: list[dict] = []
    while True:
        sched, offset = read_schedule()
        rows = phase_rounds(sched, cfg.phase)
        if not rows:
            say(f"{cfg.phase}: the schedule has no rounds", cfg.out)
            return done
        cfg.window_days = len({x["day"] for x in rows})
        write_atomic(snap, json.dumps({**sched, "clock_offset_s": offset,
                                       "read_at": str(clock()),
                                       "window_days": cfg.window_days}, default=str))
        st = State(cfg.out)
        todo = [x for x in rows if x["id"] not in st.data["rounds"]]
        if fast:
            if not todo:
                return done
            row = todo[0]
            now = et(row["deadline"]) - pd.Timedelta(seconds=cfg.lead_s)
        else:
            now = clock() + pd.Timedelta(seconds=offset)
            todo = [x for x in todo if et(x["deadline"]) > now]
            if not todo:
                say(f"{cfg.phase}: every round is done or past", cfg.out)
                return done
            row = todo[0]
        deadline = et(row["deadline"])
        wake = deadline - pd.Timedelta(seconds=cfg.lead_s)
        if not fast and now < wake:
            wait = min((wake - now).total_seconds(), 600.0)
            if wait > 60:
                say(f"next: {row['id']} wakes {wake:%a %H:%M %Z}; sleeping {wait / 60:.0f} min", cfg.out)
            sleep(wait)
            continue
        say(f"{row['id']}: start (deadline {deadline:%H:%M %Z})", cfg.out)
        timeout = 3600.0 if fast else max((deadline - now).total_seconds() + 300.0, 60.0)
        res = spawn(worker_argv(cfg, row, snap, as_of=now if fast else None), timeout)
        st = State(cfg.out)
        got = st.data["rounds"].get(row["id"])
        if got is None:
            left = (deadline - clock() - pd.Timedelta(seconds=offset)).total_seconds()
            say(f"{row['id']}: worker ended without a record ({res.summary()})", cfg.out)
            if row["id"] not in retried and (fast or left > 240):
                retried.add(row["id"])
                continue
            st.data["rounds"][row["id"]] = {"outcome": "missed: the worker died", "upload": None,
                                           "worker": res.summary()}
            st.save()
            got = st.data["rounds"][row["id"]]
        say(f"{row['id']}: {got.get('outcome')}", cfg.out)
        done.append({"round_id": row["id"], **got})
