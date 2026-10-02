"""Portfolio memory (Roadmap step 4): the journal against the ledger, across restarts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icaif import calendar, data, runner, sim
from icaif import portfolio as P
from icaif.agents import journal as J
from icaif.agents.brains import RuleBrain
from icaif.agents.desk import Desk, DeskConfig
from icaif.agents.schemas import (EntryDecision, EventDecision, Exclusion, NameCall,
                                  ReviewDecision)
from tests import journal_world as JW
from tests import test_runner as T
from tests.test_agents import Scripted
from tests.test_quant import _bars, _days
from tests.test_sim import TICKERS, _market

ROOT = Path(__file__).resolve().parents[1]
DAYS = _days(80)
START = DAYS[62]
SHOCKED = TICKERS[4]


def _shocked_bars(shock_from=None):
    """Random-walk bars, one name 20% down from 10:30 on START's second day."""
    bars = _bars(DAYS, 8, shock_from=shock_from)
    hit = (bars["ticker"] == SHOCKED) & (bars["start"] >= calendar.at(DAYS[63], calendar.ROUNDS[2][1]))
    bars.loc[hit, ["open", "high", "low", "close"]] *= 0.8
    return bars


def _world(shock_from=None):
    """Fills at each round's own bar open and closes at each day's last bar, so marks,
    fills and the simulator's valuation are one price path."""
    bars = _shocked_bars(shock_from)
    idx = pd.DatetimeIndex([r["execution"] for d in DAYS for r in calendar.rounds_for(d)])
    ex = bars.pivot(index="start", columns="ticker", values="open").reindex(index=idx, columns=TICKERS)
    cidx = pd.DatetimeIndex([calendar.at(d, calendar.session_close(d)) for d in DAYS])
    cl = bars.pivot(index="end", columns="ticker", values="close").reindex(index=cidx, columns=TICKERS)
    return sim.Market(ex, cl, bars)


def _active():
    """A brain that uses every lever the journal must remember."""
    reviews = {"n": 0}

    def review(p):
        reviews["n"] += 1
        held = [r["name"] for r in p["names"] if (r.get("weight_now") or 0) > 0]
        if reviews["n"] == 1:
            return ReviewDecision(action="set_exposure", exposure=0.45, reason=None, exit=[],
                                  trim=[], rationale="a deterioration the regime model is slow to see")
        if reviews["n"] == 2:
            return ReviewDecision(action="hold", exposure=None, reason=None, exit=held[:1],
                                  trim=[], rationale="one name's own news")
        return ReviewDecision(action="hold", exposure=None, reason=None, exit=[], trim=[], rationale="steady")

    return Scripted(
        entry=lambda p: EntryDecision(
            shape="inverse_vol", views="none", exposure=0.6, rationale="a calmer start",
            avoid=[Exclusion(name=p["names"][0]["name"], signal="earnings", why="reports tomorrow")]),
        review=review,
        event=lambda p: EventDecision(calls=[NameCall(name=t["name"], action="exit", fraction=None,
                                                          cause=None, reason="a gap")
                                             for t in p["triggers"]]))


class _Snapshots:
    """The desk, with its journal copied after every round, keyed by (day, round)."""

    def __init__(self, desk):
        self.desk, self.seen = desk, {}

    def __call__(self, ctx):
        out = self.desk(ctx)
        self.seen[(ctx.day, ctx.round)] = json.dumps(self.desk.journal.to_json(), sort_keys=True)
        return out


def _closes(m):
    return m.recent_closes(pd.Timestamp("2100-01-01", tz=calendar.TZ), 10 ** 6)


# ----------------------------------------------------------------------------- the ledger

@pytest.mark.parametrize("anonymize", [False, True])
def test_every_journal_entry_agrees_with_the_ledger_through_an_entry_a_cut_and_exits(anonymize):
    """The memory is only worth reading if it is the book's own history. A fill priced
    off the wrong bar, a cost basis that ignored an add, an exit recorded against the
    wrong round: each would read as plausible history, and the agent would reason
    from it. Every fill, cost, peak and stated lever is rebuilt from the simulator's
    ledger and compared."""
    m = _world()
    desk = Desk(_active(), DeskConfig(anonymize=anonymize))
    res = sim.run(desk, m, START, 6)
    j = desk.journal
    assert J.verify(j, J.sim_fills(res, m), to_ticker=desk.anon.ticker, closes=_closes(m)) == []
    roles = [d["role"] for e in j.rounds for d in e["decisions"]]
    assert {"entry", "review", "event"} <= set(roles)
    assert len(j.closed) >= 2 and len([e for e in j.rounds if e.get("fill")]) >= 4
    assert len(j.rounds) == 6 * 7 and j.issues == []


def test_a_rounds_own_order_has_no_fill_until_a_later_round_sees_it_in_the_book():
    """The entry is decided at 09:10 and fills at 09:30. Recorded as filled at 09:10, the
    entry's own observation would carry a fill price from the future."""
    m = _world()
    snap = _Snapshots(Desk(RuleBrain()))
    sim.run(snap, m, START, 1)
    at_entry = json.loads(snap.seen[(START, 1)])
    assert at_entry["rounds"][-1]["action"] == "trade" and at_entry["rounds"][-1].get("fill") is None
    assert [o["key"] for o in at_entry["orders"]] == [at_entry["rounds"][-1]["key"]]
    after = json.loads(snap.seen[(START, 2)])
    assert after["rounds"][0]["fill"]["at"] == at_entry["rounds"][0]["execution"]
    assert after["orders"] == []
    with pytest.raises(ValueError):
        m.fill_prices(calendar.rounds_for(START)[0]["execution"], calendar.rounds_for(START)[0]["execution"])


def test_the_journal_a_round_sees_is_unchanged_when_every_later_bar_and_fill_is_rewritten():
    """No look-ahead, the repo's standing rule: rewrite every bar and every fill price from
    a day on, and every round before that day's first fill must see the same journal and
    the same memory. A door cut a round late would hand round 1 its own fill price."""
    d = DAYS[64]
    a, b = _active(), _active()
    sa, sb = _Snapshots(Desk(a)), _Snapshots(Desk(b))
    sim.run(sa, _world(), START, 4)
    sim.run(sb, _world(shock_from=d), START, 4)   # every bar and fill from day 3 on, x3
    before = [k for k in sa.seen if k < (d, 2)]
    assert len(before) == 2 * 7 + 1
    assert [sa.seen[k] for k in before] == [sb.seen[k] for k in before]

    def shown(br):
        return [json.dumps(p, sort_keys=True) for _, p in br.seen
                if (p["clock"]["day"], p["clock"]["round"]) <= (3, 1)]

    assert shown(a) == shown(b) and len(shown(a)) >= 3
    assert sa.seen[(d, 2)] != sb.seen[(d, 2)]   # the rewrite reaches the journal once it is past


def _keys(x) -> set:
    if isinstance(x, dict):
        return set(x) | {k for v in x.values() for k in _keys(v)}
    if isinstance(x, list):
        return {k for v in x for k in _keys(v)}
    return set()


def _numbers(x) -> list:
    if isinstance(x, dict):
        return [n for v in x.values() for n in _numbers(v)]
    if isinstance(x, list):
        return [n for v in x for n in _numbers(v)]
    return [x] if isinstance(x, (int, float)) and not isinstance(x, bool) else []


def test_replayed_memory_carries_no_ticker_no_date_and_no_price_level():
    """Claude has read 2016-25. A ticker, a date or a price level in the memory would let
    a replay score recall as judgement, the reason replays are anonymised at all."""
    import re

    b = _active()
    sim.run(Desk(b, DeskConfig(anonymize=True)), _world(), START, 6)
    assert len(b.seen) >= 6
    for _, p in b.seen:
        shown = {"memory": p["memory"], "names": p["names"]}
        text = json.dumps(shown)
        assert not [t for t in TICKERS if re.search(rf"(?<![A-Za-z0-9]){t}(?![A-Za-z0-9])", text)]
        assert "2026-" not in text
        assert not [k for k in _keys(shown) if "price" in k or "date" in k or k.startswith("nav")]
        # Prices here are near 100 and NAV near 1e6; the memory shows returns, weights,
        # fee bps, counts and day numbers, every one far below either.
        journal_fields = [{k: r[k] for k in r if k.endswith("since_entry") or k == "entry_day"}
                          for r in p["names"]]
        assert max(map(abs, _numbers(p["memory"]) + _numbers(journal_fields)), default=0) < 50
    last = b.seen[-1][1]
    assert last["memory"]["closed"] and any("entry_day" in r for r in last["names"])


def test_live_memory_shows_dates_and_entry_prices():
    b = _active()
    sim.run(Desk(b), _world(), START, 3)
    p = b.seen[-1][1]
    held = [r for r in p["names"] if "entry_day" in r]
    assert held and all(r["entry_date"] == str(START) and r["entry_price"] > 0 for r in held)
    assert str(START) in json.dumps(p["memory"])


# ----------------------------------------------------------------------------- held names

def test_a_held_names_gain_and_high_water_mark_come_from_its_fill_and_the_bars_after_it():
    """Step 5's trim reads the give-back from a name's peak. A peak taken from a bar that
    closed before the fill, or a gain measured from the last close instead of the fill,
    would show a winner that never was."""
    day = DAYS[62]
    rnds = calendar.rounds_for(day)
    name = TICKERS[0]
    closes = [101.0, 110.0, 104.0, 108.0, 103.0, 107.0, 106.0]
    rows = []
    for r, c in zip(rnds, closes):
        end = r["execution"] + pd.Timedelta(hours=1) if r["round"] < 7 else calendar.at(day, calendar.session_close(day))
        for t in TICKERS:
            px = c if t == name else 50.0
            rows.append({"ticker": t, "start": r["execution"], "end": end,
                         "open": px, "high": px, "low": px, "close": px})
    # Last session's closes, far above: they must never count as marks after the fill.
    prev = DAYS[61]
    for t in TICKERS:
        rows.append({"ticker": t, "start": calendar.at(prev, calendar.SESSION_OPEN),
                     "end": calendar.at(prev, calendar.session_close(prev)),
                     "open": 500.0, "high": 500.0, "low": 500.0, "close": 500.0})
    bars = pd.DataFrame(rows)
    ex = {(day, r["round"]): {name: o} for r, o in zip(rnds, [100.0, 102.0, 109.0, 104.0, 108.0, 103.0, 106.0])}
    m = _market([prev, day], exec_px=ex, info_bars=bars)
    j = J.Journal()
    plan = {1: {name: 0.2}, 4: {name: 0.3}, 6: {name: 0.0}}   # buy, add, sell out

    def strategy(ctx):
        j.open_round(ctx, 1)
        if name in j.positions:
            strategy.at[ctx.round] = j.name_fields(real=True)[name]
        w = plan.get(ctx.round)
        out = None if w is None else {t: w.get(t, 0.0) for t in TICKERS}
        j.close_round([], out)
        return out

    strategy.at = {}

    res = sim.run(strategy, m, day, 1)
    # Bought at 100 at 09:30. By 12:25 the bars closed since are 101 (10:30), 110 (11:30).
    got = strategy.at[4]
    assert got["entry_price"] == 100.0
    assert got["gain_since_entry"] == pytest.approx(0.10) and got["peak_gain_since_entry"] == pytest.approx(0.10)
    # Added at 104 at 12:30; by 13:25, 104 (12:30) is the last close and 110 the peak.
    sh1 = 0.2 * 1e6 / 100.0
    nav4 = float(res.ledger.iloc[2]["cash"]) + sh1 * 104.0
    sh2 = 0.3 * nav4 / 104.0
    cost = (sh1 * 100.0 + (sh2 - sh1) * 104.0) / sh2
    got = strategy.at[5]
    assert got["entry_price"] == round(cost, 2)   # shown to the cent; kept exact below
    assert got["gain_since_entry"] == pytest.approx(104.0 / cost - 1, abs=1e-4)
    assert got["peak_gain_since_entry"] == pytest.approx(110.0 / cost - 1, abs=1e-4)
    closed = j.closed[-1]
    assert closed["entry_price"] == pytest.approx(cost, rel=1e-12)
    assert closed["exit_price"] == 103.0 and closed["gain"] == pytest.approx(103.0 / cost - 1)
    assert closed["peak_gain"] == pytest.approx(110.0 / cost - 1)
    assert name not in j.positions
    assert J.verify(j, J.sim_fills(res, m), closes=_closes(m)) == []


def test_a_desk_state_from_before_the_journal_says_its_entries_are_unknown_rather_than_inventing_them():
    """A phase begun on the old state (a list of short notes) and resumed on this code has
    a book with no journal behind it. Entries made up from the first price seen would
    read as gains the book never had; adopted, they are flagged estimated, and the
    journal says why."""
    m = _world()
    d = Desk(RuleBrain())
    sim.run(d, m, START, 1)
    old = d.state()
    old["journal"] = [{"day": 1, "role": "entry", "decision": {}, "why": "rule"}]
    again = Desk(RuleBrain())
    again.restore(json.loads(json.dumps(old)), m.tickers)
    r = calendar.rounds_for(DAYS[63])[0]
    held = {t: 100.0 for t in TICKERS}
    again(sim.RoundContext(DAYS[63], 1, r["deadline"], r["execution"], held, 1000.0, m))
    assert [i["kind"] for i in again.journal.issues] == ["no_journal"]
    fields = again.journal.name_fields(real=False)
    assert len(fields) == 30 and all(f["entry_estimated"] for f in fields.values())


# ----------------------------------------------------------------------------- the budget

def _worst_round(code, i, rnd):
    """A decision at every lever's longest: what no real round would carry, all at once."""
    long = ("x" * 1499) + "."
    if rnd == 1 and i == 0:
        return {"role": "entry", "brain": "claude", "source": "brain", "reason": None, "same_as_rule": False,
                "decision": {"shape": "risk_parity", "views": "strong", "exposure": 0.95,
                             "avoid": [{"name": code(t), "signal": "other", "why": "y" * 300} for t in TICKERS[:8]],
                             "rationale": long}}
    if rnd == 1:
        return {"role": "review", "brain": "claude", "source": "fallback", "reason": "TimeoutError: " + "z" * 300,
                "same_as_rule": False,
                "decision": {"action": "rebalance", "exposure": 0.6, "reason": "vol_change",
                             "exit": [code(t) for t in TICKERS[:8]], "rationale": long}}
    return {"role": "event", "brain": "claude", "source": "brain", "reason": None, "same_as_rule": False,
            "decision": {"calls": [{"name": code(t), "action": "exit" if k % 2 else "hold", "reason": "r" * 600}
                                   for k, t in enumerate(TICKERS)]}}


@pytest.mark.parametrize("real", [False, True])
def test_memory_stays_under_its_budget_at_every_round_of_a_105_round_phase_at_its_worst(real):
    """Unbounded, 105 rounds of reasons would grow the prompt past what a round's budget
    can send and cost more each round of the phase. Every round here carries the longest
    decision each role may give, and trades, and the block must stay under
    MEMORY_MAX_CHARS while keeping the latest rounds in full and every earlier day
    accounted for."""
    from icaif.agents.observe import Anonymizer

    m = _market(DAYS, info_bars=_bars(DAYS, 8))
    anon = Anonymizer(TICKERS, None if real else 5)
    j, sizes, books = J.Journal(), [], ({t: 1 / 30 for t in TICKERS}, {t: 0.02 for t in TICKERS})
    state = {"day": None, "n": 0, "i": 0}

    def strategy(ctx):
        if ctx.day != state["day"]:
            state["day"], state["n"] = ctx.day, state["n"] + 1
        j.open_round(ctx, state["n"])
        view = j.memory(anon.code, real=real)
        sizes.append(J.size(view))
        if state["i"] >= J.RECENT_ROUNDS:
            assert len(view["rounds"]) == J.RECENT_ROUNDS
            assert all(r.get("decided") for r in view["rounds"])
            covered = set()
            for line in view["earlier_days"]:
                a, b = J._span(line)
                covered |= set(range(a, b + 1))
            first_recent = j.rounds[-1 - J.RECENT_ROUNDS]["day"]
            assert covered >= set(range(1, first_recent)) or \
                "earlier days dropped" in view.get("trimmed_to_fit", [])
        rec = _worst_round(anon.code, state["i"], ctx.round)
        w = books[state["i"] % 2]
        j.close_round([rec], w)
        state["i"] += 1
        return w

    sim.run(strategy, m, START, 15)
    assert len(sizes) == 105
    assert max(sizes) <= J.MEMORY_MAX_CHARS, f"max {max(sizes)}"


# ----------------------------------------------------------------------------- the dry phase

def _norm(out: Path) -> dict:
    st = json.loads((out / "state.json").read_text())
    for d in st["desks"].values():
        for e in d["log"]:
            e.pop("latency_s", None)
    return st


def _full_phase(world, out: Path) -> list[dict]:
    c = JW.cfg(world, out)
    return [JW.play(world, c, row) for row in JW.rows(world)]


@pytest.fixture(scope="module")
def unbroken(tmp_path_factory):
    """The two-day dry phase run straight through: where every restart must end up."""
    tmp = tmp_path_factory.mktemp("unbroken")
    with pytest.MonkeyPatch.context() as mp:
        _full_phase(JW.setup(tmp, mp.setattr), tmp / "test")
    return _norm(tmp / "test")


def _check_journals(out: Path) -> None:
    st = runner.State(out)
    for name, paper in (("rule", "submitted"), ("agent", "agent")):
        j = J.Journal.from_json(st.data["desks"][name]["journal"])
        assert J.verify(j, J.paper_fills(st.data["paper"][paper])) == [], name
        disk = json.loads((out / "journal" / f"{name}.json").read_text())
        assert disk["journal"] == st.data["desks"][name]["journal"]


def test_every_journal_entry_agrees_with_its_paper_book_through_a_two_day_dry_phase(monkeypatch, tmp_path):
    """Two desks, two books: the rule's submitted book and the shadow's paper book, each
    with its own journal. Each is rebuilt from its own paper ledger and must agree with
    it entry for entry, and the copy on disk must be the committed one."""
    world = JW.setup(tmp_path, monkeypatch.setattr)
    recs = _full_phase(world, tmp_path / "test")
    assert all(r["errors"] == [] for r in recs) and not any("journal" in w for r in recs for w in r["warnings"])
    _check_journals(tmp_path / "test")
    st = runner.State(tmp_path / "test")
    shadow = J.Journal.from_json(st.data["desks"]["agent"]["journal"])
    assert [d["role"] for e in shadow.rounds for d in e["decisions"]].count("review") == 1
    assert {c["ticker"] for c in shadow.closed} >= {"AAPL"} and len(shadow.rounds) == 14
    # Each held name's weight, and the cash, add up to the book the journal marked.
    nav = shadow.nav()
    assert sum(p["weight"] for p in shadow.positions.values()) + shadow.book["cash"] / nav == pytest.approx(1.0)
    disk = json.loads((tmp_path / "test" / "journal" / "agent.json").read_text())
    assert [x["key"] for x in disk["pnl_since"]] == [e["key"] for e in shadow.rounds]
    assert disk["pnl_since"][-1]["pnl_usd"] == pytest.approx(0.0, abs=1e-6)


def test_what_the_shadow_was_shown_is_the_book_its_paper_ledger_held(monkeypatch, tmp_path):
    """The other half of the gate: a reason is only as good as the memory it was given.
    At every round a role was asked, the memory's book and held names must be the paper
    book's as that round began."""
    world = JW.setup(tmp_path, monkeypatch.setattr)
    c = JW.cfg(world, tmp_path / "test")
    b = JW.brain()
    asked = 0
    for row in JW.rows(world):
        n_seen = len(b.seen)
        JW.play(world, c, row, b)
        # The round settled the earlier orders before it decided, and its own order is
        # still pending: the paper book as saved is the book the round read.
        book = P.PaperBook.from_json(runner.State(c.out).data["paper"]["agent"])
        held = {t for t, s in book.shares.items() if s > P.SHARE_EPS}
        for _, p in b.seen[n_seen:]:
            asked += 1
            shown = {r["name"] for r in p["names"] if "entry_day" in r}
            weighted = {r["name"] for r in p["names"] if (r.get("weight_now") or 0) > 0}
            assert shown == held == weighted and p["memory"]["book"]["names_held"] == len(held)
    assert asked >= 3


def test_a_runner_killed_between_rounds_resumes_with_the_journals_and_books_of_one_that_never_stopped(
        monkeypatch, tmp_path, unbroken):
    """The scheduler dies (a reboot, a closed lid) and is started again. Each round runs
    from what the last one committed, so the restarted phase must end with the same
    journals, books and entry as one that never stopped."""
    world = JW.setup(tmp_path, monkeypatch.setattr)

    class Killed(BaseException):
        pass

    def spawner(c, die_after):
        ran = []

        def spawn(argv, timeout):
            rid = argv[argv.index("--round-id") + 1]
            if len(ran) == die_after:
                raise Killed()
            ran.append(rid)
            JW.play(world, c, next(r for r in JW.rows(world) if r["id"] == rid))
            from icaif.watchdog import Outcome
            return Outcome(True, 0, False, 0.1, "", "")
        return spawn

    c = JW.cfg(world, tmp_path / "again")
    read = lambda: (world["sched"], 0.0)  # noqa: E731
    with pytest.raises(Killed):
        runner.run_phase(c, read, fast=True, spawn=spawner(c, 5))
    assert len(runner.State(c.out).data["rounds"]) == 5
    with pytest.raises(Killed):
        runner.run_phase(c, read, fast=True, spawn=spawner(c, 4))
    runner.run_phase(c, read, fast=True, spawn=spawner(c, 99))
    assert _norm(tmp_path / "again") == unbroken
    _check_journals(tmp_path / "again")


@pytest.mark.parametrize("name,nth", [("state.json", 3), ("state.json", 8), ("decision.json", 1),
                                      ("agent.json", 6), ("rounds.jsonl", 2)])
def test_a_worker_killed_in_the_middle_of_a_write_leaves_the_last_commit_and_its_retry_matches_an_unbroken_run(
        monkeypatch, tmp_path, unbroken, name, nth):
    """A kill mid-write must leave every file as it was, never half of it: a torn
    state.json reads as "no entry yet", and a torn decision.json would be re-used and
    uploaded as it lay. The round never committed, so the scheduler's retry runs it
    again, and the phase must end exactly where an unbroken one does."""
    world = JW.setup(tmp_path, monkeypatch.setattr)

    class Killed(BaseException):
        pass

    real, seen = runner.write_atomic, {"n": 0}

    def write(path, text, mode=0o644):
        if Path(path).name == name:
            seen["n"] += 1
            if seen["n"] == nth:
                Path(path).with_name(f".{Path(path).name}.tmp").write_text(text[: len(text) // 2])
                raise Killed()
        return real(path, text, mode)

    monkeypatch.setattr(runner, "write_atomic", write)
    c = JW.cfg(world, tmp_path / "again")
    killed = 0
    for row in JW.rows(world):
        try:
            JW.play(world, c, row)
        except Killed:
            killed += 1
            before = runner.State(c.out)   # the last commit, whole
            if row["id"] not in before.data["rounds"]:
                JW.play(world, c, row)      # the retry
    assert killed == 1
    assert _norm(tmp_path / "again") == unbroken
    _check_journals(tmp_path / "again")
    listed = [x["round_id"] for x in runner.read_lines(tmp_path / "again" / "rounds.jsonl")]
    assert sorted(set(listed)) == sorted(r["id"] for r in JW.rows(world))   # none lost


def test_a_worker_sigkilled_mid_write_in_its_own_process_recovers_on_retry(monkeypatch, tmp_path, unbroken):
    """The real kill: SIGKILL, so no `finally`, no flush, no lock release but the OS's.
    Run as the scheduler runs a round, in its own process, killed halfway through the
    temp of the round's state.json and, on another round, of its decision.json."""
    world = JW.setup(tmp_path, monkeypatch.setattr)
    c = JW.cfg(world, tmp_path / "test")
    rows = JW.rows(world)
    kills = {rows[0]["id"]: "decision.json:1", rows[2]["id"]: "state.json:1"}
    for row in rows:
        if row["id"] in kills:
            before = (c.out / "state.json").read_text() if (c.out / "state.json").exists() else None
            p = subprocess.run([sys.executable, "-m", "tests.journal_world", "--out", str(tmp_path),
                                "--round-id", row["id"], "--kill-at", kills[row["id"]]],
                               cwd=ROOT, capture_output=True, text=True, timeout=120)
            assert p.returncode == -9, p.stderr[-2000:]
            after = (c.out / "state.json").read_text() if (c.out / "state.json").exists() else None
            assert after == before
            assert not (c.out / row["id"] / "decision.json").exists() or row["id"] != rows[0]["id"]
        JW.play(world, c, row)
    assert _norm(c.out) == unbroken
    _check_journals(c.out)


def test_a_torn_rounds_line_from_an_old_append_is_dropped_and_the_file_rewritten_whole(monkeypatch, tmp_path):
    world = JW.setup(tmp_path, monkeypatch.setattr)
    c = JW.cfg(world, tmp_path / "test")
    rows = JW.rows(world)
    JW.play(world, c, rows[0])
    with open(c.out / "rounds.jsonl", "a") as f:
        f.write('{"round_id": "torn", "outco')
    JW.play(world, c, rows[1])
    lines = (c.out / "rounds.jsonl").read_text().splitlines()
    assert [json.loads(x)["round_id"] for x in lines] == [rows[0]["id"], rows[1]["id"]]


def test_an_atomic_write_cut_short_leaves_the_old_file_whole(tmp_path):
    path = tmp_path / "state.json"
    runner.write_atomic(path, "old")

    class Boom(BaseException):
        pass

    class Half:
        def __init__(self, f):
            self.f = f

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.f.close()

        def fileno(self):
            return self.f.fileno()

        def write(self, text):
            self.f.write(text[:2])
            raise Boom()

    real = os.fdopen
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "fdopen", lambda fd, *a, **k: Half(real(fd, *a, **k)))
        with pytest.raises(Boom):
            runner.write_atomic(path, "new content")
    assert path.read_text() == "old"


# ----------------------------------------------------------------------------- the server

class Server(T.FakeSession):
    """A server that executes an upload at the round's 09:30 opens, `slip` away from the
    Yahoo opens our journal prices it at, and can lose a name (`drop`)."""

    def __init__(self, world, slip=0.0, drop=None):
        super().__init__()
        self.world, self.slip, self.drop = world, slip, drop
        self.cash, self.hold = 1_000_000.0, {}

    def portfolio(self, phase):
        return {"cash": self.cash, "positions": dict(self.hold)}

    def decision(self, path):
        out = super().decision(path)
        sent = json.loads(Path(path).read_text())
        row = next(r for r in self.world["sched"]["rounds"] if r["id"] == sent["round_id"])
        opens = runner.live.fills(self.world["bars30"]).loc[runner.et(row["execution_time"])]
        px = opens * (1 + self.slip)
        nav = self.cash + sum(s * px[t] for t, s in self.hold.items())
        new = {t: float(w) * nav / px[t] for t, w in sent["weights"].items() if float(w) > 0 and t != self.drop}
        moved = sum(abs(new.get(t, 0) - self.hold.get(t, 0)) * px[t] for t in set(new) | set(self.hold))
        self.cash -= sum((new.get(t, 0) - self.hold.get(t, 0)) * px[t] for t in set(new) | set(self.hold))
        self.cash -= sim.FEE_RATE * moved
        self.hold = new
        return out


def _live_phase(monkeypatch, tmp_path, server_kw, rounds=4, arm=True):
    world = JW.setup(tmp_path, monkeypatch.setattr)
    world["sched"] = runner.rehearsal_schedule(list(JW.DAYS), "validation")
    if arm:
        T._arm()
    server = Server(world, **server_kw)
    c = runner.Config(phase="validation", shadow="rule", scoring=False, live=True,
                      out=tmp_path / "validation", window_days=2)
    recs = []
    for row in JW.rows(world)[:rounds]:
        world["now"] = runner.et(row["deadline"]) - pd.Timedelta(minutes=12)
        d = JW.doors(world)
        d.session = lambda: server
        recs.append(runner.run_round(c, row, d))
    return world, server, c, recs


def test_a_servers_fill_within_vendor_noise_of_the_expectation_is_no_issue(monkeypatch, tmp_path):
    """The organizers fill at Alpaca's opens and our journal expects Yahoo's (p99 21 bps
    apart). A tolerance tight enough to flag that would cry wolf every entry, and the one
    disagreement that matters would drown in it."""
    _, server, c, recs = _live_phase(monkeypatch, tmp_path, {"slip": 3e-4})
    j = J.Journal.from_json(runner.State(c.out).data["desks"]["rule"]["journal"])
    assert j.issues == [] and j.rounds[0]["fill"]["matched"] == "within tolerance"
    assert {t: s for t, s in j.book["shares"].items() if s > 0} == server.hold
    assert j.book["cash"] == server.cash


def test_the_journal_records_a_server_book_that_disagrees_and_never_overrides_it(monkeypatch, tmp_path):
    """The server's book is what the competition scores. A journal that kept its own
    expectation would show the agent a name the book does not hold; one that quietly
    took the server's numbers would hide a fill that went wrong. The book stands, the
    journal adopts it, and the disagreement is an issue every later round can read."""
    _, server, c, recs = _live_phase(monkeypatch, tmp_path, {"drop": "AAPL"}, rounds=8)
    j = J.Journal.from_json(runner.State(c.out).data["desks"]["rule"]["journal"])
    assert [i["kind"] for i in j.issues] == ["book_differs"] and j.issues[0]["names"] == ["AAPL"]
    assert any("book_differs" in w for w in recs[1]["warnings"])
    assert {t: s for t, s in j.book["shares"].items() if s > 0} == server.hold
    assert "AAPL" not in j.positions and recs[1]["book"]["names_held"] == sorted(server.hold)
    assert all(not any("book_differs" in w for w in r["warnings"]) for r in recs[2:])
    review = next(r for r in recs if r["round_id"].endswith("10-09-r1"))
    assert review["rule"]["roles"][0]["role"] == "review"   # decided on the server's book


def test_an_order_that_was_never_uploaded_is_not_expected_to_fill(monkeypatch, tmp_path):
    """Not armed, the rule's entry stays on disk. A journal that expected it anyway would
    flag the unchanged book next round as a fill that failed."""
    _, server, c, recs = _live_phase(monkeypatch, tmp_path, {}, rounds=3, arm=False)
    j = J.Journal.from_json(runner.State(c.out).data["desks"]["rule"]["journal"])
    assert server.calls == [] and j.issues == [] and j.orders == []
    assert j.rounds[0]["order"]["status"] == "not-armed"
    view = j.memory(lambda t: t, real=True)
    assert view["rounds"][0]["submitted"].startswith("no: not-armed")


def test_no_journal_or_record_holds_the_team_token(monkeypatch, tmp_path):
    """decision.json carries the one-time team token, which can never be reset. The
    journal records what was submitted, never the file, so the token stays in the one
    place that must have it."""
    _, server, c, recs = _live_phase(monkeypatch, tmp_path, {}, rounds=8)
    token = server.creds["team_token"]
    hits = [p for p in c.out.rglob("*") if p.is_file() and p.name != "decision.json"
            and token in p.read_text(errors="ignore")]
    assert hits == [] and (c.out / "journal" / "rule.json").exists()
    assert token in (c.out / recs[0]["round_id"] / "decision.json").read_text()
