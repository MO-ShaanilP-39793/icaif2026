"""The live runner, offline: a synthetic market, a fake kit session, a hand-moved clock."""

import json
import os
import stat
import sys
from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from icaif import calendar, data, kit, live, runner, sim
from icaif import portfolio as P
from icaif import quant_strategies as qs
from icaif.agents.schemas import EntryDecision, Exclusion
from tests.test_agents import Failing, Scripted

TICKERS = sorted(data.load_universe())
DAY1, DAY2 = date(2026, 10, 8), date(2026, 10, 9)
END = pd.Timestamp("2026-10-09")


def _daily(n=800, seed=0):
    rng = np.random.default_rng(seed)
    dates = live.sessions(END - pd.Timedelta(days=1400), END)[-n:]
    vol = np.linspace(0.008, 0.025, len(TICKERS))
    frames = []
    for i, t in enumerate(TICKERS):
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, vol[i], n)))
        frames.append(pd.DataFrame({"date": dates, "ticker": t, "open": close, "high": close * 1.004,
                                    "low": close * 0.996, "close": close, "adj_close": close,
                                    "volume": 1e6}))
    return pd.concat(frames, ignore_index=True)


def _bars30(daily, days, shock=None):
    """Flat 30m bars at each day's prior close; `shock` = (ticker, day, factor) from 10:30."""
    prev = daily.pivot(index="date", columns="ticker", values="close").shift(1)
    rows = []
    for d in days:
        px = prev.loc[pd.Timestamp(d)]
        t = calendar.at(d, calendar.SESSION_OPEN)
        while t < calendar.at(d, calendar.session_close(d)):
            for s in TICKERS:
                p = float(px[s])
                if shock and s == shock[0] and d == shock[1] and t >= calendar.at(d, calendar.ROUNDS[2][1]):
                    p *= shock[2]
                rows.append({"ticker": s, "start": t, "end": t + pd.Timedelta(minutes=30), "open": p,
                             "high": p, "low": p, "close": p, "volume": 1e5, "source": "yahoo_30m"})
            t += pd.Timedelta(minutes=30)
    return pd.DataFrame(rows)


@pytest.fixture
def world(monkeypatch, tmp_path):
    daily = _daily()
    state = {"daily": daily, "bars30": _bars30(daily, [DAY1, DAY2]), "now": None}
    monkeypatch.setattr(live, "yahoo_daily", lambda symbols, start: (state["daily"], []))
    monkeypatch.setattr(live, "yahoo_intraday", lambda *a, **k: state["bars30"])
    monkeypatch.setattr(runner, "ARM_FILE", tmp_path / "kit" / "ARMED.json")
    state["sched"] = runner.rehearsal_schedule([DAY1, DAY2], "test")
    state["tmp"] = tmp_path
    return state


def _row(world, day, n):
    return next(r for r in world["sched"]["rounds"] if r["day"] == str(day) and r["number"] == n)


def _cfg(world, **kw):
    kw.setdefault("phase", "test")
    kw.setdefault("shadow", "rule")
    kw.setdefault("scoring", False)
    return runner.Config(out=world["tmp"] / kw["phase"], window_days=2, **kw)


def _doors(world, brain=None, **kw):
    kw.setdefault("score", lambda *a: {"status": "off"})
    kw.setdefault("inputs", lambda cfg, mkt, r: ({}, {}))
    return runner.Doors(now=lambda: world["now"],
                        brain=(lambda cfg, st: brain) if brain is not None else runner.make_brain, **kw)


def _play(world, cfg, doors, day, n, minutes_before=12):
    row = _row(world, day, n)
    world["now"] = runner.et(row["deadline"]) - pd.Timedelta(minutes=minutes_before)
    return runner.run_round(cfg, row, doors)


def _written(rec):
    return json.loads(open(rec["submitted"]["file"]).read(), parse_float=Decimal)


# ----------------------------------------------------------------------------- the rule's book

def test_round_one_from_cash_submits_the_backtested_rule_desks_book_and_the_kit_accepts_it(world):
    """The dry run used to print the compiler's top-10 default, a book the backtests had
    already ranked below the hold. What goes in is the rule desk's: risk parity at the
    regime-blended exposure, exactly as the backtest candidate buys it on the same data."""
    cfg = _cfg(world)
    rec = _play(world, cfg, _doors(world), DAY1, 1)
    sub = rec["submitted"]
    assert sub["action"] == "trade" and sub["source"] == "rule" and sub["upload"]["status"] == "dry-run"
    written = _written(rec)
    kit.validate_weights(written["weights"])
    assert written["round_id"].startswith("dryrun-") and kit.contracts.is_placeholder(written["team_token"])

    daily = world["daily"][world["daily"]["date"] <= pd.Timestamp("2026-10-07")]
    mkt = live.market(daily, None, live.market_days(daily, pd.Timestamp("2026-10-07")))
    r = calendar.rounds_for(DAY1)[0]
    ctx = sim.RoundContext(DAY1, 1, r["deadline"], r["execution"], {}, sim.INITIAL_NAV, mkt)
    want = qs.QuantBook("risk_parity", qs.Regime(), qs.ENTRY_ONLY)(ctx)
    assert {t: float(w) for t, w in written["weights"].items()} == want
    p = rec["rule"]["p_turbulent_next"]
    assert sum(want.values()) == pytest.approx(0.85 * (1 - p) + 0.30 * p, abs=30 * 1e-6)
    assert runner.State(cfg.out).data["entry"]["status"] == "dry-run"


def test_after_the_entry_every_round_holds_and_no_decision_file_is_written(world):
    """The kit re-sizes every name to its target at the fill, so a hold uploaded as
    weights is a trade of every name's drift, fee and turnover rank included. A hold
    is written as hold.json, a name the kit's upload refuses."""
    cfg = _cfg(world)
    _play(world, cfg, _doors(world), DAY1, 1)
    rounds = [(DAY1, n) for n in range(2, 8)] + [(DAY2, 1), (DAY2, 2)]
    for day, n in rounds:
        rec = _play(world, cfg, _doors(world), day, n)
        assert rec["submitted"]["action"] == "hold" and rec["submitted"]["upload"]["status"] == "hold"
        rdir = cfg.out / rec["round_id"]
        assert not (rdir / "decision.json").exists() and (rdir / "hold.json").exists()
        assert rec["errors"] == []
    paper = P.PaperBook.from_json(runner.State(cfg.out).data["paper"]["submitted"])
    assert paper.pending == [] and sum(s > 0 for s in paper.shares.values()) == 30


def test_a_lost_state_file_over_a_held_book_never_buys_the_entry_again(world):
    """Rebuilt from nothing, the runner would see a fresh phase. The server's book still
    holds 30 names, and that alone must stop the rule from buying them again."""
    cfg = _cfg(world)
    _play(world, cfg, _doors(world), DAY1, 1)
    _play(world, cfg, _doors(world), DAY1, 2)
    held = P.PaperBook.from_json(runner.State(cfg.out).data["paper"]["submitted"]).book(TICKERS)
    (cfg.out / "state.json").unlink()
    rec = _play(world, cfg, _doors(world, book=lambda: held), DAY2, 1)
    assert rec["entered"] == {"value": True, "why": "the book holds shares"}
    assert rec["submitted"]["action"] == "hold"


@pytest.mark.parametrize("kw,ok,why", [
    (dict(target=None), False, "returned no trade"),
    (dict(current=pd.Series(np.nan, index=TICKERS)), False, "cannot be valued"),
    (dict(target={t: 0.03 for t in TICKERS}, current=pd.Series(0.03, index=TICKERS)), False, "as it stands"),
    (dict(entered=True), False, "trades once"),
    (dict(book_all_cash=False), False, "holds shares"),
    (dict(round_no=3), False, "only at round 1"),
    (dict(source="agent", target={t: 0.0301 for t in TICKERS}, current=pd.Series(0.03, index=TICKERS)),
     False, "drift"),
    (dict(), True, "trade"),
    (dict(source="agent", entered=True, book_all_cash=False, round_no=4,
          target={**{t: 0.03 for t in TICKERS}, "AAPL": 0.0}, current=pd.Series(0.03, index=TICKERS)),
     True, "trade"),
])
def test_the_guard_uploads_only_a_real_trade(kw, ok, why):
    args = dict(target={t: 0.025 for t in TICKERS}, current=pd.Series(0.0, index=TICKERS),
                source="rule", entered=False, book_all_cash=True, round_no=1)
    args.update(kw)
    target, current = args.pop("target"), args.pop("current")
    got, reason = runner.guard(target, current, **args)
    assert got is ok and why in reason


# ----------------------------------------------------------------------------- the upload gate

class FakeSession:
    def __init__(self, result=None, raises=None, holds=None):
        self.result = result if result is not None else {"validation_status": "VALID",
                                                         "selection_status": "SELECTED"}
        self.raises, self.calls = raises, []
        self.creds = {"team_id": "team-1", "team_token": "tok-not-a-real-token"}
        self.holds = holds or {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def portfolio(self, phase):
        return {"cash": 1_000_000.0 - sum(self.holds.values()) * 100, "positions": dict(self.holds)}

    def decision(self, path):
        self.calls.append(path)
        if self.raises:
            raise self.raises
        return self.result


def _arm(phase="validation", submit="rule", expires="2026-10-10T00:00:00-04:00"):
    runner.ARM_FILE.parent.mkdir(parents=True, exist_ok=True)
    runner.ARM_FILE.write_text(json.dumps({"phase": phase, "submit": submit, "expires": expires,
                                           "armed_at": "test"}))


def _live(world):
    world["sched"] = runner.rehearsal_schedule([DAY1, DAY2], "validation")
    return _cfg(world, phase="validation", live=True)


@pytest.mark.parametrize("arm", [None, dict(phase="official"), dict(submit="agent"),
                                 dict(expires="2026-10-08T08:00:00-04:00")])
def test_nothing_uploads_without_the_owners_arm_for_this_phase_and_book(world, arm):
    """The owner approves uploads for one phase, one desk's book and until a time. A
    file armed for Official, for the agent or yesterday is not approval for this round."""
    if arm is not None:
        _arm(**arm)
    fake = FakeSession()
    cfg = _live(world)
    rec = _play(world, cfg, _doors(world, session=lambda: fake), DAY1, 1)
    assert rec["submitted"]["upload"]["status"] == "not-armed" and fake.calls == []
    assert runner.State(cfg.out).data["entry"]["status"] == "failed"


def test_an_armed_round_uploads_its_decision_file_once_through_the_kit(world):
    _arm()
    fake = FakeSession()
    cfg = _live(world)
    rec = _play(world, cfg, _doors(world, session=lambda: fake), DAY1, 1)
    path = cfg.out / rec["round_id"] / "decision.json"
    assert fake.calls == [str(path)] and rec["submitted"]["upload"]["status"] == "uploaded"
    written = _written(rec)
    assert written["round_id"] == rec["round_id"] and written["team_token"] == fake.creds["team_token"]
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600   # it carries the team token
    assert "tok-not-a-real-token" not in (cfg.out / "rounds.jsonl").read_text()
    rec2 = _play(world, cfg, _doors(world, session=lambda: fake), DAY1, 2)
    assert rec2["submitted"]["action"] == "hold" and len(fake.calls) == 1


def test_an_upload_too_close_to_the_deadline_is_not_started(world):
    _arm()
    fake = FakeSession()
    rec = _play(world, _live(world), _doors(world, session=lambda: fake), DAY1, 1, minutes_before=0.5)
    assert rec["submitted"]["upload"]["status"] == "too-late" and fake.calls == []


def test_an_ambiguous_upload_counts_as_entered_so_the_next_round_one_cannot_buy_again(world):
    """The kit could not say whether the upload committed. Re-entering on a maybe would
    buy the whole book a second time if it had."""
    from kit.original_client import AmbiguousSubmission

    _arm()
    fake = FakeSession(raises=AmbiguousSubmission("Submission creation may have committed."))
    cfg = _live(world)
    rec = _play(world, cfg, _doors(world, session=lambda: fake), DAY1, 1)
    assert rec["submitted"]["upload"]["status"] == "ambiguous"
    fake.raises = None
    rec = _play(world, cfg, _doors(world, session=lambda: fake), DAY2, 1)
    assert rec["submitted"]["action"] == "hold" and len(fake.calls) == 1


def test_an_invalid_entry_lets_the_next_round_one_enter_again(world):
    """An invalid upload holds the round and the book stays in cash; still counted as
    entered, the rule would sit in cash for the whole phase."""
    _arm()
    fake = FakeSession(result={"validation_status": "INVALID"})
    cfg = _live(world)
    _play(world, cfg, _doors(world, session=lambda: fake), DAY1, 1)
    assert runner.State(cfg.out).data["entry"]["status"] == "failed"
    fake.result = {"validation_status": "VALID"}
    rec = _play(world, cfg, _doors(world, session=lambda: fake), DAY2, 1)
    assert rec["submitted"]["action"] == "trade" and len(fake.calls) == 2


@pytest.mark.parametrize("result,want", [
    ({"status": "MISSED_DEADLINE"}, "failed"), ({"status": "SLOT_CONSUMED"}, "ambiguous"),
    ({"selection_status": "DUPLICATE"}, "ambiguous"), ({"validation_status": "INVALID"}, "failed"),
    ({"execution_status": "HELD"}, "failed"), ({"execution_status": "EXECUTED"}, "executed"),
    ({"validation_status": "VALID", "selection_status": "PENDING"}, "uploaded"), ({}, "uploaded"),
])
def test_a_receipt_reads_as_entered_unless_it_clearly_failed(result, want):
    """A failure read as success costs a day in cash; a success read as a failure buys
    the whole entry a second time."""
    assert runner.receipt_outcome(result) == want


def test_the_server_book_is_parsed_and_an_unknown_shape_submits_nothing(world):
    _arm()

    class Odd(FakeSession):
        def portfolio(self, phase):
            return {"balance": 1e6, "lots": []}

    fake = Odd()
    rec = _play(world, _live(world), _doors(world, session=lambda: fake), DAY1, 1)
    assert fake.calls == [] and any("PortfolioFormatError" in e for e in rec["errors"])
    assert rec["outcome"].startswith("no submission")


# ----------------------------------------------------------------------------- fallback chain

def test_a_failing_agent_hands_the_round_to_the_rule(world):
    """In agent mode the agent's book goes first; a crash in it must cost the round its
    judgement, never its book."""
    def broken(cfg, mkt, r):
        raise RuntimeError("inputs down")

    cfg = _cfg(world, submit="agent")
    rec = _play(world, cfg, _doors(world, brain=Scripted(), inputs=broken), DAY1, 1)
    assert rec["submitted"]["source"] == "rule" and rec["submitted"]["action"] == "trade"
    assert any("agent desk" in e for e in rec["errors"])


def test_a_working_agent_submits_its_own_book(world):
    pick = EntryDecision(shape="inverse_vol", views="none", exposure=0.5, avoid=[], rationale="calmer start")
    cfg = _cfg(world, submit="agent")
    rec = _play(world, cfg, _doors(world, brain=Scripted(entry=pick)), DAY1, 1)
    assert rec["submitted"]["source"] == "agent"
    assert sum(float(w) for w in _written(rec)["weights"].values()) == pytest.approx(0.5, abs=30e-6)


def test_with_no_prices_no_desk_answers_and_nothing_is_submitted(world, monkeypatch):
    def down(symbols, start):
        raise ConnectionError("yahoo is down")

    monkeypatch.setattr(live, "yahoo_daily", down)
    cfg = _cfg(world, submit="agent")
    rec = _play(world, cfg, _doors(world, brain=Scripted()), DAY1, 1)
    assert rec["submitted"]["source"] == "none" and rec["submitted"]["action"] == "hold"
    assert not (cfg.out / rec["round_id"] / "decision.json").exists()


# ----------------------------------------------------------------------------- the shadow

def test_the_shadow_is_logged_beside_the_submitted_book_and_trades_only_on_paper(world):
    pick = EntryDecision(shape="inverse_vol", views="none", exposure=0.5, rationale="earnings gap",
                         avoid=[Exclusion(name="TSLA", signal="earnings", why="reports tomorrow")])
    cfg = _cfg(world)
    rec = _play(world, cfg, _doors(world, brain=Scripted(entry=pick)), DAY1, 1)
    assert rec["submitted"]["source"] == "rule" and rec["rule"]["gross"] > 0.5
    sh = rec["shadow"]
    assert sh["action"] == "trade" and sh["gross"] == pytest.approx(0.5, abs=30e-6)
    assert "TSLA" not in sh["weights"] and sh["roles"][0]["source"] == "brain"
    calls = json.loads((cfg.out / rec["round_id"] / "shadow_calls.json").read_text())
    assert calls[0]["role"] == "entry" and "rule_proposal" in calls[0]["payload"]

    rec2 = _play(world, cfg, _doors(world, brain=Scripted(entry=pick)), DAY1, 2)
    paper = P.PaperBook.from_json(runner.State(cfg.out).data["paper"]["agent"])
    assert paper.pending == [] and paper.shares["TSLA"] == 0 and paper.shares["AAPL"] > 0
    assert rec2["shadow"]["action"] == "hold"


def test_a_failing_brain_shadows_as_the_rule_and_says_so(world):
    rec = _play(world, _cfg(world), _doors(world, brain=Failing()), DAY1, 1)
    assert rec["shadow"]["action"] == "trade"
    assert [x["source"] for x in rec["shadow"]["roles"]] == ["fallback"]
    assert rec["shadow"]["weights"] == rec["rule"]["weights"]


def test_a_spent_shadow_budget_stops_calling_the_model(world):
    cfg = _cfg(world, shadow="claude")
    st = runner.State(cfg.out)
    st.data["spent_usd"] = cfg.shadow_cost_cap
    st.save()
    assert isinstance(runner.make_brain(cfg, runner.State(cfg.out)), runner._Spent)


def test_a_hanging_scorer_is_killed_and_the_submitted_book_is_already_out(world, monkeypatch):
    """Scoring is the step that deadlocked. It runs after the upload, in a child the
    watchdog kills, so a hang costs the shadow its scores and nothing else."""
    hang = world["tmp"] / "hang.py"
    hang.write_text("import threading; threading.Event().wait()\n")
    monkeypatch.setattr(runner, "TOOL", hang)
    cfg = _cfg(world, scoring=True)
    cfg.scoring_timeout_s = 1.0
    doors = _doors(world)
    doors.score = runner.score_in_child
    rec = _play(world, cfg, doors, DAY1, 1)
    assert rec["submitted"]["action"] == "trade" and rec["submitted"]["upload"]["status"] == "dry-run"
    s = rec["shadow_scores"]
    assert s["status"] == "timeout" and len(s["tries"]) == 2 and all(t["timed_out"] for t in s["tries"])
    assert rec["shadow"]["action"] == "trade"   # the agent still ran, without scores


def test_two_workers_cannot_hold_one_phase_at_once(world):
    """Two would each read "not entered" and each upload an entry."""
    cfg = _cfg(world)
    with runner.phase_lock(cfg.out):
        with pytest.raises(runner.RunnerError):
            with runner.phase_lock(cfg.out):
                pass


# ----------------------------------------------------------------------------- the scheduler

class Clock:
    def __init__(self, start):
        self.t = runner.et(start)

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += pd.Timedelta(seconds=s)


def _spawner(cfg, clock, log, die=()):
    def spawn(argv, timeout):
        rid = argv[argv.index("--round-id") + 1]
        log.append((rid, clock(), argv))
        if rid not in die:
            st = runner.State(cfg.out)
            st.data["rounds"][rid] = {"outcome": "hold", "upload": "hold"}
            st.save()
        clock.sleep(30)
        from icaif.watchdog import Outcome
        return Outcome(rid not in die, 0 if rid not in die else None, rid in die, 30.0, "", "")
    return spawn


def test_the_scheduler_wakes_each_round_before_its_deadline_and_skips_a_cancelled_one(world):
    sched = runner.rehearsal_schedule([DAY1], "test")
    sched["rounds"][2]["status"] = "CANCELLED"
    cfg = _cfg(world)
    clock, log = Clock("2026-10-08 06:00"), []
    runner.run_phase(cfg, lambda: (sched, 0.0), sleep=clock.sleep, spawn=_spawner(cfg, clock, log),
                     clock=clock)
    ran = [rid for rid, _, _ in log]
    assert ran == [r["id"] for r in sched["rounds"] if r["status"] != "CANCELLED"]
    for rid, at, _ in log:
        dl = runner.et(next(r for r in sched["rounds"] if r["id"] == rid)["deadline"])
        assert dl - pd.Timedelta(minutes=12) <= at < dl - pd.Timedelta(minutes=10)


def test_a_deadline_moved_by_the_organizers_is_seen_before_the_wake(world):
    sched = runner.rehearsal_schedule([DAY1], "test")
    moved = json.loads(json.dumps(sched))
    moved["rounds"][0]["deadline"] = "2026-10-08T09:40:00-04:00"
    reads = []

    def read():
        reads.append(1)
        return (sched if len(reads) == 1 else moved), 0.0

    cfg = _cfg(world)
    clock, log = Clock("2026-10-08 07:00"), []
    runner.run_phase(cfg, read, sleep=clock.sleep, spawn=_spawner(cfg, clock, log), clock=clock)
    assert log[0][0].endswith("r1") and log[0][1] >= runner.et("2026-10-08 09:28")


def test_a_worker_that_dies_without_a_record_is_retried_once_then_marked_missed(world):
    sched = runner.rehearsal_schedule([DAY1], "test")
    cfg = _cfg(world)
    clock, log = Clock("2026-10-08 08:50"), []
    first = sched["rounds"][0]["id"]
    runner.run_phase(cfg, lambda: (sched, 0.0), sleep=clock.sleep,
                     spawn=_spawner(cfg, clock, log, die={first}), clock=clock)
    assert [rid for rid, _, _ in log].count(first) == 2
    assert runner.State(cfg.out).data["rounds"][first]["outcome"] == "missed: the worker died"
    assert len(log) == 8   # the other six rounds still ran


def test_a_fast_rehearsal_runs_every_round_with_its_clock_at_the_wake(world):
    sched = runner.rehearsal_schedule([DAY1], "test-fast")
    cfg = _cfg(world, phase="test-fast")
    clock, log = Clock("2026-10-08 06:00"), []
    runner.run_phase(cfg, lambda: (sched, 0.0), fast=True, spawn=_spawner(cfg, clock, log))
    for (rid, _, argv), row in zip(log, sched["rounds"]):
        assert runner.et(argv[argv.index("--as-of") + 1]) == runner.et(row["deadline"]) - pd.Timedelta(minutes=12)
    assert len(log) == 7


def test_a_rehearsals_rounds_cannot_name_a_real_one():
    """Its decision files are written by the same code as a live round's; their round
    ids must match no schedule the server holds, even before the dryrun- prefix."""
    sched = runner.rehearsal_schedule([DAY1], "rehearsal-2026-10-08")
    real = {r["id"] for r in runner.bundled_schedule()["rounds"]}
    assert not {r["id"] for r in sched["rounds"]} & real
    with pytest.raises(ValueError):
        runner.Config(phase="rehearsal-2026-10-08", live=True)


def test_a_sharp_move_today_wakes_the_shadows_analyst_while_the_submitted_book_holds(world):
    """Live, the trigger reads today's 60m bars against yesterday's close. Without them
    (or with today's bars read as a close) the move is invisible and the analyst never
    wakes in rounds 2-7; the rule desk, which holds through every event, must not trade."""
    from icaif.agents.schemas import EventDecision, NameCall

    world["bars30"] = _bars30(world["daily"], [DAY1, DAY2], shock=("AAPL", DAY1, 0.8))
    exits = lambda p: EventDecision(calls=[NameCall(name=t["name"], action="exit", reason="gap")  # noqa: E731
                                           for t in p["triggers"]])
    brain = Scripted(event=exits)
    cfg = _cfg(world)
    _play(world, cfg, _doors(world, brain=brain), DAY1, 1)
    r3 = _play(world, cfg, _doors(world, brain=brain), DAY1, 3)
    assert r3["shadow"]["roles"] == []          # the 10:30-11:30 hour has not ended by 11:13
    r4 = _play(world, cfg, _doors(world, brain=brain), DAY1, 4)
    roles = r4["shadow"]["roles"]
    assert [x["role"] for x in roles] == ["event"] and roles[0]["decision"]["calls"][0]["name"] == "AAPL"
    assert r4["submitted"]["action"] == "hold" and r4["rule"]["roles"][0]["role"] == "event"
    assert r4["shadow"]["action"] == "trade" and "AAPL" not in r4["shadow"]["weights"]
    r5 = _play(world, cfg, _doors(world, brain=brain), DAY1, 5)
    paper = P.PaperBook.from_json(runner.State(cfg.out).data["paper"]["agent"])
    assert paper.shares["AAPL"] == 0 and r5["shadow"]["roles"] == []   # fired once a day


def test_an_upload_whose_receipt_was_still_pending_counts_as_uploaded_and_is_reread(world):
    """The kit creates the submission, then polls for its receipt and raises if that is
    slow. Read as refused, the entry would count as failed while it executes; and a
    receipt that later says INVALID must free the next round 1 to enter."""
    from kit.original_client import AutomationError

    _arm()
    fake = FakeSession(raises=AutomationError("Submission 77 is pending. Resume fetch with this original ID."))
    fake.state = {"operations": {"decision:validation:validation-2026-10-08-r1": {"state": "submitted",
                                                                                   "submission_id": 77}}}
    fake.fetch = lambda sid: {"validation_status": "INVALID", "platform_submission_id": sid}
    cfg = _live(world)
    rec = _play(world, cfg, _doors(world, session=lambda: fake), DAY1, 1)
    assert rec["submitted"]["upload"]["status"] == "uploaded"
    entry = runner.State(cfg.out).data["entry"]
    assert entry["status"] == "uploaded" and entry["submission_id"] == 77
    rec = _play(world, cfg, _doors(world, session=lambda: fake), DAY1, 2)
    assert rec["submitted"]["action"] == "hold"       # still entered while pending
    fake.raises = None
    rec = _play(world, cfg, _doors(world, session=lambda: fake), DAY2, 1)
    assert any("uploaded -> failed" in w for w in rec["warnings"])
    assert rec["submitted"]["action"] == "trade" and len(fake.calls) == 2


def test_a_har_forecast_that_cannot_be_made_is_recorded_and_the_other_inputs_still_load(world, monkeypatch):
    """The agent's inputs load one by one. A HAR failure that raised through would take
    the scores, the calendar and the round's shadow with it, for want of one signal."""
    from types import SimpleNamespace

    r = runner.Round.from_row(_row(world, DAY1, 1))
    mkt = SimpleNamespace(tickers=TICKERS, days=[DAY1, DAY2])
    calls = []

    def broken(deadline, out_dir, tickers):
        calls.append((deadline, out_dir))
        raise live.LiveDataError("no completed 30m bars")

    monkeypatch.setattr(live, "vol_forecasts", broken)
    kw, meta = runner.agent_inputs(_cfg(world), mkt, r)
    assert "vol" not in kw and meta["vol"].startswith("LiveDataError")
    assert {"scores", "context", "earnings", "fomc", "filings"} <= set(meta)
    assert calls == [(r.deadline, _cfg(world).out / "vol" / str(DAY1))]

    monkeypatch.setattr(live, "vol_forecasts", lambda *a: "forecasts")
    kw, meta = runner.agent_inputs(_cfg(world), mkt, r)
    assert kw["vol"] == "forecasts" and meta["vol"] == "ok"
