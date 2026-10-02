"""Profit booking: the trim rule, its desk twin and the agent's trim lever (Roadmap step 5)."""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from icaif import calendar, quant_strategies as qs, sim, trim as TR
from icaif.agents import journal as J
from icaif.agents.brains import RuleBrain
from icaif.agents.desk import Desk, DeskConfig
from icaif.agents.schemas import EntryDecision, EventDecision, NameCall, ReviewDecision, Trim
from tests.test_agents import DAYS as A_DAYS, START as A_START, Scripted, _mkt
from tests.test_quant import _bars
from tests.test_signals import DAYS, START, _world
from tests.test_sim import TICKERS, _market

N = min(15, len(DAYS) - DAYS.index(START))
EAGER = TR.TrimSettings(a=0.0, b=0.5, fraction=0.5, estimator="trailing")


def _closes(m):
    return m.recent_closes(pd.Timestamp("2100-01-01", tz="UTC"), 10 ** 7)


# ----------------------------------------------------------------------------- the rule

def test_the_trim_rule_desk_trades_exactly_as_the_standalone_book_it_was_scored_as():
    """The selection scores `TrimmedRiskParity`, which is fast; the desk is what would
    trade. If a gain, a peak or a weight differed between them, the score would belong
    to a book the desk never holds, and nothing would look wrong in either."""
    m, kw = _world()
    rule = TR.TrimRule(EAGER)
    twin = TR.TrimmedRiskParity(rule, kw["vol"], window_days=N)
    want = sim.run(twin, m, START, N)
    d = Desk(RuleBrain(), DeskConfig(window_days=N), trim=rule, **kw)
    got = sim.run(d, m, START, N)
    pd.testing.assert_frame_equal(got.ledger, want.ledger)
    assert twin.trims and [(x["ticker"], x["day"]) for x in twin.trims] == \
        [(x["ticker"], x["day"]) for x in d.trims]
    assert J.verify(d.journal, J.sim_fills(got, m), closes=_closes(m)) == []
    reviews = [e for e in d.log if e["role"] == "review" and e["decision"]["trim"]]
    assert reviews and all(t["cause"] == "give_back" and t["why"].startswith("rule:")
                           for e in reviews for t in e["decision"]["trim"])


def test_without_a_trim_the_book_and_the_desk_are_the_candidate():
    """A trim rule that never fires must be the backtested candidate call for call, or a
    score difference would measure the plumbing rather than the trims."""
    m, kw = _world()
    never = TR.TrimRule(TR.TrimSettings(a=50.0, b=50.0, fraction=0.25, estimator="trailing"))
    ref = sim.run(qs.CANDIDATES["q_riskparity_entry_regime"](), m, START, N)
    for strat in (TR.TrimmedRiskParity(never, kw["vol"], N),
                  Desk(RuleBrain(), DeskConfig(window_days=N), trim=never, **kw)):
        pd.testing.assert_frame_equal(sim.run(strat, m, START, N).ledger, ref.ledger)


def _history(y: float, end=date(2026, 1, 30)) -> TR.GiveBack:
    rows = [{"start": end - pd.Timedelta(days=20), "end": end, "k": 5, "ticker": t,
             "zg": 3.0, "zdd": 3.0, "y": y} for t in TICKERS * 2]
    return TR.GiveBack(pd.DataFrame(rows))


def test_a_rule_whose_history_says_winners_recover_never_trims():
    """The expected give-back is minus the history's forward return: where winners that
    turned went on to rise, the rule must refuse to pay for a reversal that is not there."""
    m, kw = _world()
    ref = sim.run(qs.CANDIDATES["q_riskparity_entry_regime"](), m, START, N)
    s = TR.TrimSettings(a=0.0, b=0.5, fraction=0.5, estimator="expanding")
    recover = TR.TrimmedRiskParity(TR.TrimRule(s, _history(+0.3)), kw["vol"], N)
    pd.testing.assert_frame_equal(sim.run(recover, m, START, N).ledger, ref.ledger)
    giveback = TR.TrimmedRiskParity(TR.TrimRule(s, _history(-0.3)), kw["vol"], N)
    sim.run(giveback, m, START, N)
    assert giveback.trims and all(x["egb"] > TR.COST for x in giveback.trims)


def test_the_give_back_history_reads_no_window_that_had_not_ended_by_the_asking_window():
    """Built once from every window, the table holds each window's future. Rewrite every
    bar after the first window ends: its rows, and the estimate a window asking from
    after it reads, must not move; and that estimate reads no row of a window still open."""
    i = DAYS.index(START)
    s1, s2 = DAYS[i], DAYS[i + 5]   # two 5-day windows, back to back
    m0, kw0 = _world()
    m1, kw1 = _world(bars=_bars(DAYS, 8, shock_from=DAYS[i + 5]))
    a = TR.GiveBack.build(m0, kw0["vol"], [s1, s2], window_days=5)
    b = TR.GiveBack.build(m1, kw1["vol"], [s1, s2], window_days=5)
    first = lambda g: g.states[g.states["start"] == s1].reset_index(drop=True)  # noqa: E731
    assert len(first(a)) == 4 * len(TICKERS)
    pd.testing.assert_frame_equal(first(a), first(b))
    assert not a.states[a.states["start"] == s2].equals(b.states[b.states["start"] == s2])
    assert a.mu(-10, -10, s2) == b.mu(-10, -10, s2) == pytest.approx(first(a)["y"].mean())


def test_settings_off_the_levers_grid_are_refused():
    with pytest.raises(ValueError):
        TR.TrimSettings(a=1, b=1, fraction=0.3, estimator="trailing")
    with pytest.raises(ValueError):
        TR.TrimSettings(a=1, b=1, fraction=0.5, estimator="oracle")
    with pytest.raises(ValueError):
        TR.TrimRule(TR.TrimSettings(a=1, b=1, fraction=0.5, estimator="rolling3y"))


# ----------------------------------------------------------------------------- the lever

class Recorder:
    """The desk, and every book it asked for."""

    def __init__(self, desk):
        self.desk, self.sent = desk, []

    def __call__(self, ctx):
        w = self.desk(ctx)
        if w is not None:
            self.sent.append((ctx.day, ctx.round, w, self.desk._current.copy()))
        return w


def _biggest(p) -> str:
    return max(p["names"], key=lambda r: r["weight_now"] or 0)["name"]


def _trim_review(fraction="half", cause="give_back", names=None, action="hold"):
    def review(p):
        if p["clock"]["day"] != 2:
            return ReviewDecision.model_validate(p["rule_proposal"])
        picked = names(p) if names else [_biggest(p)]
        return ReviewDecision(action=action, exposure=None, reason=None if action != "rebalance"
                              else "vol_change", exit=[],
                              trim=[Trim(name=n, fraction=fraction, cause=cause, why="took a gain")
                                    for n in picked], rationale="book part of a winner")
    return review


def test_an_agents_trim_sells_its_fraction_keeps_the_rest_and_is_on_the_journal():
    m = _mkt()
    rec = Recorder(Desk(Scripted(review=_trim_review("half")), None))
    res = sim.run(rec, m, A_START, 3)
    (day, rnd, w, current), = [x for x in rec.sent if x[0] != A_START]
    t = max(current.index, key=lambda k: current[k])
    assert w[t] == pytest.approx(current[t] * 0.5, abs=1e-6)
    others = [k for k in current.index if k != t]
    assert all(w[k] == pytest.approx(current[k], abs=1e-6) for k in others)
    d = rec.desk
    assert [(x["ticker"], x["fraction"], x["role"]) for x in d.trims] == [(t, 0.5, "review")]
    assert J.verify(d.journal, J.sim_fills(res, m), closes=_closes(m)) == []
    assert d.journal.positions[t]["shares"] > 0
    memory = json.dumps(d.journal.memory(lambda x: x, real=True))
    assert f"trim {t}:half(give_back)" in memory


def test_a_trim_under_the_floor_is_a_hold_and_spends_nothing():
    """A quarter of a 1% position sells 0.25% of NAV: the fee and a turnover rank for a
    sale too small to change the book's risk."""
    small = EntryDecision(shape="risk_parity", views="none", exposure=0.30, avoid=[], rationale="x")
    rec = Recorder(Desk(Scripted(entry=small, review=_trim_review("quarter")), None))
    sim.run(rec, _mkt(), A_START, 3)
    assert len(rec.sent) == 1 and rec.desk.trims == []
    review = next(e for e in rec.desk.log if e["role"] == "review" and e["day"] == 2)
    assert review["source"] == "brain" and review["trims_done"] == []


def test_trims_past_the_windows_cap_or_with_a_rebalance_fall_back_to_the_rule():
    def top2(p):
        return [r["name"] for r in sorted(p["names"], key=lambda r: -(r["weight_now"] or 0))[:2]]
    for cfg, review in ((DeskConfig(max_trims=1), _trim_review("half", names=top2)),
                        (None, _trim_review("half", action="rebalance"))):
        rec = Recorder(Desk(Scripted(review=review), cfg))
        sim.run(rec, _mkt(), A_START, 3)
        assert len(rec.sent) == 1 and rec.desk.trims == []
        assert [e["source"] for e in rec.desk.log if e["role"] == "review" and e["day"] == 2] == ["fallback"]


def _filed(day, ticker):
    return pd.DataFrame({"ticker": [ticker], "accepted": [calendar.at(day, pd.Timestamp("07:00").time())],
                         "items": ["5.02"], "amended": [False]})


@pytest.mark.parametrize("answer, trades", [
    (lambda t, other: [NameCall(name=t, action="trim", fraction="half", cause="filing", reason="x")], True),
    (lambda t, other: [NameCall(name=t, action="trim", fraction=None, cause="filing", reason="x")], False),
    (lambda t, other: [NameCall(name=t, action="hold", fraction="half", cause=None, reason="x")], False),
    (lambda t, other: [NameCall(name=other, action="trim", fraction="half", cause="news", reason="x")], False),
])
def test_an_event_trim_needs_a_triggered_name_a_fraction_and_a_cause(answer, trades):
    """The analyst may trim only a name its trigger named, on the grid, with its cause;
    anything else is refused whole and the rule's hold stands."""
    day2 = A_DAYS[A_DAYS.index(A_START) + 1]
    t, other = TICKERS[3], TICKERS[4]
    b = Scripted(event=lambda p: EventDecision(calls=answer(t, other)))
    rec = Recorder(Desk(b, None, filings=_filed(day2, t)))
    sim.run(rec, _mkt(), A_START, 2)
    event = next(e for e in rec.desk.log if e["role"] == "event")
    assert (event["source"] == "brain") == trades
    assert [x["ticker"] for x in rec.desk.trims] == ([t] if trades else [])
    if trades:
        _, _, w, current = rec.sent[-1]
        assert w[t] == pytest.approx(current[t] * 0.5, abs=1e-6)


def test_a_desk_with_trims_restored_before_every_round_trades_as_one_that_never_stopped():
    """The trims spent and the names trimmed live in the desk's state: rebuilt without
    them, a restarted desk would spend the window's cap again and re-trim the same name."""
    m, kw = _world()
    rule = TR.TrimRule(EAGER)
    whole = Desk(RuleBrain(), DeskConfig(window_days=N), trim=rule, **kw)
    want = sim.run(whole, m, START, N)
    saved = {}

    def restarted(ctx):
        d = Desk(RuleBrain(), DeskConfig(window_days=N), trim=rule, **kw)
        d.restore(json.loads(saved["s"]) if saved else None, m.tickers)
        w = d(ctx)
        saved["s"] = json.dumps(d.state())
        restarted.desk = d
        return w

    got = sim.run(restarted, m, START, N)
    pd.testing.assert_frame_equal(got.ledger, want.ledger)
    assert restarted.desk.trims == whole.trims and whole.trims
