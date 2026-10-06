"""The v2 desk's rounds 2-7 and its reflection (icaif/agents/v2.py)."""

import json

import pandas as pd
import pytest

from icaif import calendar, sim
from icaif.agents.desk import EarningsCalendar
from icaif.agents.schemas import (AddLine, CutLine, Lesson, PMDecision, Reflection, TradeList,
                                  TriggerDecision, Trim)
from icaif.agents.triggers import EarningsHistory
from icaif.agents.v2 import V2Desk, V2Config
from tests.test_agents import DAYS, START, _mkt
from tests.test_quant import _bars
from tests.test_sim import TICKERS
from tests.test_v2 import HOLD, Script, _hold_weights

I0 = DAYS.index(START)
D2, D3 = DAYS[I0 + 1], DAYS[I0 + 2]
SHOCKED, REPORTER, FILER = TICKERS[4], TICKERS[3], TICKERS[2]


def _at(day, hhmm):
    return calendar.at(day, pd.Timestamp(hhmm).time())


def _shocked_bars():
    """SHOCKED falls 20% from D2's 10:30 bar: a move of many daily sigmas by round 3."""
    bars = _bars(DAYS, 8)
    hit = (bars["ticker"] == SHOCKED) & (bars["start"] >= _at(D2, "10:30"))
    bars.loc[hit, ["open", "high", "low", "close"]] *= 0.8
    return bars


def _desk(brain, m, events=None, filings=None, **cfg):
    kw = {}
    if events is not None:
        kw = dict(earnings=EarningsCalendar(events, m.days),
                  earnings_history=EarningsHistory.from_market(events, m))
    return V2Desk({"quick": brain, "deep": brain}, V2Config(**cfg), filings=filings, **kw)


def _tl(adds=(), cuts=(), trims=(), exposure=None):
    return TradeList(adds=[AddLine(name=n, weight=w, why="x") for n, w in adds],
                     cuts=[CutLine(name=n, why="x") for n in cuts],
                     trims=[Trim(name=n, fraction=f, cause="volatility", why="x") for n, f in trims],
                     target_exposure=exposure, rationale="r")


def _trade(tl):
    return TriggerDecision(action="trade", trade_list=tl, rationale="act")


def _asked(brain, role):
    return [p for r, _, p, _ in brain.seen if r == f"v2_{role}"]


# ----------------------------------------------------------------------------- triggers

def test_a_sharp_move_wakes_the_event_analyst_and_the_pm_for_that_name_alone():
    """The trigger carries its tag, so the PM reads that the fall has already happened
    and a sale now sells after it; the trim trades that name and nothing else."""
    m = _mkt(_shocked_bars())
    b = Script(event_pm=lambda p: _trade(_tl(trims=[(p["triggers"][0]["name"], "half")])))
    d = _desk(b, m)
    sent = []

    def watch(ctx):
        w = d(ctx)
        sent.append((ctx.day, ctx.round, d._current.copy(), w))
        return w

    sim.run(watch, m, START, 3)
    first = _asked(b, "event")[0]
    assert [x["name"] for x in first["triggers"]] == [SHOCKED]
    tag = first["triggers"][0]["tag"]
    assert tag["kind"] == "move" and tag["status"] == "already_reacted"
    assert tag["move_since_close_sigmas"] <= -3
    _, _, cur, w = next(x for x in sent if x[:2] == (D2, first["clock"]["round"]))
    assert w[SHOCKED] == pytest.approx(cur[SHOCKED] / 2, abs=2e-6)
    assert all(abs(w[t] - cur[t]) <= 2e-6 for t in TICKERS if t != SHOCKED)
    assert [e["decision"]["lines"] for e in d.log if e["role"] == "event_pm"] == [[SHOCKED]]
    assert len(_asked(b, "event")) == 1            # once a day a name, not every round after


@pytest.mark.parametrize("bad", ["other_name", "exposure"])
def test_a_trigger_list_naming_another_name_or_an_exposure_is_refused(bad):
    """A trigger is about its names; a list that reaches further is a morning decision
    taken at 11:25 without the analysts, the debate or the risk manager."""
    m = _mkt(_shocked_bars())
    tl = (_tl(trims=[(SHOCKED, "half"), (TICKERS[9], "half")]) if bad == "other_name"
          else _tl(trims=[(SHOCKED, "half")], exposure=0.7))
    d = _desk(Script(event_pm=lambda p: _trade(tl)), m)
    res = sim.run(d, m, START, 2)
    pd.testing.assert_frame_equal(res.ledger, sim.run(HOLD(), m, START, 2).ledger)
    assert d.fallbacks["event_pm:failed"] == 1


def test_results_at_the_next_open_wake_the_pm_at_the_last_round_with_an_upcoming_tag():
    m = _mkt()
    events = pd.DataFrame({"ticker": [REPORTER], "accepted": [_at(D2, "16:30")]})
    b = Script()
    sim.run(_desk(b, m, events=events), m, START, 3)
    asked = _asked(b, "event")
    assert [(p["clock"]["day"], p["clock"]["round"]) for p in asked] == [(2, 7)]
    tag = asked[0]["triggers"][0]["tag"]
    assert tag["kind"] == "earnings" and tag["status"] == "upcoming" and tag["sessions_to_reaction"] == 1


def test_a_new_8k_wakes_its_name_once_with_how_long_the_market_has_traded_on_it():
    m = _mkt()
    filings = pd.DataFrame({"ticker": [FILER], "accepted": [_at(D2, "10:00")], "items": ["5.02"],
                            "amended": [False]})
    b = Script()
    sim.run(_desk(b, m, filings=filings), m, START, 2)
    asked = _asked(b, "event")
    assert [(p["clock"]["day"], p["clock"]["round"]) for p in asked] == [(2, 2)]
    tag = asked[0]["triggers"][0]["tag"]
    assert tag["kind"] == "8k" and tag["status"] == "already_reacted" and tag["minutes_traded"] == 30


def test_a_failed_trigger_pm_trades_nothing_and_is_counted():
    m = _mkt(_shocked_bars())

    def down(_):
        from icaif.agents.brains import BrainError
        raise BrainError("down")

    d = _desk(Script(event_pm=down), m)
    res = sim.run(d, m, START, 2)
    pd.testing.assert_frame_equal(res.ledger, sim.run(HOLD(), m, START, 2).ledger)
    assert d.fallbacks["event_pm:failed"] == 1
    assert [e["source"] for e in d.log if e["role"] == "event_pm"] == ["fallback"]


# ----------------------------------------------------------------------------- reflection

def _entry_then_cut(hold, cut_day=2, name=TICKERS[0]):
    """Enter on the hold's weights, then cut `name` at the morning of `cut_day`."""
    names = [t for t in TICKERS if hold[t] > 0]
    return dict(
        trader=lambda p: (_tl(adds=[(t, hold[t]) for t in names]) if not p["book"]["entered"]
                          else _tl(cuts=[name]) if p["clock"]["day"] == cut_day else _tl()),
        pm=lambda p: PMDecision(action="approve", trade_list=None, rationale="ok"))


def test_a_sale_is_settled_at_the_next_sessions_close_as_its_weight_times_the_move():
    """The reflection's facts are code's, so a sale before a rise reads as what it cost,
    in the example's terms: sold 8% before results, it rose 10.4%, cost 83 bp."""
    m = _mkt()
    hold = _hold_weights(m)
    b = Script(**_entry_then_cut(hold))
    d = _desk(b, m)
    res = sim.run(d, m, START, 4)
    trades = [s for s in d.journal.settlements if s["kind"] == "trade"]
    assert [s["name"] for s in trades] == [TICKERS[0]]   # the entry is the window's base, not graded
    s = trades[0]
    ex = calendar.rounds_for(D2)[0]["execution"]
    led = res.ledger.set_index("execution")
    sold = led.loc[calendar.rounds_for(START)[-1]["execution"], TICKERS[0]] - led.loc[ex, TICKERS[0]]
    px = m.exec_prices.loc[ex, TICKERS[0]]
    nav = res.periods[[p["execution"] for p in res.periods].index(ex)]["nav_before"]
    close = m.recent_closes(_at(D3, "16:00"), 1).iloc[-1][TICKERS[0]]
    assert s["weight"] == pytest.approx(-sold * px / nav, abs=1e-4)
    assert s["move"] == pytest.approx(close / px - 1, abs=1e-4)
    assert s["effect_bp"] == pytest.approx(-sold * px / nav * (close / px - 1) * 1e4, abs=0.1)
    reflect = _asked(b, "reflect")
    assert [p["clock"]["day"] for p in reflect] == [4]   # known after D3's close, read on day 4
    assert reflect[0]["settlements"][0]["id"] == s["id"] and "decisions" in reflect[0]


def test_a_name_held_through_its_results_is_settled_at_the_reaction():
    m = _mkt()
    events = pd.DataFrame({"ticker": [REPORTER], "accepted": [_at(D2, "16:30")]})
    d = _desk(Script(), m, events=events)
    sim.run(d, m, START, 4)
    held = [s for s in d.journal.settlements if s["kind"] == "held_through_results"]
    assert [s["name"] for s in held] == [REPORTER] and held[0]["day"] == 3
    closes = m.recent_closes(_at(D3, "16:00"), 10 ** 6)
    daily = closes.groupby(closes.index.date).tail(1)
    c = daily[REPORTER]
    assert held[0]["move"] == pytest.approx(c.iloc[-1] / c.iloc[-2] - 1, abs=1e-4)


def test_lessons_reach_the_desk_only_after_they_are_written_and_cite_real_settlements():
    m = _mkt()
    hold = _hold_weights(m)

    def reflect(p):
        return Reflection(summary="s", lessons=[Lesson(text="the cut cost us",
                                                       settlements=[p["settlements"][0]["id"]])])

    b = Script(reflect=reflect, **_entry_then_cut(hold))
    d = _desk(b, m)
    sim.run(d, m, START, 5)
    by_day = {p["clock"]["day"]: p["lessons"] for p in _asked(b, "trader")}
    assert by_day[3] == [] and by_day[4] == [{"day": 4, "text": "the cut cost us"}]
    assert d.journal.lessons[0]["from"] == [d.journal.settlements[0]["id"]]
    bad = Script(reflect=lambda p: Reflection(summary="s", lessons=[Lesson(text="x", settlements=["made-up"])]),
                 **_entry_then_cut(hold))
    d2 = _desk(bad, m)
    sim.run(d2, m, START, 5)
    assert d2.journal.lessons == [] and d2.fallbacks["reflect:failed"] == 1
    assert d2.journal.settlements == d.journal.settlements   # the facts stand without the lessons


# ----------------------------------------------------------------------------- point in time

def _busy():
    """A script that trades at triggers and mornings and writes lessons: every path on."""
    def event_pm(p):
        name = p["triggers"][0]["name"]
        held = next((x["weight_now"] for x in [p["triggers"][0]] if x.get("weight_now")), 0)
        return _trade(_tl(trims=[(name, "half")])) if held and held > 0.01 else \
            TriggerDecision(action="hold", trade_list=None, rationale="-")

    def reflect(p):
        return Reflection(summary="s", lessons=[Lesson(text=f"lesson on {p['settlements'][0]['name']}",
                                                       settlements=[p["settlements"][0]["id"]])])
    return event_pm, reflect


def _busy_script(hold):
    event_pm, reflect = _busy()
    return Script(event_pm=event_pm, reflect=reflect, **_entry_then_cut(hold))


def test_nothing_from_a_later_round_reaches_an_earlier_decision():
    """Every v2 input at once (morning chain, triggers, settlements, lessons): a payload
    that changed when only later prices changed read past its deadline."""
    bars = _shocked_bars()
    cut = calendar.rounds_for(DAYS[I0 + 3])[2]["deadline"]          # day 4, round 3
    late = bars.copy()
    late.loc[late["end"] > cut, ["open", "high", "low", "close"]] *= 1.7
    events = pd.DataFrame({"ticker": [REPORTER], "accepted": [_at(D2, "16:30")]})
    runs = []
    for b_ in (bars, late):
        m = _mkt(b_)
        rec = _busy_script(_hold_weights(m))
        sim.run(_desk(rec, m, events=events), m, START, 5)
        runs.append({(r, p["clock"]["day"], p["clock"]["round"]): p for r, _, p, _ in rec.seen
                     if (p["clock"]["day"], p["clock"]["round"]) <= (4, 3)})
    roles = {r for r, _, _ in runs[0]}
    assert {"v2_event", "v2_event_pm", "v2_reflect", "v2_pm"} <= roles
    assert runs[0] == runs[1]


class _Restarted:
    def __init__(self, brain, m, events):
        self.brain, self.m, self.events, self.saved, self.desk = brain, m, events, None, None

    def __call__(self, ctx):
        d = _desk(self.brain, self.m, events=self.events)
        d.restore(json.loads(self.saved) if self.saved else None, ctx.market.tickers)
        out = d(ctx)
        self.saved = json.dumps(d.state())
        self.desk = d
        return out


def test_a_desk_restarted_every_round_replays_triggers_and_reflection_identically():
    """Live, each round is its own process: a desk that forgot what fired today, which
    settlements it had made or the lessons it had written would wake twice, settle twice
    and lose its memory between rounds."""
    m = _mkt(_shocked_bars())
    events = pd.DataFrame({"ticker": [REPORTER], "accepted": [_at(D2, "16:30")]})
    hold = _hold_weights(m)
    whole = _desk(_busy_script(hold), m, events=events)
    again = _Restarted(_busy_script(hold), m, events)
    a, b = sim.run(whole, m, START, 6), sim.run(again, m, START, 6)
    pd.testing.assert_frame_equal(a.ledger, b.ledger)
    assert again.desk.journal.settlements == whole.journal.settlements
    assert again.desk.journal.lessons == whole.journal.lessons and whole.journal.lessons
    assert [(e["day"], e["round"], e["role"], e["source"]) for e in again.desk.chain] == \
        [(e["day"], e["round"], e["role"], e["source"]) for e in whole.chain]
    assert any(e["role"] == "event_pm" and e["traded"] for e in whole.log)


def test_lessons_never_cross_windows():
    """One window's memory: a desk for the next window starts with no lessons, so no
    window learns from another's outcome."""
    m = _mkt()
    d = _desk(_busy_script(_hold_weights(m)), m)
    sim.run(d, m, START, 5)
    assert d.journal.lessons
    fresh = _desk(Script(), m)
    sim.run(fresh, m, DAYS[I0 + 5], 1)
    assert fresh.journal.lessons == [] and fresh.journal.settlements == []
