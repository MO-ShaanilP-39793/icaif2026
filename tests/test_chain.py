"""Desk v3 stage 2, variation B: code wakes the PM, which may only reduce (icaif/agents/chain.py)."""

import json

import numpy as np
import pandas as pd
import pytest

from icaif import baselines, calendar, sim
from icaif.agents import chain as CH
from icaif.agents.schemas import ChainDecision, PMCheck, PMEntry, EntryWeight, ReduceLine
from icaif.agents.v3 import V3Config, V3Desk
from tests.test_agents import DAYS, START, _mkt
from tests.test_quant import _bars
from tests.test_sim import TICKERS

N = 4
A, B_, C_ = TICKERS[0], TICKERS[1], TICKERS[2]
STATED = {A: 0.3, B_: 0.3, C_: 0.2}


class Script(CH.ChainHoldBrain):
    """The chain's hold brain, with roles answered by `answers[key](payload)` instead."""

    def __init__(self, name="script", **answers):
        self.name, self.answers, self.seen = name, answers, []

    def decide(self, role, system, payload, schema, timeout):
        self.seen.append((role, system, payload))
        key = role.removeprefix("v3_")
        if key in self.answers:
            return self.answers[key](payload)
        return super().decide(role, system, payload, schema, timeout)


def _entry(_=None):
    return PMEntry(weights=[EntryWeight(name=n, weight=w) for n, w in STATED.items()],
                   thesis="t", conditions=[])


def _confirm(_):
    return PMCheck(action="confirm", revised=None, rationale="ok")


def _hold(_=None):
    return ChainDecision(action="hold", lines=[], gross=None, drop_conditions=[], add_conditions=[],
                         cites=[], rationale="hold")


def _reduce(lines=(), gross=None, cites=()):
    return ChainDecision(action="reduce", lines=[ReduceLine(name=n, to=t) for n, t in lines], gross=gross,
                         drop_conditions=[], add_conditions=[], cites=list(cites), rationale="r")


def _desk(brain, **cfg):
    return CH.ChainDesk({"quick": brain, "deep": brain}, CH.ChainConfig(**cfg))


def _asked(brain):
    return [p for r, _, p in brain.seen if r == "v3_chain_pm"]


# ----------------------------------------------------------------------------- plumbing

@pytest.mark.parametrize("overlay", [None, 0.10])
def test_a_chain_desk_answered_by_code_trades_exactly_as_the_hold(overlay):
    """Every PM-vs-hold comparison in stage 2 measures the PM only if a chain that holds
    at every wake is, trade for trade, the hold it is compared with."""
    m = _mkt()
    d = _desk(CH.ChainHoldBrain(), overlay=overlay)
    hold = baselines.scaled(baselines.InverseVolHold, 0.10 if overlay else 0.75)
    got, want = sim.run(d, m, START, N), sim.run(hold(), m, START, N)
    pd.testing.assert_frame_equal(got.ledger, want.ledger)
    assert [w["day"] for w in d.wakes if w["morning"]] == [2, 3, 4], "each morning after entry"


def test_a_chain_pm_that_always_holds_repeats_the_stage1_desk_on_the_same_entry():
    """Stage 2 is judged against stage 1 on the same cached entry: the chain must not move
    the entry, or its "value" would partly be a different book."""
    m = _mkt()
    mk = lambda: Script(pm=_entry, pm_check=_confirm, chain_pm=_hold)  # noqa: E731
    one = V3Desk({"quick": mk(), "deep": mk()}, V3Config(overlay=0.10))
    two = _desk(mk(), overlay=0.10)
    pd.testing.assert_frame_equal(sim.run(one, m, START, N).ledger, sim.run(two, m, START, N).ledger)
    assert two.entry == one.entry


# ----------------------------------------------------------------------------- the PM's scale

def test_under_the_overlay_the_pm_reads_its_own_book_and_its_exit_reaches_the_real_one():
    """Shown the real book, the PM would see a 8% gross it never chose; its decision,
    taken at its scale, must land on the real book scaled back down."""
    b = Script(pm=_entry, pm_check=_confirm,
               chain_pm=lambda p: _reduce([(A, "exit")]) if p["clock"]["day"] == 2 else _hold())
    d = _desk(b, overlay=0.10)
    res = sim.run(d, _mkt(), START, N)
    first = _asked(b)[0]
    # The same entry bought at the PM's own gross: what its book really holds that morning.
    ref = Script(pm=_entry, pm_check=_confirm, chain_pm=_hold)
    sim.run(_desk(ref), _mkt(), START, 2)
    own = _asked(ref)[0]
    assert first["book"]["gross"] == pytest.approx(own["book"]["gross"], abs=2e-4)
    assert first["book"]["gross"] != pytest.approx(0.8, abs=5e-3), "the prices moved; the test must see it"
    for x, y in zip(first["book"]["names"], own["book"]["names"]):
        assert x["weight"] == pytest.approx(y["weight"], abs=2e-4)
    assert "0.1" not in json.dumps(first["budget"]) and "overlay" not in json.dumps(first).lower()
    led = res.ledger
    assert led.iloc[-1][A] == pytest.approx(0.0, abs=1e-6), "exited"
    assert led.iloc[-1][B_] > 0 and led.iloc[-1][C_] > 0
    assert d.interventions[0]["sold_pm"][A] == pytest.approx(
        {r["name"]: r["weight"] for r in first["book"]["names"]}[A], abs=2e-4)
    assert d.interventions[0]["counts"]


# ----------------------------------------------------------------------------- what code refuses

@pytest.mark.parametrize("answer, why", [
    (lambda p: _reduce(gross=0.95), "not below"),                                    # adds
    (lambda p: _reduce([(TICKERS[9], "exit")]), "not in the book"),
    (lambda p: _reduce([(A, "exit"), (A, "trim_half")]), "twice"),
    (lambda p: _reduce([(A, "exit")], cites=["move:ZZZ"]), "did not happen"),
    (lambda p: ChainDecision(action="hold", lines=[ReduceLine(name=A, to="exit")], gross=None,
                             drop_conditions=[], add_conditions=[], cites=[], rationale="x"), "hold carries"),
])
def test_an_answer_that_could_add_or_names_what_is_not_there_is_refused_whole_and_holds(answer, why):
    b = Script(pm=_entry, pm_check=_confirm, chain_pm=answer)
    d = _desk(b, overlay=0.10)
    res = sim.run(d, _mkt(), START, N)
    hold = Script(pm=_entry, pm_check=_confirm, chain_pm=_hold)
    want = sim.run(_desk(hold, overlay=0.10), _mkt(), START, N)
    pd.testing.assert_frame_equal(res.ledger, want.ledger)
    assert d.fallbacks["chain:hold"] == 3 and why in d.chain[-1]["reason"]


def test_outside_the_morning_the_pm_may_reduce_only_the_names_an_event_named():
    """An intraday wake about one name is not a licence to rework the book: that is how
    v1's events turned into rewrites."""
    bars = _bars(DAYS, 8)
    day2 = DAYS[DAYS.index(START) + 1]
    crash = calendar.rounds_for(day2)[1]["execution"]       # a bar ending before round 3
    hit = (bars["ticker"] == A) & (bars["end"] > crash)
    bars.loc[hit, ["open", "high", "low", "close"]] *= 0.80
    asked = []

    def pm(p):
        asked.append(p)
        if "morning" in p["wake"]:
            return _hold()
        return _reduce([(B_, "exit")], cites=[f"move:{A}"])

    b = Script(pm=_entry, pm_check=_confirm, chain_pm=pm)
    d = _desk(b, overlay=0.10)
    sim.run(d, _mkt(bars), START, N)
    intraday = [p for p in asked if "morning" not in p["wake"]]
    assert intraday and intraday[0]["events"][0]["kind"] == "move" and intraday[0]["events"][0]["name"] == A
    assert "only the event names" in [e for e in d.chain if e["role"] == "chain_pm" and e["source"] == "failed"][0]["reason"]
    assert d.interventions == []


def test_the_budget_refuses_a_third_intervention_but_not_all_cash_at_the_end():
    """Every trade after entry costs turnover rank; the first one costs the most. Two is
    the cap, and booking the window to cash at its end is the one move it never blocks."""
    def pm(p):
        day = p["clock"]["day"]
        if day == 2:
            return _reduce([(A, "trim_quarter")])
        if day == 3:
            return _reduce([(B_, "trim_quarter")])
        if day == 4 and p["clock"]["round"] == 1:
            return _reduce([(C_, "trim_quarter")])
        return _hold()

    b = Script(pm=_entry, pm_check=_confirm, chain_pm=pm)
    d = _desk(b, overlay=0.10, window_days=6)
    sim.run(d, _mkt(), START, 6)
    assert [iv["day"] for iv in d.interventions] == [2, 3] and d.fallbacks["chain:over_budget"] == 1

    def late(p):
        return _reduce(gross=0.0) if p["clock"]["day"] == 5 else pm(p)

    b = Script(pm=_entry, pm_check=_confirm, chain_pm=late)
    d = _desk(b, overlay=0.10, window_days=6)
    res = sim.run(d, _mkt(), START, 6)
    assert [iv["day"] for iv in d.interventions] == [2, 3, 5] and not d.interventions[-1]["counts"]
    assert res.ledger.iloc[-1][TICKERS].abs().sum() == pytest.approx(0.0, abs=1e-6)


def test_a_reduction_under_the_minimum_is_a_hold_and_spends_nothing():
    b = Script(pm=_entry, pm_check=_confirm, chain_pm=lambda p: _reduce(gross=p["book"]["gross"] - 0.001))
    d = _desk(b, overlay=0.10)
    sim.run(d, _mkt(), START, N)
    assert d.interventions == [] and d.fallbacks["chain:below_min_trade"] == 3


# ----------------------------------------------------------------------------- point in time

def test_no_wake_or_payload_changes_when_every_later_bar_is_rewritten():
    """A payload that read past its deadline would show the PM the move it is deciding on."""
    bars = _bars(DAYS, 8)
    day3 = DAYS[DAYS.index(START) + 2]
    cut = calendar.rounds_for(day3)[0]["deadline"]
    late = bars.copy()
    late.loc[late["end"] > cut, ["open", "high", "low", "close"]] *= 3.0
    seen = []
    for b_ in (bars, late):
        b = Script(pm=_entry, pm_check=_confirm, chain_pm=_hold)
        d = _desk(b, overlay=0.10)
        sim.run(d, _mkt(b_), START, 3)
        upto = [p for p in _asked(b) if (p["clock"]["day"], p["clock"]["round"]) <= (3, 1)]
        seen.append(json.dumps(upto, sort_keys=True, default=str))
    assert seen[0] == seen[1] and len(json.loads(seen[0])) == 2


# ----------------------------------------------------------------------------- restarts

class _Restarted:
    def __init__(self, brain, **cfg):
        self.brain, self.cfg, self.saved, self.desk = brain, cfg, None, None

    def __call__(self, ctx):
        d = _desk(self.brain, **self.cfg)
        d.restore(json.loads(self.saved) if self.saved else None, ctx.market.tickers)
        out = d(ctx)
        self.saved = json.dumps(d.state())
        self.desk = d
        return out


def test_a_chain_desk_restored_before_every_round_trades_as_one_that_never_stopped():
    """Live, each round is its own process: a desk that forgot its spent budget or its
    wakes would intervene a third time, or answer the same results twice."""
    def pm(p):
        return _reduce([(A, "trim_half")]) if p["clock"]["day"] in (2, 3) else _hold()

    m = _mkt()
    mk = lambda: Script(pm=_entry, pm_check=_confirm, chain_pm=pm)  # noqa: E731
    whole, again = _desk(mk(), overlay=0.10), _Restarted(mk(), overlay=0.10)
    a, b = sim.run(whole, m, START, N), sim.run(again, m, START, N)
    pd.testing.assert_frame_equal(a.ledger, b.ledger)
    assert again.desk.interventions == whole.interventions and again.desk.wakes == whole.wakes


def test_the_chain_pm_reaches_the_cache_under_its_own_name_and_prompt():
    b = Script(pm=_entry, pm_check=_confirm, chain_pm=_hold)
    sim.run(_desk(b, overlay=0.10), _mkt(), START, 2)
    roles = {r for r, _, _ in b.seen}
    assert "v3_chain_pm" in roles
    system = [s for r, s, _ in b.seen if r == "v3_chain_pm"][0]
    assert "You cannot buy" in system and "overlay" not in system.lower()
    assert np.isfinite(len(system))


# ----------------------------------------------------------------------------- what wakes the PM

def test_results_and_a_serious_filing_wake_the_pm_at_the_rounds_that_can_still_act_on_them():
    """Before results, only the day's last fill comes before the reaction; after them, the
    PM must see the session trade first; a restatement must not wait for the morning."""
    from icaif.agents.desk import EarningsCalendar
    from icaif.agents.triggers import EarningsHistory

    m = _mkt()
    i = DAYS.index(START)
    release = calendar.at(DAYS[i + 1], pd.Timestamp("16:30").time())
    events = pd.DataFrame({"ticker": [A], "accepted": [release]})
    filings = pd.DataFrame({"ticker": [A, B_, C_],
                            "accepted": [release, calendar.at(DAYS[i + 1], pd.Timestamp("12:00").time()),
                                         calendar.at(DAYS[i + 1], pd.Timestamp("12:00").time())],
                            "items": ["2.02,9.01", "4.02", "8.01"], "amended": [False] * 3,
                            "text": ["results text", "restatement text", "routine"]})
    asked = []
    b = Script(pm=_entry, pm_check=_confirm, chain_pm=lambda p: asked.append(p) or _hold())
    d = CH.ChainDesk({"quick": b, "deep": b}, CH.ChainConfig(overlay=0.10),
                     earnings=EarningsCalendar(events, m.days),
                     earnings_history=EarningsHistory.from_market(events, m), filings=filings)
    sim.run(d, m, START, N)
    when = {e["id"]: (p["clock"]["day"], p["clock"]["round"]) for p in asked for e in p["events"]}
    last = len(calendar.rounds_for(DAYS[i + 1]))
    assert when[f"before_results:{A}"] == (2, last)
    assert when[f"after_results:{A}"] == (3, 3)          # no bar of the session ends by round 2
    assert when[f"filing:{B_}"] == (2, 4)                # the first round after it was accepted
    assert not any(k.startswith(f"filing:{C_}") for k in when), "a routine 8-K wakes no one"
    after = [e for p in asked for e in p["events"] if e["id"] == f"after_results:{A}"][0]
    assert after["tag"]["status"] == "already_reacted" and after["release"]
    assert sum(1 for p in asked for e in p["events"] if e["kind"] == "after_results") == 1, "once"
