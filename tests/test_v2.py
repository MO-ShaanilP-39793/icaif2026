"""The v2 desk's morning chain (icaif/agents/v2.py, prompts_v2.py)."""

import json
import time

import pandas as pd
import pytest

from icaif import baselines, calendar, sim
from icaif.agents import prompts_v2
from icaif.agents.brains import BrainError, CachedBrain
from icaif.agents.desk import EarningsCalendar
from icaif.agents.schemas import AddLine, AnalystReport, CutLine, PMDecision, RiskReview, TradeList
from icaif.agents.triggers import EarningsHistory
from icaif.agents.v2 import HOLD_GROSS, HoldBrain, V2Config, V2Desk
from tests.test_agents import DAYS, START, _mkt
from tests.test_quant import _bars
from tests.test_sim import TICKERS

HOLD = baselines.scaled(baselines.InverseVolHold, HOLD_GROSS)
N = 5


class Script(HoldBrain):
    """The hold brain, with some roles answered by `answers[key](payload)` instead (a
    callable may raise or sleep); records every role, payload and prompt it was asked."""

    def __init__(self, name="script", **answers):
        self.name, self.answers, self.seen = name, answers, []

    def decide(self, role, system, payload, schema, timeout):
        self.seen.append((role, system, payload, timeout))
        key = role.removeprefix("v2_")
        if key in self.answers:
            return self.answers[key](payload)
        return super().decide(role, system, payload, schema, timeout)


def _desk(quick, deep=None, cfg=None, **kw):
    return V2Desk({"quick": quick, "deep": deep or quick}, cfg or V2Config(), **kw)


def _ctx(m, day, rnd=1):
    r = calendar.rounds_for(day)[rnd - 1]
    return sim.RoundContext(day, rnd, r["deadline"], r["execution"], {}, sim.INITIAL_NAV, m)


def _hold_weights(m) -> dict:
    return HOLD()(_ctx(m, START))


def _fail(_):
    raise BrainError("down")


def _tl(adds=(), cuts=(), exposure=None):
    return TradeList(adds=[AddLine(name=n, weight=w, why="x") for n, w in adds],
                     cuts=[CutLine(name=n, why="x") for n in cuts], trims=[],
                     target_exposure=exposure, rationale="r")


def _approve(_):
    return PMDecision(action="approve", trade_list=None, rationale="ok")


# ----------------------------------------------------------------------------- plumbing

def test_a_v2_desk_answered_by_code_trades_exactly_as_the_hold():
    """Every role answered in code (the hold brain) must trade as `inv_vol_hold_75`, or
    every LLM-vs-hold comparison measures the desk's plumbing instead of the LLM."""
    m = _mkt()
    d = _desk(HoldBrain())
    got, want = sim.run(d, m, START, N), sim.run(HOLD(), m, START, N)
    pd.testing.assert_frame_equal(got.ledger, want.ledger)
    assert got.metrics() == want.metrics()
    assert d.fallbacks == {"pm:failed": 1, "entry:hold_book": 1}
    assert {e["role"] for e in d.chain} == {"market", "quant", "bull_1", "bear_1", "bull_2",
                                            "bear_2", "trader", "risk", "pm"}


def test_an_approved_entry_list_trades_exactly_the_weights_it_states():
    """The compiled path, not the fallback: a stated weight that went through
    `weights.safe` a second time could be submitted a grid step lower than stated."""
    m = _mkt()
    hold = _hold_weights(m)
    assert min(w for w in hold.values() if w > 0) >= 0.005
    entry = lambda p: (_tl(adds=[(t, w) for t, w in hold.items() if w > 0])  # noqa: E731
                       if not p["book"]["entered"] else _tl())
    b = Script(trader=entry, pm=lambda p: _approve(p) if not p["book"]["entered"]
               else PMDecision(action="hold", trade_list=None, rationale="hold"))
    d = _desk(b)
    got, want = sim.run(d, m, START, N), sim.run(HOLD(), m, START, N)
    pd.testing.assert_frame_equal(got.ledger, want.ledger)
    assert d.fallbacks == {} and d.log[0]["source"] == "brain" and d.log[0]["traded"]


# ----------------------------------------------------------------------------- fallbacks

def test_a_late_or_failing_role_is_skipped_counted_and_the_desk_goes_on_without_it():
    """A slow role must cost its own answer, not the morning: the PM still decides, told
    which report is missing and why."""
    slots = {k: v * 0.01 for k, v in V2Config().slots.items()}           # 1.8 s for analysts
    b = Script(quant=_fail, market=lambda p: time.sleep(2.4) or AnalystReport(summary="late", names=[]))
    d = _desk(b, cfg=V2Config(slots=slots, grace_s=0.1, min_call_s=0.0))
    sim.run(d, _mkt(), START, 1)
    src = {e["role"]: e["source"] for e in d.chain}
    assert src["quant"] == "failed" and src["market"] == "late" and src["pm"] == "failed"
    assert d.fallbacks["quant:failed"] == 1 and d.fallbacks["market:late"] == 1
    trader = next(p for r, _, p, _ in b.seen if r == "v2_trader")
    assert "unavailable" in trader["reports"]["quant"] and "unavailable" in trader["reports"]["market"]


def test_a_role_whose_slot_is_spent_is_not_asked_and_each_call_gets_its_slots_time():
    slots = dict(V2Config().slots, bull_2=0.0, bear_2=0.0)
    b = Script()
    d = _desk(b, cfg=V2Config(slots=slots))
    sim.run(d, _mkt(), START, 1)
    asked = [r for r, _, _, _ in b.seen]
    assert "v2_bull_2" not in asked and "v2_bear_2" not in asked
    assert d.fallbacks["bull:skipped"] == 1 and d.fallbacks["bear:skipped"] == 1
    timeouts = {r: t for r, _, _, t in b.seen}
    assert 175 < timeouts["v2_market"] <= 180 and 1010 < timeouts["v2_pm"] <= 1020


def test_a_failed_pm_holds_after_entry_and_buys_the_hold_book_only_before_it():
    m = _mkt()
    d = _desk(Script(pm=_fail))
    got = sim.run(d, m, START, N)
    pd.testing.assert_frame_equal(got.ledger, sim.run(HOLD(), m, START, N).ledger)
    assert d.fallbacks["entry:hold_book"] == 1 and d.fallbacks["pm:hold"] == N - 1
    assert all(e["source"] == "fallback" for e in d.log)


def test_a_pm_list_code_refuses_is_no_trade_never_a_repaired_one():
    """Clipped to the cap or stripped of its bad line, the list would trade a book no
    role proposed, logged beside a rationale for another."""
    m = _mkt()
    hold = _hold_weights(m)
    big = [t for t in TICKERS if hold[t] > 0][:2]
    amend = lambda p: (PMDecision(action="hold", trade_list=None, rationale="-")  # noqa: E731
                       if not p["book"]["entered"] else
                       PMDecision(action="amend", trade_list=_tl(adds=[(big[0], 0.31 - 0.01), (big[1], 0.29)]),
                                  rationale="concentrate"))
    d = _desk(Script(pm=lambda p: _fail(p) if not p["book"]["entered"] else amend(p)))
    got = sim.run(d, m, START, 3)
    pd.testing.assert_frame_equal(got.ledger, sim.run(HOLD(), m, START, 3).ledger)
    assert d.fallbacks["pm:refused"] == 2
    assert "over the 0.75" in d.log[1]["reason"] or "allowed" in d.log[1]["reason"]


@pytest.mark.parametrize("signoff,traded_gross", [(None, None), (0.9, 0.9)])
def test_gross_above_75_trades_only_within_the_risk_managers_signoff(signoff, traded_gross):
    """The v1 desks entered at 91-94% every time; above 0.75 now takes a reasoned
    sign-off, and without one the list is refused (and the hold book bought)."""
    m = _mkt()
    hold = _hold_weights(m)
    up = {t: round(w * 1.2, 6) for t, w in hold.items() if w > 0}
    b = Script(trader=lambda p: _tl(adds=list(up.items())) if not p["book"]["entered"] else _tl(),
               risk=lambda p: RiskReview(verdict="approve", objections=[], exposure_signoff=signoff,
                                         rationale="calm enough"),
               pm=lambda p: _approve(p) if not p["book"]["entered"]
               else PMDecision(action="hold", trade_list=None, rationale="-"))
    d = _desk(b)
    got = sim.run(d, m, START, 1)
    first = got.ledger.iloc[0]
    px = m.exec_prices.iloc[m.exec_prices.index.get_loc(_ctx(m, START).execution)]
    gross = float((first[TICKERS] * px[TICKERS]).sum()) / sim.INITIAL_NAV
    if traded_gross is None:
        assert d.fallbacks["pm:refused"] == 1 and gross == pytest.approx(0.75, abs=1e-3)
        risk_view = next(p for r, _, p, _ in b.seen if r == "v2_risk")
        assert "needs_exposure_signoff" in risk_view["compiled"]
    else:
        assert d.fallbacks == {} and gross == pytest.approx(traded_gross, abs=1e-3)


def test_turnover_is_shown_and_capped_only_against_a_runaway():
    m = _mkt()
    hold = _hold_weights(m)
    names = [t for t in TICKERS if hold[t] > 0]
    flip = lambda p: (_tl(adds=[(t, hold[t]) for t in names]) if not p["book"]["entered"]  # noqa: E731
                      else _tl(cuts=names[:10], exposure=None))
    b = Script(trader=flip, pm=_approve)
    cut = sum(hold[t] for t in names[:10])
    d = _desk(b, cfg=V2Config(turnover_cap=0.75 + cut / 2))
    sim.run(d, m, START, 3)
    shown = [p["turnover"] for r, _, p, _ in b.seen if r == "v2_pm"]
    assert shown[0]["spent"] == 0 and shown[1]["spent"] == pytest.approx(0.75, abs=1e-3)
    assert shown[1]["window_budget"] == round(0.75 + cut / 2, 4)
    assert d.fallbacks["pm:refused"] == 2 and "budget" in d.log[1]["reason"]
    loose = _desk(Script(trader=flip, pm=_approve))                  # the default cap: 5 books
    sim.run(loose, m, START, 2)
    assert loose.fallbacks == {} and loose.log[1]["traded"]


# ----------------------------------------------------------------------------- what each role sees

def test_no_role_sees_a_regime_label_or_the_rules_answer_and_only_risk_hears_the_evidence():
    """Told that holding wins, v1 held through 86 questions; told the regime was calm,
    the free desk stayed 93% invested through a 3% fall."""
    b = Script()
    sim.run(_desk(b), _mkt(), START, 2)
    for role, system, payload, _ in b.seen:
        text = json.dumps(payload)
        for banned in ("p_turbulent_next_session", "regime_persistence_days", "rule_proposal"):
            assert banned not in text, (role, banned)
        assert ("Buying once and holding wins" in system) == (role == "v2_risk")
        assert system == prompts_v2.SYSTEM[role.removeprefix("v2_").split("_")[0]]


def test_each_role_runs_on_its_tier_and_reaches_the_cache_under_its_own_key(tmp_path):
    """A cache keyed without the role, or the round of the debate, would hand the bear the
    bull's answer whenever their payloads matched."""
    quick, deep = Script("quick"), Script("deep")
    sim.run(_desk(quick, deep), _mkt(), START, 1)
    assert {r for r, *_ in deep.seen} == {"v2_risk", "v2_pm"}
    assert {r for r, *_ in quick.seen} == {"v2_market", "v2_quant", "v2_bull_1", "v2_bear_1",
                                           "v2_bull_2", "v2_bear_2", "v2_trader"}
    cached = CachedBrain(HoldBrain(), tmp_path)
    d = _desk(cached)
    sim.run(d, _mkt(), START, 1)
    assert len(list(tmp_path.glob("*.json"))) == len([e for e in d.chain if e["role"] != "pm"])


def test_the_earnings_analyst_is_asked_only_when_a_name_reports_and_reads_the_tags():
    m = _mkt()
    i = DAYS.index(START)
    events = pd.DataFrame({"ticker": [TICKERS[3]], "accepted": [calendar.at(DAYS[i + 1], pd.Timestamp("16:30").time())]})
    b = Script()
    d = _desk(b, earnings=EarningsCalendar(events, m.days),
              earnings_history=EarningsHistory.from_market(events, m))
    sim.run(d, m, START, 4)
    asked = [(p["clock"]["day"], p) for r, _, p, _ in b.seen if r == "v2_earnings"]
    assert [day for day, _ in asked] == [1, 2, 3]          # 2 sessions ahead, 1, then reacted
    tags = [p["reporting"][0]["tag"] for _, p in asked]
    assert [t["status"] for t in tags] == ["upcoming", "upcoming", "already_reacted"]
    assert tags[2]["minutes_traded"] == 0
    assert "news" not in {e["role"] for e in d.chain}        # no headlines, no filings


# ----------------------------------------------------------------------------- point in time

def test_no_morning_payload_changes_when_every_later_bar_is_rewritten():
    """The v2 roles read the same observation doors as v1, plus the trigger tags; one that
    read past its deadline would show a morning the day it is deciding about."""
    bars = _bars(DAYS, 8)
    i = DAYS.index(START)
    cut = calendar.rounds_for(DAYS[i + 2])[0]["deadline"]
    late = bars.copy()
    late.loc[late["end"] > cut, ["open", "high", "low", "close"]] *= 3.0
    events = pd.DataFrame({"ticker": [TICKERS[3], TICKERS[5]],
                           "accepted": [calendar.at(DAYS[i + 1], pd.Timestamp("16:30").time()),
                                        cut + pd.Timedelta(minutes=1)]})
    runs = []
    for b_ in (bars, late):
        m = _mkt(b_)
        ev = events if b_ is late else events.iloc[:1]
        rec = Script()
        sim.run(_desk(rec, earnings=EarningsCalendar(events.iloc[:1], m.days),
                      earnings_history=EarningsHistory.from_market(ev, m)), m, START, 3)
        # Keyed, not listed: the analysts ask in parallel, so the order they reach the
        # brain is the threads', not the desk's.
        runs.append({(r, p["clock"]["day"]): p for r, _, p, _ in rec.seen})
    assert len(runs[0]) == 3 * 9 + 3 and runs[0] == runs[1]      # earnings asked each morning


class _Restarted:
    """A v2 desk rebuilt from its JSON state before every round, as live rounds run."""

    def __init__(self, brain):
        self.brain, self.saved, self.desk = brain, None, None

    def __call__(self, ctx):
        d = _desk(self.brain)
        d.restore(json.loads(self.saved) if self.saved else None, ctx.market.tickers)
        out = d(ctx)
        self.saved = json.dumps(d.state())
        self.desk = d
        return out


def test_a_v2_desk_restored_before_every_round_trades_as_one_that_never_stopped():
    m = _mkt()
    hold = _hold_weights(m)
    names = [t for t in TICKERS if hold[t] > 0]

    def script():
        return Script(trader=lambda p: (_tl(adds=[(t, hold[t]) for t in names]) if not p["book"]["entered"]
                                        else _tl(cuts=names[:1]) if p["clock"]["day"] == 3 else _tl()),
                      pm=_approve)

    whole, again = _desk(script()), _Restarted(script())
    a, b = sim.run(whole, m, START, N), sim.run(again, m, START, N)
    pd.testing.assert_frame_equal(a.ledger, b.ledger)
    assert again.desk.fallbacks == whole.fallbacks
    assert [e["role"] for e in again.desk.chain] == [e["role"] for e in whole.chain]
