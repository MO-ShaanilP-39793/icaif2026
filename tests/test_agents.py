import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from icaif import calendar, quant_strategies as qs, sim
from icaif import weights as W
from icaif.agents import brains
from icaif.agents.brains import BrainError, CachedBrain, ClaudeBrain, RuleBrain
from icaif.agents.desk import Desk, DeskConfig
from icaif.agents.schemas import EntryDecision, EventDecision, Exclusion, NameCall, ReviewDecision
from tests.test_quant import _bars, _days
from tests.test_sim import TICKERS, _market

DAYS = _days(70)
START = DAYS[62]


def _mkt(bars=None, seed=8):
    return _market(DAYS, info_bars=_bars(DAYS, seed) if bars is None else bars)


class Scripted:
    """Answers each role from a script; records every payload it was shown."""

    name = "scripted"

    def __init__(self, **answers):
        self.answers, self.seen = answers, []

    def decide(self, role, system, payload, schema, timeout):
        self.seen.append((role, payload))
        a = self.answers.get(role)
        if a is None:
            return schema.model_validate(payload["rule_proposal"])
        return a(payload) if callable(a) else a


class Failing:
    name = "failing"

    def decide(self, *a, **k):
        raise BrainError("the model is down")


def _gross_after_trade(res, k):
    """Gross weight the k-th trade bought, at its own fill (every fill is 100 here)."""
    i = [j for j, p in enumerate(res.periods) if p["traded_notional"] > 0][k]
    row = res.ledger.iloc[i]
    return float(row[TICKERS].sum()) * 100 / res.periods[i]["nav_before"]


def _run(brain, cfg=None, market=None, n=5):
    d = Desk(brain, cfg)
    return d, sim.run(d, market or _mkt(), START, n)


def test_a_rule_desk_trades_exactly_as_the_backtested_quant_candidate():
    """If the desk's plumbing (shape, exposure, cap, grid) drifted from the candidate the
    rule stands for, every LLM-vs-rule comparison would measure the drift."""
    m = _mkt()
    ref = sim.run(qs.CANDIDATES["q_riskparity_entry_regime"](), m, START, 5)
    _, got = _run(RuleBrain(), market=m)
    pd.testing.assert_frame_equal(got.ledger, ref.ledger)
    assert got.metrics() == ref.metrics()


def test_a_failing_brain_falls_back_to_the_rule_every_time_and_never_misses_a_round():
    m = _mkt()
    _, want = _run(RuleBrain(), market=m)
    desk, got = _run(Failing(), market=m)
    pd.testing.assert_frame_equal(got.ledger, want.ledger)
    assert desk.log and all(e["source"] == "fallback" for e in desk.log)
    assert got.invalid_rounds == []


def test_an_avoid_code_the_agent_invented_is_rejected_not_guessed():
    """Mapped by a guess, an unknown code would drop a name the agent never named."""
    bad = EntryDecision(shape="risk_parity", views="none", exposure=0.6,
                        avoid=[Exclusion(name="S99", signal="other", why="x")], rationale="x")
    desk, _ = _run(Scripted(entry=bad), DeskConfig(anonymize=True))
    assert desk.log[0]["role"] == "entry" and desk.log[0]["source"] == "fallback"
    assert "S99" in desk.log[0]["reason"]


def test_an_avoided_name_is_left_out_and_the_book_still_lands_on_its_exposure():
    pick = lambda p: EntryDecision(  # noqa: E731
        shape="inverse_vol", views="none", exposure=0.6, rationale="x",
        avoid=[Exclusion(name=p["names"][0]["name"], signal="earnings", why="reports tomorrow")])
    desk, res = _run(Scripted(entry=pick))
    first = res.ledger.iloc[0]
    left_out = desk.log[0]["decision"]["avoid"][0]["name"]
    assert first[left_out] == 0
    assert _gross_after_trade(res, 0) == pytest.approx(0.6, abs=30 * W.GRID)


def test_replayed_observations_carry_no_real_ticker_and_no_date():
    """Shown real names and dates from its training years, the model can recall what
    happened next, and the replay would score memory as judgement."""
    b = Scripted()
    _run(b, DeskConfig(anonymize=True))
    text = json.dumps([p for _, p in b.seen])
    assert not [t for t in TICKERS if f'"{t}"' in text]
    assert "2026-" not in text
    codes = {row["name"] for row in b.seen[0][1]["names"]}
    assert codes == {f"S{i:02d}" for i in range(1, 31)}


def test_the_entry_observation_is_unchanged_when_every_later_bar_is_rewritten():
    seen = []
    for shock in (None, START):
        b = Scripted()
        _run(b, market=_market(DAYS, info_bars=_bars(DAYS, 8, shock_from=shock)), n=1)
        seen.append(json.dumps(b.seen[0][1], sort_keys=True))
    assert seen[0] == seen[1]


def test_a_review_change_smaller_than_the_band_is_a_hold_not_a_trade():
    """A 1-point exposure trim pays the fee and a turnover rank for nothing."""
    nudge = lambda p: ReviewDecision(action="set_exposure", reason=None,  # noqa: E731
                                     exposure=p["book"]["gross"] - 0.01, exit=[], rationale="x")
    _, res = _run(Scripted(review=nudge))
    assert sum(p["traded_notional"] > 0 for p in res.periods) == 1


def test_a_review_cut_rescales_the_book_and_set_exposure_without_a_number_falls_back():
    cut = ReviewDecision(action="set_exposure", exposure=0.30, reason=None, exit=[], rationale="storm")
    desk, res = _run(Scripted(review=cut), n=2)
    assert _gross_after_trade(res, 1) == pytest.approx(0.30, abs=30 * W.GRID)
    broken = ReviewDecision(action="set_exposure", exposure=None, reason=None, exit=[], rationale="x")
    desk, _ = _run(Scripted(review=broken), n=2)
    assert [e["source"] for e in desk.log if e["role"] == "review"] == ["fallback"]


def test_a_sharp_move_wakes_the_analyst_for_that_name_and_an_exit_sells_only_it():
    bars = _bars(DAYS, 8)
    day = DAYS[63]
    victim = TICKERS[4]
    hit = (bars["ticker"] == victim) & (bars["start"] >= calendar.at(day, calendar.ROUNDS[2][1]))
    bars.loc[hit, ["open", "high", "low", "close"]] *= 0.8
    exit_all = lambda p: EventDecision(calls=[  # noqa: E731
        NameCall(name=t["name"], action="exit", reason="gap") for t in p["triggers"]])
    b = Scripted(event=exit_all)
    desk, res = _run(b, market=_mkt(bars), n=3)
    events = [p for r, p in b.seen if r == "event"]
    assert [t["name"] for t in events[0]["triggers"]] == [victim]
    assert desk.book.weights[victim] == 0
    others = desk.book.weights.drop(victim)
    assert (others > 0).all()


def test_a_held_name_with_no_price_makes_the_desk_hold_rather_than_value_the_book_without_it():
    """pandas' sum skips NaN, so the book was valued as if the unpriced name were worth
    nothing: every weight overstated, and an exit target built from them could carry the
    NaN into weights.safe."""
    m = _mkt()
    d = Desk(Scripted())
    sim.run(d, m, START, 1)
    held = {t: 100.0 for t in TICKERS}
    ctx = sim.RoundContext(DAYS[63], 1, calendar.at(DAYS[63], calendar.ROUNDS[1][0]),
                           calendar.at(DAYS[63], calendar.ROUNDS[1][1]), held, 1000.0, m)
    last = ctx.recent_closes(1).index[-1]
    m._close_panel.loc[last, TICKERS[0]] = np.nan
    value, nav = d._value(ctx, TICKERS)
    assert np.isnan(nav)
    unheld = dict(held, **{TICKERS[0]: 0.0})
    ctx.shares = unheld
    _, nav = d._value(ctx, TICKERS)
    assert np.isfinite(nav)  # a name not held needs no price


class _Restarted:
    """A desk rebuilt from its JSON state before every round, as each live round runs
    in its own process."""

    def __init__(self, brain, cfg=None):
        self.brain, self.cfg, self.saved = brain, cfg, None

    def __call__(self, ctx):
        d = Desk(self.brain, self.cfg)
        d.restore(json.loads(self.saved) if self.saved else None, ctx.market.tickers)
        out = d(ctx)
        self.saved = json.dumps(d.state())
        self.desk = d
        return out


@pytest.mark.parametrize("anonymize", [False, True])
def test_a_desk_restored_before_every_round_trades_exactly_as_one_that_never_stopped(anonymize):
    """Rebuilt from nothing, a desk re-enters a book it already holds, tells the agent
    it is day 1 again and fires the same event twice in a day. Each of those is a
    trade the backtest never made, and the decision log would read as judgement."""
    bars = _bars(DAYS, 8)
    hit = (bars["ticker"] == TICKERS[4]) & (bars["start"] >= calendar.at(DAYS[63], calendar.ROUNDS[2][1]))
    bars.loc[hit, ["open", "high", "low", "close"]] *= 0.8
    m = _mkt(bars)

    def script():
        # The analyst holds, so the name stays in the book and only the desk's memory
        # of what already fired today stops it waking the analyst every round.
        return Scripted(
            review=lambda p: ReviewDecision(action="set_exposure", exposure=round(p["book"]["gross"] - 0.1, 4), reason=None,
                                            exit=[], rationale="trim"),
            event=lambda p: EventDecision(calls=[NameCall(name=t["name"], action="hold", reason="noise")
                                                 for t in p["triggers"]]))

    cfg = lambda: DeskConfig(anonymize=anonymize)  # noqa: E731
    b_whole, b_again = script(), script()
    whole = Desk(b_whole, cfg())
    want = sim.run(whole, m, START, 4)
    again = _Restarted(b_again, cfg())
    got = sim.run(again, m, START, 4)
    pd.testing.assert_frame_equal(got.ledger, want.ledger)
    strip = lambda log: [{k: v for k, v in e.items() if k != "latency_s"} for e in log]  # noqa: E731
    assert strip(again.desk.log) == strip(whole.log)
    # What the agent was shown (clock, memory, regime, NAV path, turnover) is the same too.
    assert [json.dumps(p, sort_keys=True) for _, p in b_again.seen] == \
        [json.dumps(p, sort_keys=True) for _, p in b_whole.seen]
    assert [e["role"] for e in whole.log].count("event") == 1
    assert {e["role"] for e in whole.log} == {"entry", "review", "event"}


def test_a_cached_replay_asks_the_model_nothing_and_repeats_every_decision(tmp_path):
    """A replay that re-asked the model would score a different sample each run."""
    class Counting(Scripted):
        def decide(self, *a, **k):
            self.calls = getattr(self, "calls", 0) + 1
            return super().decide(*a, **k)

    inner = Counting(entry=lambda p: EntryDecision(shape="inverse_vol", views="none", exposure=0.5,
                                                   avoid=[], rationale="x"))
    m = _mkt()
    _, first = _run(CachedBrain(inner, tmp_path), market=m)
    n = inner.calls
    cached = CachedBrain(inner, tmp_path, offline=True)
    _, second = _run(cached, market=m)
    assert inner.calls == n and cached.misses == 0 and cached.hits == n
    pd.testing.assert_frame_equal(first.ledger, second.ledger)


def test_an_offline_cache_miss_falls_back_rather_than_calling_out(tmp_path):
    desk, _ = _run(CachedBrain(Failing(), tmp_path, offline=True), n=1)
    assert desk.log[0]["source"] == "fallback" and "offline" in desk.log[0]["reason"]


# ----------------------------------------------------------------------------- ClaudeBrain

class FakeClient:
    """Stands in for anthropic.Anthropic: records the request, returns a canned reply."""

    def __init__(self, reply):
        self.reply, self.requests = reply, []

    def with_options(self, **opts):
        self.opts = opts
        return self

    @property
    def messages(self):
        return self

    def parse(self, **kw):
        self.requests.append(kw)
        return self.reply


def _reply(parsed, stop="end_turn"):
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200, cache_read_input_tokens=0,
                            cache_creation_input_tokens=0)
    return SimpleNamespace(parsed_output=parsed, stop_reason=stop, usage=usage,
                           stop_details=SimpleNamespace(category="cyber"))


def test_only_models_the_competition_allows_can_be_asked():
    """A round answered by a model the rules don't list would disqualify the entry."""
    with pytest.raises(ValueError):
        ClaudeBrain(model="claude-opus-5-5")
    assert brains.DEFAULT_MODEL in brains.ALLOWED_MODELS


def test_a_claude_request_caches_the_system_prompt_and_sends_no_model_fallback():
    ans = EntryDecision(shape="risk_parity", views="none", exposure=0.7, avoid=[], rationale="calm")
    fake = FakeClient(_reply(ans))
    brain = ClaudeBrain(client=fake)
    got = brain.decide("entry", "SYSTEM", {"a": 1}, EntryDecision, timeout=30)
    req = fake.requests[0]
    assert got == ans
    assert req["model"] == "claude-opus-5"
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "fallbacks" not in req and "betas" not in req
    assert req["output_format"] is EntryDecision
    assert fake.opts["timeout"] == 30
    assert brain.cost() == pytest.approx((1000 * 5 + 200 * 25) / 1e6)


def test_a_refused_or_truncated_answer_is_an_error_the_desk_falls_back_on():
    for stop in ("refusal", "max_tokens"):
        brain = ClaudeBrain(client=FakeClient(_reply(None, stop)))
        with pytest.raises(BrainError):
            brain.decide("entry", "S", {}, EntryDecision, timeout=5)
    brain = ClaudeBrain(client=FakeClient(_reply(None, "refusal")))
    desk, _ = _run(brain, n=1)
    assert desk.log[0]["source"] == "fallback" and "refused" in desk.log[0]["reason"]


def test_the_call_budget_stops_spending_and_the_desk_keeps_trading_on_the_rule():
    ans = EntryDecision(shape="risk_parity", views="none", exposure=0.7, avoid=[], rationale="x")
    brain = ClaudeBrain(client=FakeClient(_reply(ans)), max_calls=1)
    desk, res = _run(brain, n=3)
    assert [e["source"] for e in desk.log] == ["brain", "fallback", "fallback"]
    assert res.invalid_rounds == []


# ----------------------------------------------------------------------------- free desk

from icaif import baselines  # noqa: E402
from icaif.agents import prompts  # noqa: E402
from icaif.agents.free import FreeDesk  # noqa: E402
from icaif.agents.schemas import FreeDecision, NameWeight  # noqa: E402


def _free(brain, arm="informed", n=3, market=None):
    d = FreeDesk(brain, arm, DeskConfig(anonymize=True))
    return d, sim.run(d, market or _mkt(), START, n)


def test_a_failing_opus_trades_exactly_the_hold_we_would_submit():
    """A lookalike fallback (inverse-vol on daily closes) scored 0.13-0.20 worse on
    2025-26; every Opus-vs-fallback gap would have carried it."""
    m = _mkt()
    ref = sim.run(baselines.scaled(baselines.InverseVolHold, 0.75)(), m, START, 3)
    desk, got = _free(Failing(), market=m)
    pd.testing.assert_frame_equal(got.ledger, ref.ledger)
    assert all(e["source"] == "fallback" for e in desk.log)


def test_a_book_over_100pct_or_with_an_invented_name_is_rejected_not_rescaled():
    over = FreeDecision(action="rebalance", rationale="x",
                        weights=[NameWeight(name=f"S{i:02d}", weight=0.3) for i in range(1, 5)])
    desk, _ = _free(Scripted(free_informed=over), n=1)
    assert desk.log[0]["source"] == "fallback" and "> 1" in desk.log[0]["reason"]
    bad = FreeDecision(action="rebalance", rationale="x", weights=[NameWeight(name="AAPL", weight=0.1)])
    desk, _ = _free(Scripted(free_informed=bad), n=1)
    assert desk.log[0]["source"] == "fallback"


def test_the_blank_arm_sees_neither_the_rule_nor_the_backtest_evidence():
    """If the blank arm saw the proposal, the test of "Opus on its own" would be a
    test of Opus copying our rule."""
    b = Scripted()
    _free(b, arm="blank", n=2)
    assert all("rule_proposal" not in p for _, p in b.seen)
    assert "backtest" not in prompts.SYSTEM["free_blank"].lower()
    assert "backtest" in prompts.SYSTEM["free_informed"].lower()


def test_opus_holding_after_entry_trades_nothing_and_its_own_book_is_executed_exactly():
    pick = FreeDecision(action="rebalance", rationale="x",
                        weights=[NameWeight(name=f"S{i:02d}", weight=0.05) for i in range(1, 11)])
    hold = FreeDecision(action="hold", weights=[], rationale="x")
    answers = iter([pick, hold, hold])
    desk, res = _free(Scripted(free_informed=lambda p: next(answers)))
    traded = [p for p in res.periods if p["traded_notional"] > 0]
    assert len(traded) == 1
    assert _gross_after_trade(res, 0) == pytest.approx(0.5, abs=30 * W.GRID)
