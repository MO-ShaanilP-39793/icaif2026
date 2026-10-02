"""Our own signals in every role's observation, and the levers built on them (Roadmap step 2)."""

import json

import numpy as np
import pandas as pd
import pytest

from icaif import calendar, compiler, quant_strategies as qs, sim
from icaif import weights as W
from icaif.agents import signals
from icaif.agents.brains import RuleBrain
from icaif.agents.desk import Desk, DeskConfig, EarningsCalendar
from icaif.agents.schemas import EntryDecision, Exclusion, ReviewDecision
from tests.test_agents import Scripted
from tests.test_quant import _bars
from tests.test_sim import TICKERS, _market

# Longer than test_agents' market: HAR fits the basket on its own, and its first refit
# needs 100 training rows before the quarter starts (Oct 2025 -> the Apr 2026 refit).
DAYS = [d.date() for d in pd.bdate_range("2025-10-01", periods=150)
        if d.date() not in calendar.EARLY_CLOSES]
START = next(d for d in DAYS if d >= pd.Timestamp("2026-04-08").date())
I0 = DAYS.index(START)
REPORTER = TICKERS[0]


def _scores(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(len(DAYS), len(TICKERS))),
                        index=pd.to_datetime(DAYS), columns=TICKERS)


def _events(ahead: int = 3) -> pd.DataFrame:
    """REPORTER releases after the close so that its reaction is `ahead` sessions past START."""
    after_close = pd.Timestamp("16:30").time()
    return pd.DataFrame({"ticker": [REPORTER],
                         "accepted": [calendar.at(DAYS[I0 + ahead - 1], after_close)]})


def _world(bars=None, scores=None, events=None):
    bars = _bars(DAYS, 8) if bars is None else bars
    m = _market(DAYS, info_bars=bars)
    kw = {"vol": signals.VolForecasts.from_bars(bars, TICKERS),
          "scores": compiler.DailyPanel(_scores() if scores is None else scores, TICKERS),
          "earnings": EarningsCalendar(_events() if events is None else events, m.days)}
    return m, kw


def _entry_payload(m, kw, cfg=None):
    b = Scripted()
    sim.run(Desk(b, cfg, **kw), m, START, 1)
    return b.seen[0][1]


def _row(payload, code):
    return next(r for r in payload["names"] if r["name"] == code)


# ----------------------------------------------------------------------------- the rule

def test_a_rule_desk_reading_every_signal_trades_exactly_as_the_backtested_quant_candidate():
    """The new inputs feed only the levers an LLM can pull. If any of them reached the
    rule's own path (a views book standing in for risk parity, a rebalance preview that
    traded), every LLM-vs-rule comparison would measure that leak instead."""
    m, kw = _world()
    ref = sim.run(qs.CANDIDATES["q_riskparity_entry_regime"](), m, START, 5)
    got = sim.run(Desk(RuleBrain(), None, **kw), m, START, 5)
    pd.testing.assert_frame_equal(got.ledger, ref.ledger)
    assert got.metrics() == ref.metrics()


def test_every_role_sees_the_har_vols_the_score_rank_and_the_earnings_flag():
    """Present but null everywhere would pass every other test here and show the agent
    nothing: the inputs reach the payload, with numbers in them."""
    m, kw = _world()
    b = Scripted()
    sim.run(Desk(b, None, **kw), m, START, 3)
    entry, review = b.seen[0][1], b.seen[1][1]
    for p in (entry, review):
        assert p["market"]["vol_ann_har_1d"] > 0 and p["market"]["vol_ann_har_3d"] > 0
        assert all(r["vol_ann_har_3d"] > 0 for r in p["names"])
        assert sorted(r["model_score_rank"] for r in p["names"]) == list(range(1, 31))
    assert _row(entry, REPORTER)["earnings_in_sessions"] == 3
    assert _row(review, REPORTER)["earnings_in_sessions"] == 2
    assert "model_score_rank_at_entry" not in entry["names"][0]
    first = {r["name"]: r for r in entry["names"]}
    for r in review["names"]:
        assert r["model_score_rank_at_entry"] == first[r["name"]]["model_score_rank"]
        assert r["vol_ann_har_3d_at_entry"] == first[r["name"]]["vol_ann_har_3d"]


# ----------------------------------------------------------------------------- no look-ahead

def test_the_har_vols_the_entry_sees_are_unchanged_when_every_later_bar_is_rewritten():
    """The forecasts are built once from every bar the replay has, future included. Only
    the regressors' cut at the prior session and the refit on targets that had ended
    keep a 3x jump on the entry day out of the entry's own forecast."""
    seen = []
    for shock in (None, START):
        m, kw = _world(bars=_bars(DAYS, 8, shock_from=shock))
        seen.append(_entry_payload(m, kw))
    assert seen[0]["market"]["vol_ann_har_3d"] is not None
    assert json.dumps(seen[0], sort_keys=True) == json.dumps(seen[1], sort_keys=True)


def test_the_score_ranks_the_entry_sees_are_unchanged_when_every_later_score_is_rewritten():
    """A lookup off by one day would rank names on tomorrow's prediction, whose features
    contain today's close: the ranks would look prescient and the views would act on them."""
    later = _scores()
    later.loc[later.index > pd.Timestamp(START)] *= -1
    a, b = (_entry_payload(*_world(scores=s)) for s in (_scores(), later))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    want = _scores().loc[pd.Timestamp(START)].rank(ascending=False)
    assert {r["name"]: r["model_score_rank"] for r in a["names"]} == want.astype(int).to_dict()


def test_a_release_further_ahead_than_dates_are_announced_is_invisible_to_the_entry():
    """Replays read realised EDGAR releases. One more than ten sessions out would not
    have had its date announced yet, so a flag for it is information the live desk
    never has, and a replay that avoids it scores foresight as judgement."""
    near, edge, far = (_entry_payload(*_world(events=_events(k))) for k in (3, 10, 11))
    none = _entry_payload(*_world(events=_events(3).iloc[:0]))
    assert _row(near, REPORTER)["earnings_in_sessions"] == 3
    assert _row(edge, REPORTER)["earnings_in_sessions"] == 10
    assert json.dumps(far, sort_keys=True) == json.dumps(none, sort_keys=True)


def test_the_reviews_and_their_rebalance_previews_are_unchanged_when_the_future_is_rewritten():
    """The review shows a rebuilt book and its cost every morning, from today's scores,
    vols and closes. Rewrite every bar from day 3's open and every score after day 3:
    each round-1 observation through day 3 must read exactly as before."""
    day3 = DAYS[I0 + 2]
    later = _scores()
    later.loc[later.index > pd.Timestamp(day3)] *= -1
    seen = []
    for bars, scores in ((_bars(DAYS, 8), _scores()), (_bars(DAYS, 8, shock_from=day3), later)):
        b = Scripted(entry=_views("light"), review=_rebalancer())
        sim.run(Desk(b, None, **_world(bars=bars, scores=scores)[1]), _market(DAYS, info_bars=bars), START, 4)
        seen.append([json.dumps(p, sort_keys=True) for _, p in b.seen
                     if p["clock"]["round"] == 1 and p["clock"]["day"] <= 3])
    assert len(seen[0]) == 3 and '"weight_if_rebalanced"' in seen[0][2]
    assert seen[0] == seen[1]


def test_a_signal_asked_for_on_another_day_raises_rather_than_serving_it():
    """The door, not each caller, keeps a round-1 decision on its own day's row."""
    _, kw = _world()
    deadline = calendar.at(START, calendar.ROUNDS[1][0])
    assert kw["vol"].for_day(START, deadline).notna().all().all()
    with pytest.raises(compiler.LookAheadError):
        kw["vol"].for_day(DAYS[I0 + 1], deadline)


def test_replayed_observations_with_every_signal_still_carry_no_ticker_and_no_date():
    m, kw = _world()
    b = Scripted()
    sim.run(Desk(b, DeskConfig(anonymize=True), **kw), m, START, 3)
    text = json.dumps([p for _, p in b.seen])
    assert not [t for t in TICKERS if f'"{t}"' in text]
    assert "2026-" not in text


# ----------------------------------------------------------------------------- entry views

def _views(level, shape="risk_parity", exposure=0.6, avoid=()):
    return lambda p: EntryDecision(shape=shape, views=level, exposure=exposure, rationale="x",
                                   avoid=[Exclusion(name=c, signal="model_score", why="x")
                                          for c in avoid])


def test_a_views_entry_buys_exactly_the_book_it_was_shown():
    """The Strategist chooses a level by the book it is shown. A book built any other
    way (another covariance, the views applied after the exposure) would trade
    something it never saw, and its rationale would describe that other book."""
    m, kw = _world()
    b = Scripted(entry=_views("strong"))
    d = Desk(b, None, **kw)
    res = sim.run(d, m, START, 1)
    shown = pd.Series({r["name"]: r["weight_if_risk_parity_views_strong"] for r in b.seen[0][1]["names"]})
    rp = pd.Series({r["name"]: r["weight_if_risk_parity"] for r in b.seen[0][1]["names"]})
    bought = res.ledger.iloc[0][TICKERS] * 100 / res.periods[0]["nav_before"]
    assert (bought - shown * 0.6).abs().max() < 1e-4
    assert 0.5 * (shown - rp).abs().sum() > 0.05  # the views did move the book
    assert d.recipe == {"shape": "risk_parity", "views": "strong", "excluded": []}


@pytest.mark.parametrize("why, decide, with_scores", [
    ("inverse_vol has no views book", _views("light", shape="inverse_vol"), True),
    ("no scores, no views", _views("light"), False),
])
def test_views_the_desk_cannot_build_fall_back_to_the_rule(why, decide, with_scores):
    """Bought as the plain shape instead, the book would be one the agent did not
    choose, logged as the views it asked for."""
    m, kw = _world()
    if not with_scores:
        del kw["scores"]
    b = Scripted(entry=decide)
    d = Desk(b, None, **kw)
    sim.run(d, m, START, 1)
    assert d.log[0]["source"] == "fallback", why
    if not with_scores:
        assert "weight_if_risk_parity_views_light" not in b.seen[0][1]["names"][0]


def test_an_exclusion_keeps_its_signal_and_reason_in_the_log_and_the_name_out_of_the_book():
    m, kw = _world()
    b = Scripted(entry=lambda p: EntryDecision(
        shape="risk_parity", views="none", exposure=0.6, rationale="x",
        avoid=[Exclusion(name=REPORTER, signal="earnings", why="reports in 3 sessions")]))
    d = Desk(b, None, **kw)
    sim.run(d, m, START, 1)
    assert d.book.weights[REPORTER] == 0
    assert d.log[0]["decision"]["avoid"] == [{"name": REPORTER, "signal": "earnings",
                                              "why": "reports in 3 sessions"}]
    assert d.recipe["excluded"] == [REPORTER]


# ----------------------------------------------------------------------------- rebalance

def _rebalancer(reason="score_change", exit_first=None):
    """Review: exit `exit_first` on day 2 (if given), else rebalance every morning."""
    def review(p):
        if exit_first and p["clock"]["day"] == 2:
            return ReviewDecision(action="hold", exposure=None, reason=None, exit=[exit_first],
                                  trim=[], rationale="x")
        return ReviewDecision(action="rebalance", exposure=None, reason=reason, exit=[],
                              trim=[], rationale="the ranks moved")
    return review


def test_a_rebalance_rebuilds_the_entrys_book_on_todays_numbers_at_todays_gross():
    """It must be the entry's recipe (shape, views, exclusions) on today's inputs. Any
    other book is a new decision the Strategist never made, priced as a rebalance."""
    m, kw = _world()
    b = Scripted(entry=_views("light", avoid=(TICKERS[5],)), review=_rebalancer())
    d = Desk(b, DeskConfig(max_rebalances=1), **kw)
    res = sim.run(d, m, START, 2)
    review = [p for r, p in b.seen if r == "review"][0]
    shown = pd.Series({r["name"]: r["weight_if_rebalanced"] for r in review["names"]})
    held = res.ledger.iloc[-1][TICKERS] * 100
    nav = res.periods[-1]["nav_before"]
    assert (held / nav - shown).abs().max() < 1e-4
    assert shown[TICKERS[5]] == 0
    assert review["rebalance"]["rebalances_left"] == 1
    assert review["rebalance"]["fee_bps_of_nav"] == pytest.approx(review["rebalance"]["turnover"] * 10, abs=0.01)
    assert d.rebalances == 1


def test_a_rebalance_past_the_windows_budget_or_without_a_reason_falls_back_to_hold():
    """"Occasional" is the code's to enforce: a prompt asking nicely would let a
    persuaded agent churn every morning, each a fee and a turnover rank."""
    m, kw = _world()
    d = Desk(Scripted(entry=_views("light"), review=_rebalancer()), DeskConfig(max_rebalances=1), **kw)
    res = sim.run(d, m, START, 3)
    reviews = [e for e in d.log if e["role"] == "review"]
    assert [e["source"] for e in reviews] == ["brain", "fallback"]
    assert "rebalances are spent" in reviews[1]["reason"]
    assert sum(p["traded_notional"] > 0 for p in res.periods) == 2

    d = Desk(Scripted(entry=_views("light"), review=lambda p: ReviewDecision(
        action="rebalance", exposure=None, reason=None, exit=[], trim=[], rationale="x")), None, **kw)
    sim.run(d, m, START, 2)
    assert [e["source"] for e in d.log if e["role"] == "review"] == ["fallback"]


def test_a_rebalance_smaller_than_its_minimum_is_a_hold_and_spends_nothing():
    m, kw = _world()
    d = Desk(Scripted(entry=_views("light"), review=_rebalancer()),
             DeskConfig(rebalance_min_turnover=5.0), **kw)
    res = sim.run(d, m, START, 3)
    assert sum(p["traded_notional"] > 0 for p in res.periods) == 1
    assert d.rebalances == 0


def test_a_name_sold_in_review_stays_out_of_a_later_rebalance():
    """The recipe's exclusions grow with every exit. A rebalance that rebuilt the entry's
    book from its original list would quietly buy back what the desk had just sold."""
    m, kw = _world()
    sold = TICKERS[7]
    d = Desk(Scripted(entry=_views("light"), review=_rebalancer(exit_first=sold)), None, **kw)
    res = sim.run(d, m, START, 3)
    assert d.rebalances == 1
    assert res.ledger.iloc[-1][sold] == 0
    assert d.recipe["excluded"] == [sold]


def _held(p):
    return [r["name"] for r in p["names"] if (r.get("weight_now") or 0) > 0]


def test_a_rebalance_into_three_names_lands_on_its_exposure_not_nine_tenths_of_it():
    """A book under the cap must land on its exposure. Filled at 1.0 and then scaled, the
    three names left would each hold the cap and the book 0.9 of what was chosen
    (0.765 at 0.85), with the rationale describing a book at 0.85 and nothing else
    saying otherwise."""
    m, kw = _world()

    def review(p):
        held = _held(p)
        if len(held) > 3:
            return ReviewDecision(action="hold", exposure=None, reason=None,
                                  exit=held[: min(8, len(held) - 3)], trim=[], rationale="names out")
        return ReviewDecision(action="rebalance", exposure=0.85, reason="vol_change", exit=[],
                              trim=[], rationale="three names left; back to full size")

    entry = lambda p: EntryDecision(  # noqa: E731
        shape="inverse_vol", views="none", exposure=0.6, rationale="x",
        avoid=[Exclusion(name=r["name"], signal="other", why="x") for r in p["names"][:8]])
    d = Desk(Scripted(entry=entry, review=review), None, **kw)
    res = sim.run(d, m, START, 5)
    assert d.rebalances == 1 and len(d.recipe["excluded"]) == 27
    bought = res.ledger.iloc[-1][TICKERS] * 100 / res.periods[-7]["nav_before"]
    assert (bought > 0).sum() == 3 and bought.max() <= W.CAP
    assert bought.sum() == pytest.approx(0.85, abs=30 * W.GRID)


def test_with_every_name_kept_out_the_review_offers_no_rebalance_and_cash_comes_through_the_exposure_dial():
    """Live, a name can be on the exclusion list while still held (its exit never
    filled). Once every name is out there is no book to rebalance into; offered anyway,
    the preview would be all cash, and a rebalance would sell the book out through the
    lever meant for following the signals, spending a rebalance on it."""
    m, kw = _world()
    script = {2: ReviewDecision(action="rebalance", exposure=None, reason="score_change", exit=[],
                                trim=[], rationale="follow the ranks"),
              3: ReviewDecision(action="set_exposure", exposure=0.0, reason=None, exit=[],
                                trim=[], rationale="out of the market")}
    b = Scripted(entry=_views("light"), review=lambda p: script[p["clock"]["day"]])
    d = Desk(b, None, **kw)

    def strategy(ctx):
        if ctx.day == DAYS[I0 + 1] and ctx.round == 1:
            d.recipe["excluded"] = list(TICKERS)
        return d(ctx)

    res = sim.run(strategy, m, START, 3)
    first = [p for r, p in b.seen if r == "review"][0]
    assert first["rebalance"] is None and not any("weight_if_rebalanced" in r for r in first["names"])
    reviews = [e for e in d.log if e["role"] == "review"]
    assert [e["source"] for e in reviews] == ["fallback", "brain"]
    assert "no rebalance on offer" in reviews[0]["reason"] and d.rebalances == 0
    assert res.ledger.iloc[-1][TICKERS].abs().sum() == 0   # the dial took it to cash


def test_a_rebalance_whose_exits_leave_no_name_falls_back_rather_than_selling_out():
    m, kw = _world()
    keep = [TICKERS[3], TICKERS[9]]
    b = Scripted(entry=_views("light"), review=lambda p: ReviewDecision(
        action="rebalance", exposure=None, reason="vol_change", exit=keep, trim=[],
        rationale="sell the last two"))
    d = Desk(b, None, **kw)

    def strategy(ctx):
        if ctx.day == DAYS[I0 + 1] and ctx.round == 1:
            d.recipe["excluded"] = [t for t in TICKERS if t not in keep]
        return d(ctx)

    res = sim.run(strategy, m, START, 2)
    review = [p for r, p in b.seen if r == "review"][0]
    assert review["rebalance"] is not None   # the two names are a book to rebalance into
    e = [x for x in d.log if x["role"] == "review"][0]
    assert e["source"] == "fallback" and "set the exposure to 0" in e["reason"]
    assert d.rebalances == 0 and sum(p["traded_notional"] > 0 for p in res.periods) == 1


def test_a_desk_with_views_restored_before_every_round_trades_as_one_that_never_stopped():
    """Live, each round is a new process. A desk that lost its recipe would rebalance to
    plain risk parity, one that lost its count would rebalance past its budget, and
    one that lost the entry's signals would show the agent no change since entry."""
    m, kw = _world()

    def script():
        return Scripted(entry=_views("light", avoid=(TICKERS[2],)), review=_rebalancer())

    cfg = DeskConfig(max_rebalances=1)
    b_whole, b_again = script(), script()
    want = sim.run(Desk(b_whole, cfg, **kw), m, START, 3)
    saved = {"state": None}

    def restarted(ctx):
        d = Desk(b_again, cfg, **kw)
        d.restore(json.loads(saved["state"]) if saved["state"] else None, ctx.market.tickers)
        out = d(ctx)
        saved["state"] = json.dumps(d.state())
        return out

    got = sim.run(restarted, m, START, 3)
    pd.testing.assert_frame_equal(got.ledger, want.ledger)
    assert [json.dumps(p, sort_keys=True) for _, p in b_again.seen] == \
        [json.dumps(p, sort_keys=True) for _, p in b_whole.seen]
    assert json.loads(saved["state"])["rebalances"] == 1


# ----------------------------------------------------------------------------- the pieces

def test_score_ranks_count_from_the_best_and_leave_an_unscored_name_unranked():
    s = pd.Series([0.2, np.nan, 0.9, 0.5], index=list("abcd"))
    r = signals.score_ranks(s)
    assert r.dropna().to_dict() == {"c": 1.0, "d": 2.0, "a": 3.0}
    assert np.isnan(r["b"])


def test_a_views_level_is_offered_only_when_there_are_scores_to_take_views_from():
    rng = np.random.default_rng(3)
    rets = pd.DataFrame(rng.normal(0, 0.01, (60, 4)), columns=list("abcd"))
    prior = qs.shape_risk_parity(rets)
    assert signals.views_book(prior, rets, None, "light") is None
    assert signals.views_book(prior, rets, pd.Series(np.nan, index=list("abcd")), "light") is None
    assert signals.views_book(prior, rets, pd.Series([1.0, 2.0, 3.0, 4.0], index=list("abcd")),
                              "none") is prior
    w = signals.views_book(prior, rets, pd.Series([1.0, 2.0, 3.0, 4.0], index=list("abcd")), "light")
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all() and (w <= W.CAP + 1e-12).all()
