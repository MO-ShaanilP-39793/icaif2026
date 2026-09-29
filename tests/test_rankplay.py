import numpy as np
import pandas as pd
import pytest

from icaif import baselines, calendar, ranking, rankplay, sim
from tests.test_quant import _bars, _days
from tests.test_sim import TICKERS

DAYS = _days(30)


def _walk_market(days, seed, shock_ts=None, info_bars=None):
    """Random-walk fills; each close equals that day's 15:30 fill, so nothing moves in
    the unmodelled stub and the planner's finish can match the kit exactly."""
    rng = np.random.default_rng(seed)
    idx = [r["execution"] for d in days for r in calendar.rounds_for(d)]
    steps = np.exp(rng.normal(0, 0.004, (len(idx), len(TICKERS))))
    ex = pd.DataFrame(100 * np.cumprod(steps, axis=0), index=idx, columns=TICKERS)
    if shock_ts is not None:
        ex.loc[ex.index >= shock_ts] *= 3.0
    closes = pd.DataFrame(
        [ex.loc[calendar.rounds_for(d)[-1]["execution"]].to_numpy() for d in days],
        index=[calendar.at(d, calendar.session_close(d)) for d in days], columns=TICKERS)
    if info_bars is None:
        info_bars = pd.DataFrame({"end": pd.Series([], dtype=f"datetime64[ns, {calendar.TZ}]")})
    return sim.Market(ex, closes, info_bars)


W1 = dict(zip(TICKERS, np.round(np.linspace(0.01, 0.04, len(TICKERS)), 6)))
W2 = dict(zip(TICKERS, np.round(np.linspace(0.035, 0.005, len(TICKERS)), 6)))


def _script(plan):
    return lambda ctx: plan.get((ctx.day, ctx.round))


def _actual_rel(m, after_ts, until_day):
    ex = m.exec_prices
    seq = ex[(ex.index >= after_ts) & (ex.index <= calendar.rounds_for(until_day)[-1]["execution"])]
    return (seq.to_numpy()[1:] / seq.to_numpy()[:-1])[None, :, :]


def _deadline(d):
    return calendar.at(d, calendar.ROUNDS[1][0])


def _close_to(f, metrics):
    assert f.ret[0] == pytest.approx(metrics["cumulative_return"], rel=1e-9, abs=1e-12)
    assert f.sharpe[0] == pytest.approx(metrics["sharpe_ratio"], rel=1e-9)
    assert f.mdd[0] == pytest.approx(metrics["maximum_drawdown"], rel=1e-9, abs=1e-12)
    assert f.turnover[0] == pytest.approx(metrics["turnover"], rel=1e-9)


def test_fed_the_path_that_happened_the_planner_scores_a_hold_exactly_as_the_kit_does():
    """If the planner's metrics drifted from the kit's formulas, it would optimise a
    score the leaderboard never computes, and look clever doing it."""
    days = DAYS[:4]
    m = _walk_market(days, 1)
    res = sim.run(_script({(days[0], 1): W1}), m, days[0], 4)
    st = rankplay.state_at(res, _deadline(days[1]), m)
    rel = _actual_rel(m, calendar.rounds_for(days[0])[-1]["execution"], days[-1])
    _close_to(rankplay.finish(st, rel, len(res.periods)), res.metrics())


def test_fed_the_path_that_happened_a_retarget_costs_what_the_simulator_charges():
    days = DAYS[:4]
    m = _walk_market(days, 2)
    hold = sim.run(_script({(days[0], 1): W1}), m, days[0], 4)
    moved = sim.run(_script({(days[0], 1): W1, (days[1], 1): W2}), m, days[0], 4)
    st = rankplay.state_at(hold, _deadline(days[1]), m)
    rel = _actual_rel(m, calendar.rounds_for(days[0])[-1]["execution"], days[-1])
    target = np.array([W2[t] for t in TICKERS])
    _close_to(rankplay.finish(st, rel, len(moved.periods), target=target), moved.metrics())


def test_on_the_first_morning_the_planner_opens_the_book_at_the_first_fill():
    """Nothing is in progress before the window's first fill; a phantom zero period
    there would shift every entrant's Sharpe and turnover."""
    days = DAYS[:5]
    m = _walk_market(days, 3)
    res = sim.run(_script({(days[1], 1): W1}), m, days[1], 4)
    px = m.exec_prices.loc[calendar.rounds_for(days[0])[-1]["execution"]].to_numpy()
    st = rankplay.EntrantState(np.zeros(len(TICKERS)), sim.INITIAL_NAV, sim.INITIAL_NAV, px,
                               np.array([]), np.array([]), sim.INITIAL_NAV, 0.0, 0.0,
                               target=np.array([W1[t] for t in TICKERS]), started=False)
    rel = _actual_rel(m, calendar.rounds_for(days[0])[-1]["execution"], days[4])
    _close_to(rankplay.finish(st, rel, len(res.periods)), res.metrics())


def test_the_planners_ranks_match_the_competitions_ties_included():
    rng = np.random.default_rng(4)
    for _ in range(200):
        n = 6
        vals = rng.choice([0.0, 0.01, 0.02, -0.01], size=(n, 4))
        metrics = pd.DataFrame(vals, index=[f"e{i}" for i in range(n)],
                               columns=list(ranking.METRICS))
        want = ranking.rank_window(metrics).loc[f"e{n - 1}", "overall_score"]
        fins = [rankplay.Finish(*(np.array([v]) for v in row)) for row in vals]
        got = rankplay.expected_scores({"x": fins[-1]}, fins[:-1])["x"]
        assert got == pytest.approx(want)


PLAN_FIELD = {"cash": baselines.Cash, "ew_hold": baselines.EqualWeightHold,
              "random_churn": baselines.RandomChurn}


def _player():
    return rankplay.RankPlayer(rankplay.RankPlayConfig(
        plan_field=PLAN_FIELD, window_days=5, n_paths=40, pool=20, block=3))


def test_a_morning_decision_is_unchanged_when_every_later_price_is_rewritten():
    """The field's to-date metrics come from full-window runs cut at the deadline; a
    cut on the wrong side would hand the planner the rest of the window."""
    d = DAYS[24]
    shock = calendar.rounds_for(d)[0]["execution"]
    logs = []
    for s in (None, shock):
        m = _walk_market(DAYS, 5, shock_ts=s,
                         info_bars=_bars(DAYS, 6, shock_from=d if s is not None else None))
        p = _player()
        sim.run(p, m, DAYS[22], 5)
        logs.append(p.log[2])
    assert logs[0]["day"] == d
    assert logs[0] == logs[1]


def test_a_rank_player_only_ever_submits_legal_books():
    m = _walk_market(DAYS, 7, info_bars=_bars(DAYS, 8))
    p = _player()
    res = sim.run(p, m, DAYS[22], 5)
    assert res.invalid_rounds == []
    assert len(p.log) == 5
    assert all(p_["traded_notional"] == 0 for p_ in res.periods
               if p_["execution"].hour != 9)  # it only ever acts at round 1
