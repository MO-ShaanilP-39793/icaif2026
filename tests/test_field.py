from datetime import date

import pandas as pd
import pytest

from icaif import baselines, kit, ranking, sim, weights, windows
from tests.test_sim import TICKERS, _market


def test_a_float_sum_that_lands_over_the_cap_is_floored_back_under_it():
    """0.1 + 0.2 fails the backend's Decimal check; after `safe` it must pass."""
    w = weights.safe({"AAPL": 0.1 + 0.2, "MSFT": 0.7 + 0.1}, TICKERS)
    kit.validate_weights(w)
    assert w["AAPL"] <= 0.30 and sum(w.values()) <= 1.0


def test_weights_over_one_in_total_are_scaled_not_rejected():
    w = weights.safe({t: 0.2 for t in TICKERS}, TICKERS)
    kit.validate_weights(w)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-4)


def test_a_nan_target_raises_instead_of_buying_the_name_to_the_cap():
    """Clamped, NaN does not become zero: min(0.30, nan) is 0.30 in Python, so a name a
    strategy could not price would be bought to the 30% cap in a valid-looking file."""
    with pytest.raises(ValueError, match="AAPL"):
        weights.safe({"AAPL": float("nan"), "MSFT": 0.2}, TICKERS)


def test_equal_metric_values_share_the_average_of_their_ranks():
    """Two all-cash books tie on every metric: each takes rank 1.5, and they share a
    position rather than one being ordered above the other by name."""
    m = pd.DataFrame({"cumulative_return": [0.0, 0.0, 0.01],
                      "sharpe_ratio": [0.0, 0.0, 2.0],
                      "maximum_drawdown": [0.0, 0.0, 0.02],
                      "turnover": [0.0, 0.0, 0.01]}, index=["a", "b", "c"])
    r = ranking.rank_window(m)
    assert r.loc["a", "rank_turnover"] == r.loc["b", "rank_turnover"] == 1.5
    assert r.loc["a", "position"] == r.loc["b", "position"]
    assert r.loc["c", "rank_cumulative_return"] == 1.0


def test_overall_score_ties_break_by_return_first():
    m = pd.DataFrame({"cumulative_return": [0.02, 0.01],
                      "sharpe_ratio": [1.0, 2.0],
                      "maximum_drawdown": [0.02, 0.01],
                      "turnover": [0.01, 0.02]}, index=["hi_ret", "hi_sharpe"])
    r = ranking.rank_window(m)
    assert r.loc["hi_ret", "overall_score"] == r.loc["hi_sharpe", "overall_score"]
    assert r.loc["hi_ret", "position"] == 1 and r.loc["hi_sharpe", "position"] == 2


def test_a_window_touching_a_degraded_day_is_skipped():
    days = [date(2026, 9, d) for d in (21, 22, 23, 24, 25)]
    m = _market(days)
    m.issues["degraded_days"] = [str(days[3])]
    assert windows.window_starts(m, n_days=2, stride=2) == [days[0]]


def test_every_window_starts_its_strategies_fresh():
    """A buy-once strategy that kept its `done` flag across windows would sit in cash
    from the second window on, and rank as a cash book without anyone noticing."""
    days = [date(2026, 9, d) for d in (21, 22, 23, 24)]
    res = windows.run_field({"ew_hold": baselines.EqualWeightHold}, _market(days),
                            [days[0], days[2]], n_days=2)
    assert (res["turnover"] > 0).all()


def test_scaling_a_hold_leaves_its_turnover_proportional_to_gross():
    days = [date(2026, 9, 21)]
    full = sim.run(baselines.EqualWeightHold(), _market(days), days[0], 1).metrics()
    half = sim.run(baselines.scaled(baselines.EqualWeightHold, 0.5)(),
                   _market(days), days[0], 1).metrics()
    assert half["turnover"] == pytest.approx(full["turnover"] / 2, rel=1e-4)
