from datetime import date

import numpy as np
import pandas as pd
import pytest

from icaif import calendar, kit, quant, quant_strategies as qs, sim
from tests.test_sim import TICKERS, _market


def _random_cov(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    c = a @ a.T / n + np.eye(n) * 0.5
    names = [f"N{i}" for i in range(n)]
    return pd.DataFrame(c * 1e-4, index=names, columns=names)


def test_minimum_variance_sums_to_its_exposure_under_the_cap_and_beats_equal_weight():
    """A minimum-variance book above the 30% cap is rejected by the backend and the
    round holds; one below its exposure is a smaller book than the lever says."""
    cov = _random_cov(12, 1)
    w = quant.min_variance(cov, 0.75, cap=0.30)
    assert w.sum() == pytest.approx(0.75, abs=1e-6)
    assert (w <= 0.30 + 1e-9).all() and (w >= 0).all()
    ew = pd.Series(0.75 / 12, index=cov.index)
    assert w @ cov @ w < ew @ cov @ ew


def test_two_uncorrelated_names_get_minimum_variance_weights_inverse_to_variance():
    """The one case with a closed form: w_i proportional to 1/sigma_i^2."""
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], index=["a", "b"], columns=["a", "b"])
    w = quant.min_variance(cov, 1.0, cap=1.0)
    assert w["a"] == pytest.approx(0.2, abs=1e-4) and w["b"] == pytest.approx(0.8, abs=1e-4)


def test_risk_parity_gives_every_name_the_same_share_of_variance():
    """If the Newton steps stopped early, the book would read as risk parity in the
    report while one correlated cluster carried most of the risk."""
    cov = _random_cov(10, 2)
    w = quant.risk_parity(cov, 1.0, cap=1.0)
    rc = quant.risk_contributions(w, cov)
    assert rc.to_numpy() == pytest.approx(np.full(10, 0.1), abs=1e-6)


def test_an_ou_fit_recovers_the_speed_of_a_simulated_ou_path():
    rng = np.random.default_rng(3)
    kappa, m, s = 0.15, 0.2, 0.01
    b = np.exp(-kappa)
    x = np.zeros(4000)
    for t in range(1, len(x)):
        x[t] = m * (1 - b) + b * x[t - 1] + s * rng.normal()
    f = quant.fit_ou(x)
    assert f.valid
    assert f.kappa == pytest.approx(kappa, rel=0.15)
    assert f.m == pytest.approx(m, abs=0.02)
    assert f.sigma_eq == pytest.approx(s / np.sqrt(1 - b * b), rel=0.1)


def test_a_random_walk_is_not_scored_as_mean_reverting():
    """Least squares on a random walk fits a half-life of 30-750 days over 1,000 days,
    and often under the window/2 filter over 60: without the Dickey-Fuller test the
    OU tilt trades noise as stretched residuals."""
    rejected = [not quant.fit_ou(np.cumsum(np.random.default_rng(s).normal(size=60))).reverts
                for s in range(400)]
    assert np.mean(rejected) > 0.92  # a 5% test, with room for sampling error


def test_a_genuine_ou_residual_passes_the_reversion_test():
    rng = np.random.default_rng(12)
    b = np.exp(-0.3)
    x = np.zeros(120)
    for t in range(1, len(x)):
        x[t] = b * x[t - 1] + 0.01 * rng.normal()
    assert quant.fit_ou(x).reverts


def test_the_drawdown_policy_is_full_at_a_peak_half_at_half_cushion_and_zero_at_the_floor():
    assert quant.drawdown_exposure([1.0, 1.05], 0.75, 0.04) == pytest.approx(0.75)
    assert quant.drawdown_exposure([1.0, 0.98], 0.75, 0.04) == pytest.approx(0.75 * (0.02 / 0.98) / 0.04)
    assert quant.drawdown_exposure([1.0, 0.96], 0.75, 0.04) == 0.0
    assert quant.drawdown_exposure([1.0, 0.90], 0.75, 0.04) == 0.0


def test_the_no_trade_band_grows_with_the_cube_root_of_the_cost():
    a = quant.no_trade_half_width(0.001, 2.0, 0.75)
    assert quant.no_trade_half_width(0.008, 2.0, 0.75) == pytest.approx(2 * a)
    assert 0.02 < a < 0.04


def test_the_hmm_finds_a_calm_and_a_turbulent_regime_and_labels_the_turbulent_one_state_1():
    rng = np.random.default_rng(6)
    states = np.repeat([0, 1, 0, 1, 0], [300, 60, 250, 80, 300])
    r = np.where(states == 0, rng.normal(0.0005, 0.007, len(states)),
                 rng.normal(-0.001, 0.025, len(states)))
    model = quant.fit_hmm2(r)
    assert model.sigma[0] == pytest.approx(0.007, rel=0.2)
    assert model.sigma[1] == pytest.approx(0.025, rel=0.2)
    assert quant.hmm_filtered(r[:330], model) > 0.9   # inside the first storm
    assert quant.hmm_filtered(r[:300], model) < 0.1   # the calm before it


def test_the_regime_call_for_a_day_ignores_every_later_return():
    """The smoothed probability would know the storm is coming; the filtered one must
    not, or the regime policy backtests on foresight."""
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.01, 800)
    model = quant.fit_hmm2(r)
    later = r.copy()
    later[500:] = rng.normal(-0.01, 0.05, 300)
    assert quant.hmm_filtered(r[:500], model) == quant.hmm_filtered(later[:500], model)


# ----------------------------------------------------------------------------- strategies

N_DAYS = 70


def _days(n):
    return [d.date() for d in pd.bdate_range("2026-01-05", periods=n)
            if d.date() not in calendar.EARLY_CLOSES]


def _bars(days, seed, shock_from=None):
    """Random-walk hourly bars; from `shock_from` on, every price is rewritten."""
    rng = np.random.default_rng(seed)
    vol = np.linspace(0.002, 0.01, len(TICKERS))
    px = np.full(len(TICKERS), 100.0)
    rows = []
    for d in days:
        for r in calendar.rounds_for(d):
            px = px * np.exp(rng.normal(0, vol))
            shown = px * (3.0 if shock_from and d >= shock_from else 1.0)
            end = r["execution"] + pd.Timedelta(hours=1)
            if r["round"] == len(calendar.rounds_for(d)):
                end = calendar.at(d, calendar.session_close(d))
            for t, p in zip(TICKERS, shown):
                rows.append({"ticker": t, "start": r["execution"], "end": end,
                             "open": p, "high": p, "low": p, "close": p})
    return pd.DataFrame(rows)


def test_a_quant_book_decision_is_unchanged_when_every_later_bar_is_rewritten():
    """The daily panel is built once per market from all bars; a cut at the wrong
    edge would hand a round-1 decision the session it trades in."""
    days = _days(N_DAYS)
    d = days[65]
    got = []
    for shock in (None, d):
        m = _market(days, info_bars=_bars(days, 8, shock_from=shock))
        ctx = sim.RoundContext(d, 1, calendar.at(d, calendar.ROUNDS[1][0]),
                               calendar.at(d, calendar.ROUNDS[1][1]), {}, sim.INITIAL_NAV, m)
        got.append(qs.QuantBook("risk_parity", qs.Regime())(ctx))
    assert got[0] is not None
    assert got[0] == got[1]


def test_an_entry_only_book_trades_once_and_every_decision_is_legal():
    """Turnover is ranked, and the hold's single trade ties the best in the field:
    a book meant to decide at entry that re-trades on drift gives that rank away."""
    days = _days(N_DAYS)
    m = _market(days, info_bars=_bars(days, 9))
    res = sim.run(qs.book("risk_parity", qs.Regime, band=qs.ENTRY_ONLY)(), m, days[62], 5)
    assert res.invalid_rounds == []
    traded = [p for p in res.periods if p["traded_notional"] > 0]
    assert len(traded) == 1


def test_a_book_without_enough_history_holds_cash_rather_than_guessing_a_shape():
    days = _days(10)
    m = _market(days, info_bars=_bars(days, 10))
    res = sim.run(qs.book("min_variance")(), m, days[0], 3)
    assert all(p["traded_notional"] == 0 for p in res.periods)


def test_one_missing_close_drops_that_day_not_the_name_so_the_book_still_enters():
    """Dropping the name instead left risk parity refusing a partial book and sitting in
    cash for whole windows (15 of 167 in 2016-22), which ranks well enough to hide."""
    rets = pd.DataFrame(np.random.default_rng(13).normal(0, 0.01, (60, len(TICKERS))),
                        columns=TICKERS)
    rets.iloc[17, 3] = np.nan
    cov = quant.shrunk_cov(rets)
    assert list(cov.index) == TICKERS
    assert qs.shape_risk_parity(rets) is not None
    assert qs.shape_min_variance(rets) is not None
