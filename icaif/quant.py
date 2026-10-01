"""Stochastic-process models for the three metrics that are about risk and cost.

Three of the four ranked metrics (Sharpe, max drawdown, turnover) reward controlling a
diffusion rather than forecasting its drift, and pass 1 found almost no drift worth
paying 0.1% to chase. So each model here answers a control question, not "which name
goes up":

- **How much to hold** (the exposure dial): a drawdown-constrained growth policy
  (Grossman & Zhou 1993), volatility targeting on an EWMA variance, and a two-state
  Gaussian HMM whose *filtered* probability of the turbulent state scales exposure.
- **In what shape**: long-only minimum variance and equal risk contribution on a
  Ledoit–Wolf shrunk covariance. Inverse-vol ignores correlation, so it overweights a
  cluster of calm names that fall together.
- **When a trade is worth its fee**: the no-trade band of the Merton problem under
  proportional cost, whose width grows with the cube root of the cost (Janeček &
  Shreve 2004).
- **Which residual has stretched**: an Ornstein–Uhlenbeck fit to each name's
  cumulative residual against the basket, read as an s-score (Avellaneda & Lee 2010).

Everything here is a pure function of the history it is handed. The one place a slip
leaks is the HMM, whose *smoothed* state probabilities use the whole sample: a
decision must read the forward (filtered) probability at its own last observation,
and `hmm_filtered` returns nothing else.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ----------------------------------------------------------------------------- shape

def shrunk_cov(returns: pd.DataFrame, min_obs: int = 20) -> Optional[pd.DataFrame]:
    """Ledoit–Wolf covariance of daily returns, on the days every name has a return.

    The sample covariance of 30 names over 60 days is badly conditioned, and a
    minimum-variance book built on it piles into whichever pair looks spuriously
    anti-correlated. Shrinkage to a scaled identity pulls those eigenvalues in.

    A hole drops the *day*, never the name. Dropping names instead meant one missing
    close anywhere in 60 days removed a name, the shape then refused a partial book,
    and a risk-parity strategy sat in cash for whole windows (15 of 167 in 2016-22).
    Cash ranks respectably, so nothing looked broken. A name with no return at all
    still drops out (it has no row to keep). Returns None below `min_obs` shared days.
    """
    from sklearn.covariance import LedoitWolf

    r = returns.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if len(r) < min_obs or r.shape[1] < 2:
        return None
    lw = LedoitWolf().fit(r.to_numpy())
    return pd.DataFrame(lw.covariance_, index=r.columns, columns=r.columns)


def _capped(w: np.ndarray, exposure: float, cap: float) -> np.ndarray:
    from icaif.compiler import _water_fill

    return _water_fill(np.maximum(w, 0.0), exposure, cap)


def min_variance(cov: pd.DataFrame, exposure: float, cap: float = 0.30) -> pd.Series:
    """Long-only minimum-variance weights summing to `exposure`, none above `cap`.

    Solved on unit exposure and scaled: the fully invested problem's solution scaled to
    `exposure` is the same shape, and scaling keeps the cap check honest.
    """
    from scipy.optimize import minimize

    n = len(cov)
    sigma = cov.to_numpy()
    unit_cap = min(1.0, cap / exposure) if exposure > 0 else 1.0
    if n * unit_cap < 1.0:
        raise ValueError(f"{n} names cannot hold exposure {exposure} under cap {cap}")
    x0 = np.full(n, 1.0 / n)
    res = minimize(lambda w: w @ sigma @ w, x0, jac=lambda w: 2 * sigma @ w,
                   bounds=[(0.0, unit_cap)] * n, method="SLSQP",
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"minimum variance did not converge: {res.message}")
    return pd.Series(_capped(res.x, exposure, cap), index=cov.index)


def risk_parity(cov: pd.DataFrame, exposure: float, cap: float = 0.30,
                iters: int = 500) -> pd.Series:
    """Equal risk contribution weights (each name adds the same share of variance).

    Spinu's convex form: minimise x'Σx/2 - Σ log(x)/n; the minimiser, normalised, has
    equal contributions. Newton steps on it are robust where the fixed-point iteration
    oscillates for correlated names.
    """
    sigma = cov.to_numpy()
    n = len(sigma)
    x = np.full(n, 1.0 / np.sqrt(np.diag(sigma).mean() * n))
    for _ in range(iters):
        grad = sigma @ x - 1.0 / (n * x)
        hess = sigma + np.diag(1.0 / (n * x ** 2))
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while np.any(x - t * step <= 0):
            t /= 2
        x = x - t * step
        if np.abs(step).max() < 1e-12:
            break
    w = x / x.sum()
    return pd.Series(_capped(w * exposure, exposure, cap), index=cov.index)


def risk_contributions(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Each name's share of portfolio variance: w_i (Σw)_i / w'Σw."""
    v = w.to_numpy()
    s = cov.loc[w.index, w.index].to_numpy()
    total = v @ s @ v
    return pd.Series(v * (s @ v) / total, index=w.index)


# The prior book's Sharpe ratio, annual, assumed. It is the risk aversion that turns
# the prior's weights into the returns they imply (pi = delta Σ w): the prior is held
# as if each unit of its risk earned this much, so a view's tilt is sized against it.
PRIOR_SHARPE = 0.5


def black_litterman(prior: pd.Series, cov: pd.DataFrame, alpha: pd.Series,
                    confidence: float, cap: float = 0.30,
                    sharpe: float = PRIOR_SHARPE) -> pd.Series:
    """Long-only weights summing to 1 (none above `cap`): `prior` moved by views.

    `alpha` is each viewed name's expected daily return above what the prior implies
    (NaN: no view on it). The view variance is Idzorek's, (1 - c) / c times the prior's
    own uncertainty about that name, so τ cancels and the posterior book is

        w = w_prior + P' (P Σ P' + k diag(P Σ P'))^-1 α / δ,   k = (1 - c) / c,

    with δ = sharpe / σ_prior (daily). `confidence` c = 0 is the prior exactly, which
    is what lets a desk with views switched off reproduce its prior's backtest; c = 1
    takes the views at face value. A name with no view gets no tilt of its own and
    moves only with the renormalisation, rather than being pushed around through its
    covariance with names that have one.

    Negative weights are cut to zero before renormalising: the book is long-only, and
    a short leg would be sold for a view the rest of the book then paid for.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence {confidence} is outside [0, 1]")
    names = prior.index
    w0 = prior.to_numpy(dtype=float)
    if confidence == 0.0:
        return prior.astype(float).copy()
    s = cov.loc[names, names].to_numpy()
    a = alpha.reindex(names).to_numpy(dtype=float)
    viewed = np.isfinite(a)
    if not viewed.any():
        return prior.astype(float).copy()
    delta = sharpe / np.sqrt(TRADING_DAYS) / np.sqrt(w0 @ s @ w0)
    k = (1.0 - confidence) / confidence
    sv = s[np.ix_(viewed, viewed)]
    tilt = np.zeros(len(names))
    tilt[viewed] = np.linalg.solve(sv + k * np.diag(np.diag(sv)), a[viewed]) / delta
    w = np.maximum(w0 + tilt, 0.0)
    if w.sum() <= 0:
        raise ValueError("views removed every name from the book")
    return pd.Series(_capped(w / w.sum(), 1.0, cap), index=names)


# ----------------------------------------------------------------------------- exposure

def drawdown_exposure(nav: np.ndarray, e_max: float, dd_limit: float) -> float:
    """The Grossman–Zhou policy: risk in proportion to the cushion above the floor.

    With the floor at (1 - dd_limit) x the running peak M, the growth-optimal risky
    fraction under a drawdown constraint is proportional to (W - (1 - d) M) / W. It is
    e_max at a new peak and reaches zero exactly at the floor, so the book can never
    lose more than `dd_limit` from its peak in continuous time. Drawdown is a ranked
    metric, and a policy that bounds it by construction trades a little return for a
    rank the hold cannot control.
    """
    nav = np.asarray(nav, dtype=float)
    peak = float(np.nanmax(nav))
    w = float(nav[-1])
    floor = (1.0 - dd_limit) * peak
    cushion = max(0.0, (w - floor) / w)
    return float(e_max * min(1.0, cushion / dd_limit))


def ewma_vol(returns: pd.Series, lam: float = 0.94) -> float:
    """RiskMetrics EWMA daily volatility at the last observation (no look past it)."""
    r = returns.dropna().to_numpy()
    if len(r) == 0:
        return float("nan")
    var = r[: min(20, len(r))].var() if len(r) > 1 else r[0] ** 2
    for x in r:
        var = lam * var + (1 - lam) * x * x
    return float(np.sqrt(var))


def vol_target_exposure(current_vol: float, reference_vol: float, e_ref: float,
                        e_min: float = 0.0, e_max: float = 1.0) -> float:
    """Scale exposure by reference / current vol (Moreira & Muir 2017).

    Volatility clusters and forecasts well; returns do not scale with it. So cutting
    exposure when vol is high buys back more drawdown than it gives up in return.
    """
    if not (np.isfinite(current_vol) and current_vol > 0 and np.isfinite(reference_vol)):
        return e_ref
    return float(np.clip(e_ref * reference_vol / current_vol, e_min, e_max))


def no_trade_half_width(cost: float, risk_aversion: float, weight: float) -> float:
    """Half-width of the optimal no-trade band around a target risky weight.

    The Merton problem with proportional cost λ has a band, not a point, and for small λ
    its half-width is (3/(2γ) · π²(1-π)² · λ)^(1/3) (Janeček & Shreve 2004; the constant
    depends on whether λ is charged per side, which here it is). The cube root is the
    point: at 0.1% the band is ~3 weight points wide at π = 0.75, far wider than the
    cost suggests, because the fee is paid on every trade while the tracking error
    only costs its square. Turnover is also ranked, so the effective cost here is higher
    than the fee, and the band should be read as a floor.
    """
    pi = float(np.clip(weight, 0.0, 1.0))
    return float((1.5 / risk_aversion * pi ** 2 * (1 - pi) ** 2 * cost) ** (1 / 3))


# ----------------------------------------------------------------------------- regime

@dataclass(frozen=True)
class HMM2:
    """A two-state Gaussian HMM on daily returns; state 1 is the high-variance one."""

    mu: np.ndarray       # (2,)
    sigma: np.ndarray    # (2,)
    trans: np.ndarray    # (2, 2), rows sum to 1
    start: np.ndarray    # (2,)

    @property
    def persistence(self) -> np.ndarray:
        """Expected stay in each state, in days: 1 / (1 - p_ii)."""
        return 1.0 / (1.0 - np.diag(self.trans))


def _emission(r: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (r[:, None] - mu[None, :]) / sigma[None, :]
    return np.exp(-0.5 * z * z) / (sigma[None, :] * np.sqrt(2 * np.pi))


def _forward(r, m: HMM2):
    b = _emission(r, m.mu, m.sigma) + 1e-300
    alpha = np.zeros_like(b)
    c = np.zeros(len(r))
    a = m.start * b[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, len(r)):
        a = (alpha[t - 1] @ m.trans) * b[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    return alpha, c, b


def fit_hmm2(returns: np.ndarray, iters: int = 200, tol: float = 1e-7) -> HMM2:
    """Baum–Welch for two Gaussian states, started from a calm/turbulent split.

    Starting the states at the 50th and 90th percentile of |r| stops EM from landing
    both on the same regime. Variances are floored so one state cannot collapse onto a
    handful of days (a degenerate likelihood that reads as a certain regime call).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 60:
        raise ValueError(f"{len(r)} returns is too few to fit a regime model")
    s = r.std()
    floor = 0.05 * s
    m = HMM2(mu=np.array([r.mean(), r.mean()]),
             sigma=np.array([np.quantile(np.abs(r), 0.5), np.quantile(np.abs(r), 0.9)]) + floor,
             trans=np.array([[0.97, 0.03], [0.08, 0.92]]), start=np.array([0.7, 0.3]))
    prev = -np.inf
    for _ in range(iters):
        alpha, c, b = _forward(r, m)
        beta = np.ones_like(alpha)
        for t in range(len(r) - 2, -1, -1):
            beta[t] = (m.trans @ (b[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = (alpha[:-1, :, None] * m.trans[None] * (b[1:] * beta[1:])[:, None, :]
              / c[1:, None, None])
        trans = xi.sum(axis=0)
        trans /= trans.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)
        mu = (gamma * r[:, None]).sum(axis=0) / w
        sigma = np.sqrt((gamma * (r[:, None] - mu) ** 2).sum(axis=0) / w)
        sigma = np.maximum(sigma, floor)
        m = HMM2(mu=mu, sigma=sigma, trans=trans, start=gamma[0])
        ll = np.log(c).sum()
        if ll - prev < tol:
            break
        prev = ll
    if m.sigma[0] > m.sigma[1]:  # label state 1 as the turbulent one
        m = HMM2(mu=m.mu[::-1], sigma=m.sigma[::-1], trans=m.trans[::-1, ::-1],
                 start=m.start[::-1])
    return m


def hmm_filtered(returns: np.ndarray, model: HMM2) -> float:
    """P(turbulent state today | returns up to today): the forward probability only.

    The smoothed probability of the same day would condition on later returns, and a
    regime call that knows the crash is coming reads as a very good exposure rule.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    alpha, _, _ = _forward(r, model)
    return float(alpha[-1, 1])


def hmm_next_turbulent(returns: np.ndarray, model: HMM2) -> float:
    """P(turbulent state tomorrow | today's data): one transition past the filter."""
    p = hmm_filtered(returns, model)
    return float(np.array([1 - p, p]) @ model.trans[:, 1])


# ----------------------------------------------------------------------------- residual reversion

# Dickey–Fuller 5% critical value for the regression with a constant (MacKinnon).
DF_CRITICAL_5PCT = -2.86


@dataclass(frozen=True)
class OUFit:
    kappa: float      # mean-reversion speed per day
    m: float          # long-run level
    sigma_eq: float   # stationary standard deviation
    half_life: float  # days
    df_t: float       # Dickey–Fuller t-statistic of (b - 1)

    @property
    def valid(self) -> bool:
        return np.isfinite(self.kappa) and self.kappa > 0 and self.sigma_eq > 0

    @property
    def reverts(self) -> bool:
        """Mean reversion the data can tell apart from a random walk.

        Least squares on a random walk's AR(1) is biased below b = 1, so a random walk
        of 1,000 days fits a half-life of 30 to 750 days, and one of 60 days often
        fits under 30: a half-life filter alone passes noise as a stretched residual.
        The Dickey–Fuller test is built for exactly that bias.
        """
        return self.valid and self.df_t < DF_CRITICAL_5PCT


def fit_ou(x: np.ndarray) -> OUFit:
    """Discrete OU: X_{t+1} = a + b X_t + e is the exact discretisation with
    b = exp(-κ), so κ = -ln b, m = a / (1 - b), σ_eq = sd(e) / sqrt(1 - b²).

    b outside (0, 1) is not a mean-reverting process at all (a random walk or an
    oscillation), and the fit says so rather than returning a huge or negative κ.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    nan = OUFit(np.nan, np.nan, np.nan, np.nan, np.nan)
    if len(x) < 10:
        return nan
    b, a = np.polyfit(x[:-1], x[1:], 1)
    e = x[1:] - (a + b * x[:-1])
    lag = x[:-1] - x[:-1].mean()
    se_b = e.std(ddof=2) / np.sqrt((lag ** 2).sum())
    df_t = (b - 1) / se_b if se_b > 0 else np.nan
    if not 0 < b < 1:
        return OUFit(np.nan, np.nan, np.nan, np.nan, df_t)
    kappa = -np.log(b)
    return OUFit(kappa=kappa, m=a / (1 - b), sigma_eq=e.std(ddof=2) / np.sqrt(1 - b * b),
                 half_life=np.log(2) / kappa, df_t=df_t)


def s_scores(returns: pd.DataFrame, max_half_life: Optional[float] = None) -> pd.DataFrame:
    """Avellaneda–Lee s-scores of each name's residual against the equal-weight basket.

    Regress each name's daily returns on the basket's, cumulate the residual into X_t,
    fit an OU to X and score s = (X_T - m) / σ_eq. s < 0 is a name stretched *below*
    its residual mean. A name gets NaN rather than a score read off noise unless its
    residual passes the Dickey–Fuller test and its half-life is within
    `max_half_life` (default: half the window, Avellaneda and Lee's filter, which on
    its own passes random walks; see `OUFit.reverts`).
    """
    r = returns.dropna(axis=0, how="all")
    mkt = r.mean(axis=1)
    n = len(r)
    limit = max_half_life if max_half_life is not None else n / 2
    rows = {}
    for t in r.columns:
        y = r[t]
        ok = y.notna() & mkt.notna()
        if ok.sum() < 20:
            continue
        beta, alpha = np.polyfit(mkt[ok], y[ok], 1)
        x = np.cumsum((y[ok] - beta * mkt[ok] - alpha).to_numpy())
        f = fit_ou(x)
        s = (x[-1] - f.m) / f.sigma_eq if f.reverts and f.half_life <= limit else np.nan
        rows[t] = {"beta": beta, "kappa": f.kappa, "half_life": f.half_life,
                   "df_t": f.df_t, "s": s}
    return pd.DataFrame(rows).T.reindex(returns.columns)
