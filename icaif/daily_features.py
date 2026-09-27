"""Daily-model features: one decision a day, before the open, over the broad universe.

One row per (date, ticker) where the ticker is in that day's point-in-time universe
(`universe.build`). The decision is round 1 (deadline 09:10, fill at the 09:30 open),
so a row dated d may use data through the close of d-1 and nothing of d. Every panel
is computed "as of the close of t" and then shifted one session. Getting that shift
wrong is the classic silent leak: a feature that includes day d's close reads as a
strong signal in backtests and doesn't exist live.

Three kinds of column:
- per-name features, ranked within the day's universe to [-0.5, 0.5]. A name outside
  the universe never moves anyone else's rank;
- `e_*` earnings timing, raw (sessions are comparable across names, so ranking would
  throw away the scale);
- `ctx_*` market context, identical across a day's rows, raw.

Rows with zero volume are Yahoo placeholders carrying a stale price. They are read as
missing, not as a flat day: a stale price is a zero return that never happened,
followed by a catch-up jump.

Sector-relative returns are left out for now. The organizer file gives sectors for
the 30, but not for the broad universe.
"""

import numpy as np
import pandas as pd

from icaif import calendar, data, earnings, labels
from icaif.features import centred_rank

RETURN_SESSIONS = (1, 2, 5, 10, 20, 60, 120)
REACTION_WINDOW = 60
DEADLINE = pd.Timestamp("09:10").time()
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]


def panels(daily: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """date x ticker OHLCV panels, with zero-volume placeholder rows set to NaN."""
    d = daily.copy()
    stale = d["volume"] <= 0
    d.loc[stale, ["open", "high", "low", "close", "volume"]] = np.nan
    return {c: d.pivot_table(index="date", columns="ticker", values=c, aggfunc="last",
                             dropna=False).sort_index()
            for c in ("open", "high", "low", "close", "volume")}


def _as_of_close(p: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Per-name features known at the close of each date (not yet shifted)."""
    close, high, low, vol = p["close"], p["high"], p["low"], p["volume"]
    logret = np.log(close).diff()
    vol20 = logret.rolling(20, min_periods=15).std()
    gap = np.log(p["open"] / close.shift())
    park = np.sqrt(np.log(high / low) ** 2 / (4 * np.log(2)))
    f = {f"ret_{k}d": close / close.shift(k) - 1 for k in RETURN_SESSIONS}
    f.update({
        "mom_12_1": close.shift(20) / close.shift(250) - 1,
        "vol_5d": logret.rolling(5, min_periods=4).std(),
        "vol_20d": vol20,
        "vol_60d": logret.rolling(60, min_periods=45).std(),
        "parkinson_20d": park.rolling(20, min_periods=15).mean(),
        "gap_last": gap,
        "overnight_share_20d": (gap ** 2).rolling(20, min_periods=15).sum()
                               / (logret ** 2).rolling(20, min_periods=15).sum(),
        "volume_1d_ratio": vol / vol.rolling(20, min_periods=15).mean(),
        "dist_high_20d": (close / high.rolling(20, min_periods=15).max() - 1) / vol20,
        "dist_low_20d": (close / low.rolling(20, min_periods=15).min() - 1) / vol20,
        "dist_high_250d": (close / high.rolling(250, min_periods=200).max() - 1) / vol20,
        "z_5d": (close / close.rolling(5).mean() - 1) / vol20,
    })
    f["vol_ratio"] = f["vol_5d"] / f["vol_20d"]
    # The move on a release's reaction day, in units of the name's prior volatility.
    f["_reaction"] = (close / close.shift() - 1) / vol20.shift()
    return f


def context(ctx_daily: pd.DataFrame) -> pd.DataFrame:
    """Market context known at each close (date index, not yet shifted)."""
    w = ctx_daily.pivot_table(index="date", columns="ticker", values="close").sort_index()
    # The equity session calendar is SPY's. Yahoo's ^VIX prints on some exchange holidays
    # (Memorial Day and Labor Day 2026) when SPY doesn't, and a union of dates puts a NaN
    # SPY row there: every 20-session SPY return and vol after it is NaN for a month.
    w = w[w["SPY"].notna()]
    spy = np.log(w["SPY"]).diff()
    sectors = [s for s in SECTOR_ETFS if s in w]
    sec5 = w[sectors].pct_change(5, fill_method=None)
    return pd.DataFrame({
        "ctx_vix": w["^VIX"],
        "ctx_vix_chg_5d": np.log(w["^VIX"]).diff(5),
        "ctx_spy_ret_1d": spy,
        "ctx_spy_ret_5d": spy.rolling(5).sum(),
        "ctx_spy_ret_20d": spy.rolling(20).sum(),
        "ctx_spy_vol_20d": spy.rolling(20).std(),
        "ctx_rate_10y": w["^TNX"],
        "ctx_curve_10y_3m": w["^TNX"] - w["^IRX"],
        "ctx_rate_10y_chg_20d": w["^TNX"].diff(20),
        # Sector rotation: how far apart the sector ETFs' 5-day returns are. XLRE and
        # XLC join when they list; before that the spread is over fewer sectors.
        "ctx_sector_dispersion_5d": sec5.std(axis=1),
    })


def build(daily: pd.DataFrame, universe_mask: pd.DataFrame, ctx_daily: pd.DataFrame,
          events: pd.DataFrame | None = None) -> pd.DataFrame:
    """The daily feature frame: index (date, ticker), universe rows only."""
    p = panels(daily)
    dates = p["close"].index
    raw = {k: v.shift(1) for k, v in _as_of_close(p).items()}  # row d = as of close d-1
    mask = universe_mask.reindex(index=dates, columns=p["close"].columns).fillna(False).astype(bool)

    ranked = {k: centred_rank(v.where(mask)) for k, v in raw.items() if not k.startswith("_")}
    long = pd.concat({k: v.where(mask).stack() for k, v in ranked.items()}, axis=1)
    long.index.names = ["date", "ticker"]
    long = long[mask.stack().reindex(long.index).fillna(False).to_numpy()]

    prior_ret = raw["ret_1d"].where(mask)
    ctx = context(ctx_daily).shift(1).reindex(dates)
    ctx["ctx_breadth_1d"] = (prior_ret > 0).sum(axis=1) / prior_ret.notna().sum(axis=1)
    ctx["ctx_dispersion_1d"] = prior_ret.std(axis=1)
    ctx["ctx_weekday"] = dates.weekday
    out = long.join(ctx, on="date")
    out["is_competition"] = out.index.get_level_values("ticker").isin(
        list(data.load_universe())).astype(float)

    if events is not None:
        out = out.join(_earnings_columns(events, dates, raw["_reaction"], out.index))
    return out.sort_index()


def _earnings_columns(events: pd.DataFrame, dates: pd.DatetimeIndex,
                      reaction_asof_prior: pd.DataFrame, index: pd.MultiIndex) -> pd.DataFrame:
    decisions = pd.DataFrame({"day": dates, "deadline": [
        calendar.at(d.date(), DEADLINE) for d in dates]})
    prox = earnings.proximity(events, decisions, dates)
    prox["date"] = dates[prox["row"].to_numpy()]
    prox = prox.set_index(["date", "ticker"]).reindex(index)
    # The last release's reaction-day move, available from the session after it
    # (row d of reaction_asof_prior holds the move on d-1).
    since = prox["since"].to_numpy()
    date_pos = dates.get_indexer(index.get_level_values("date"))
    col_pos = reaction_asof_prior.columns.get_indexer(index.get_level_values("ticker"))
    ok = (since >= 1) & (since <= REACTION_WINDOW) & (col_pos >= 0)
    rows = np.where(ok, date_pos - np.nan_to_num(since).astype(int) + 1, 0)
    vals = reaction_asof_prior.to_numpy()[rows.clip(0, len(dates) - 1), col_pos.clip(0)]
    reaction = np.where(ok, vals, np.nan)
    return pd.DataFrame({"e_sessions_since": prox["since"].to_numpy(),
                         "e_sessions_to_next": prox["to_next"].to_numpy(),
                         "e_last_reaction": reaction}, index=index)


def build_labels(daily: pd.DataFrame, universe_mask: pd.DataFrame) -> pd.DataFrame:
    """Daily labels: entry at day d's 09:30 open, path the opens of d+1 .. d+h.

    The daily open is the round-1 fill (organizer vs Yahoo 09:30 open: median 0,
    p95 26 bps). Ranked within day d's universe.
    """
    opens = panels(daily)["open"]
    out = labels.build(opens, labels.DAILY_HORIZONS, upside_horizons=tuple(labels.DAILY_HORIZONS),
                       per_session=1, universe=universe_mask)
    out.index = out.index.set_names(["date", "ticker"])
    return out
