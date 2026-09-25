"""Features at each decision time, from bars that had ended by the round's deadline.

One row per (execution time, ticker). Every feature is defined in **sessions or clock
time, never in bars**, because the information grid changes on 2026-01-01 (organizer
bars end on :00, public bars on :30). "Return over the last 6 bars" means a different
span on each side of that join; "return since the prior session's close" does not.

Two kinds of column:
- per-ticker features, converted to cross-sectional ranks in [-0.5, 0.5] at each decision
  time. Ranks rather than raw values, because the model's job is ordering 30 names, and
  a raw 2% move means something different in a 15-vol month and a 40-vol one;
- `ctx_*` market-wide context (identical for every ticker at a decision), left raw.

Today's session is never "complete" at a deadline, so session features use sessions
strictly before today; only `last` (the latest completed bar's close) and the cumulative
volume so far reach into today.
"""

import numpy as np
import pandas as pd

from icaif import calendar, data

RETURN_SESSIONS = (1, 2, 5, 10, 20)


def sessions(info_bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-session OHLCV panels (index = session date, columns = tickers)."""
    b = info_bars.sort_values("start")
    day = b["start"].dt.date
    g = b.groupby([day, "ticker"])
    frame = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                          "low": g["low"].min(), "close": g["close"].last(),
                          "volume": g["volume"].sum()})
    return {c: frame[c].unstack("ticker").sort_index() for c in frame.columns}


def decision_times(info_bars: pd.DataFrame) -> pd.DataFrame:
    days = sorted(set(info_bars["start"].dt.date))
    return pd.DataFrame([{"day": d, **r} for d in days for r in calendar.rounds_for(d)])


def _cum_volume_to_deadlines(info_bars: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    """Volume traded in today's session in bars ended by each deadline (decision x ticker)."""
    vol = info_bars.pivot_table(index="end", columns="ticker", values="volume",
                                aggfunc="sum").sort_index()
    cum = vol.groupby(vol.index.date).cumsum()
    # For each deadline: the cumulative volume of the last bar ended by then, but only if
    # that bar belongs to the same session; otherwise nothing has traded today yet.
    at = cum.reindex(pd.DatetimeIndex(decisions["deadline"]), method="ffill")
    last_end = pd.Series(vol.index, index=vol.index).reindex(
        pd.DatetimeIndex(decisions["deadline"]), method="ffill")
    same_day = np.array([le is not pd.NaT and le.date() == dl.date()
                         for le, dl in zip(last_end, decisions["deadline"])])
    at = at.where(pd.DataFrame(np.repeat(same_day[:, None], at.shape[1], axis=1),
                               index=at.index, columns=at.columns), 0.0)
    at.index = pd.DatetimeIndex(decisions["execution"])
    return at


def centred_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank scaled to [-0.5, 0.5], with an all-tied row at exactly 0.

    Pandas' `pct=True` puts a fully tied row at (n+1)/2n, i.e. +0.017 for 30 names, so a
    round-1 decision (where nothing has traded today and every name ties) would carry a
    small, constant, non-zero "signal".
    """
    r = panel.rank(axis=1)
    n = panel.notna().sum(axis=1)
    return (r.sub(1).div((n - 1).where(n > 1), axis=0)) - 0.5


def build(info_bars: pd.DataFrame) -> pd.DataFrame:
    """The feature frame: index (execution, ticker)."""
    universe = data.load_universe()
    tickers = sorted(universe)
    decisions = decision_times(info_bars)
    s = sessions(info_bars)
    s = {k: v.reindex(columns=tickers) for k, v in s.items()}
    days = list(s["close"].index)
    pos = {d: i for i, d in enumerate(days)}

    closes = (info_bars.pivot_table(index="end", columns="ticker", values="close",
                                    aggfunc="last").sort_index().reindex(columns=tickers).ffill())
    last = closes.reindex(pd.DatetimeIndex(decisions["deadline"]), method="ffill")
    last.index = pd.DatetimeIndex(decisions["execution"])

    logret = np.log(s["close"]).diff()
    gap = np.log(s["open"] / s["close"].shift())
    park = np.sqrt((np.log(s["high"] / s["low"]) ** 2) / (4 * np.log(2)))
    ew = logret.mean(axis=1)
    session_feats = {
        "vol_5d": logret.rolling(5).std(),
        "vol_20d": logret.rolling(20).std(),
        "parkinson_5d": park.rolling(5).mean(),
        "gap_prev": gap,
        "overnight_share_20d": (gap ** 2).rolling(20).sum() / (logret ** 2).rolling(20).sum(),
        "volume_1d_ratio": s["volume"] / s["volume"].rolling(20).mean(),
        "high_20d": s["high"].rolling(20).max(),
        "low_20d": s["low"].rolling(20).min(),
        "mean_close_5d": s["close"].rolling(5).mean(),
    }
    ctx_session = pd.DataFrame({"ctx_mkt_vol_20d": ew.rolling(20).std(),
                                "ctx_mkt_ret_5d": ew.rolling(5).sum()})

    # Row j of `prev` = the last complete session before decision j's day.
    prev_pos = np.array([pos.get(d, np.nan) for d in decisions["day"]], dtype=float) - 1
    ok = prev_pos >= 0
    prev_idx = np.where(ok, prev_pos, 0).astype(int)

    def at_prev(panel: pd.DataFrame, lag: int = 0) -> pd.DataFrame:
        k = prev_idx - lag
        vals = panel.to_numpy()[np.clip(k, 0, None)]
        vals[(k < 0) | ~ok] = np.nan
        return pd.DataFrame(vals, index=last.index, columns=panel.columns)

    raw = {}
    for k in RETURN_SESSIONS:
        raw[f"ret_{k}s"] = last / at_prev(s["close"], lag=k - 1) - 1
    vol20 = at_prev(session_feats["vol_20d"])
    raw["vol_5d"] = at_prev(session_feats["vol_5d"])
    raw["vol_20d"] = vol20
    raw["vol_ratio"] = raw["vol_5d"] / vol20
    raw["parkinson_5d"] = at_prev(session_feats["parkinson_5d"])
    raw["gap_prev"] = at_prev(session_feats["gap_prev"])
    raw["overnight_share_20d"] = at_prev(session_feats["overnight_share_20d"])
    raw["volume_1d_ratio"] = at_prev(session_feats["volume_1d_ratio"])
    raw["dist_high_20d"] = (last / at_prev(session_feats["high_20d"]) - 1) / vol20
    raw["dist_low_20d"] = (last / at_prev(session_feats["low_20d"]) - 1) / vol20
    raw["z_5d"] = (last / at_prev(session_feats["mean_close_5d"]) - 1) / vol20

    cum = _cum_volume_to_deadlines(info_bars, decisions).reindex(columns=tickers)
    # Same clock cutoff in each of the prior 20 sessions: a volume ratio against a
    # full-session average would read every morning as unusually quiet.
    tod = decisions["deadline"].dt.time.to_numpy()
    cum_hist = cum.copy()
    cum_hist["_tod"] = tod
    base = (cum_hist.groupby("_tod", group_keys=False)
            .apply(lambda g: g.drop(columns="_tod").shift().rolling(20, min_periods=10).mean()))
    base = base.reindex(cum.index)
    raw["volume_today_ratio"] = (cum / base).where(cum > 0)

    # Sector-relative only. A market-relative return (minus the 30-name mean) is not a
    # feature at all once ranked: subtracting a constant across the cross-section leaves
    # every rank unchanged, so it would be an exact duplicate of `ret_Ks` that doubles
    # that signal's weight in any importance-based pruning.
    sector = pd.Series(universe)
    rel = {}
    for k in RETURN_SESSIONS:
        r = raw[f"ret_{k}s"]
        sec_mean = r.T.groupby(sector.reindex(r.columns).to_numpy()).transform("mean").T
        rel[f"ret_{k}s_sec"] = r - sec_mean
    raw.update(rel)

    ranked = {name: centred_rank(f) for name, f in raw.items()}
    long = pd.concat({name: f.stack(future_stack=True) for name, f in ranked.items()}, axis=1)
    long.index.names = ["execution", "ticker"]

    r1 = raw["ret_1s"]
    ctx = pd.DataFrame({
        "ctx_round": decisions["round"].to_numpy(),
        "ctx_weekday": pd.DatetimeIndex(decisions["execution"]).weekday,
        "ctx_mkt_ret_1s": r1.mean(axis=1).to_numpy(),
        "ctx_dispersion_1s": r1.std(axis=1).to_numpy(),
        "ctx_breadth_1s": (r1 > 0).mean(axis=1).to_numpy(),
        "ctx_mkt_vol_20d": at_prev(ctx_session[["ctx_mkt_vol_20d"]]).iloc[:, 0].to_numpy(),
        "ctx_mkt_ret_5d": at_prev(ctx_session[["ctx_mkt_ret_5d"]]).iloc[:, 0].to_numpy(),
    }, index=last.index)
    out = long.join(ctx, on="execution")
    return out.sort_index()
