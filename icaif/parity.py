"""How far apart are the organizer panel and the public feed, and how good is a fill
approximated from the organizer grid?

Two questions, both about numbers we would otherwise trust without checking:

1. **Source parity.** Live, our features come from Yahoo; in training they come from the
   organizer panel. If the two disagree at the open, the close or in volume, a model
   trained on one is scored live on the other, and nothing errors -- the scores just
   quietly mean something different.
2. **Fill approximation.** Rounds 2-7 execute at :30, mid-way through an organizer bar.
   A backtest on that panel has to guess the fill. The 30m public bars (Jul 2026 on)
   contain the true :30 open, so each guess can be scored against it in basis points --
   the same unit as the 10 bp cost the guess sits beside.
"""

import numpy as np
import pandas as pd


def _bps(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b - 1.0) * 1e4


def _stats(err_bps: pd.Series) -> dict:
    a = err_bps.abs()
    return {
        "n": int(err_bps.size),
        "mean_bps": round(float(err_bps.mean()), 2),
        "median_abs_bps": round(float(a.median()), 2),
        "p95_abs_bps": round(float(a.quantile(0.95)), 2),
        "p99_abs_bps": round(float(a.quantile(0.99)), 2),
        "share_over_10bps": round(float((a > 10).mean()), 4),
    }


def _daily(bars: pd.DataFrame) -> pd.DataFrame:
    day = bars["start"].dt.date.rename("day")
    g = bars.sort_values("start").groupby([day, bars["ticker"]])
    return pd.DataFrame({
        "first_start": g["start"].first(),
        "open": g["open"].first(),
        "close": g["close"].last(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "volume": g["volume"].sum(),
    })


def source_parity(organizer: pd.DataFrame, public: pd.DataFrame) -> dict:
    lo = max(organizer["start"].min(), public["start"].min())
    hi = min(organizer["end"].max(), public["end"].max())
    o = _daily(organizer[(organizer["start"] >= lo) & (organizer["end"] <= hi)])
    p = _daily(public[(public["start"] >= lo) & (public["end"] <= hi)])
    both = o.join(p, how="inner", lsuffix="_org", rsuffix="_pub")
    days_o = set(o.index.get_level_values("day"))
    days_p = set(p.index.get_level_values("day"))
    # An open is only comparable when both sources' first bar is the 09:30 bar; a
    # ticker-day missing its first organizer bar would compare 10:00 against 09:30.
    at_open = both[(both["first_start_org"].dt.time == pd.Timestamp("09:30").time())
                   & (both["first_start_pub"].dt.time == pd.Timestamp("09:30").time())]
    open_err = _bps(at_open["open_pub"], at_open["open_org"])
    close_err = _bps(both["close_pub"], both["close_org"])
    per_ticker = (close_err.abs().groupby(level="ticker").median()
                  .sort_values(ascending=False).round(2))
    return {
        "window": [str(lo), str(hi)],
        "days_organizer_only": sorted(str(d) for d in days_o - days_p),
        "days_public_only": sorted(str(d) for d in days_p - days_o),
        "ticker_days_compared": int(len(both)),
        "open_0930": _stats(open_err),
        "session_close": _stats(close_err),
        "day_high": _stats(_bps(both["high_pub"], both["high_org"])),
        "day_low": _stats(_bps(both["low_pub"], both["low_org"])),
        "volume_ratio_median": round(float((both["volume_pub"] / both["volume_org"]).median()), 4),
        "volume_ratio_p05_p95": [round(float(x), 4) for x in
                                 (both["volume_pub"] / both["volume_org"]).quantile([0.05, 0.95])],
        "close_err_worst_tickers_median_abs_bps": per_ticker.head(5).to_dict(),
        "open_err_worst_rows": _worst(at_open, open_err, "open"),
    }


def _worst(frame: pd.DataFrame, err: pd.Series, col: str, n: int = 5) -> list[dict]:
    idx = err.abs().sort_values(ascending=False).head(n).index
    return [{"day": str(d), "ticker": t, "bps": round(float(err.loc[(d, t)]), 1),
             "organizer": float(frame.loc[(d, t), f"{col}_org"]),
             "public": float(frame.loc[(d, t), f"{col}_pub"])} for d, t in idx]


def fill_approximation(bars_30m: pd.DataFrame) -> dict:
    """Score each way of guessing a :30 fill from the :00-grid bar that contains it.

    From 30m bars, rebuild the organizer grid (09:30-10:00 alone, then 10:00-11:00 as the
    10:00 and 10:30 halves, ...). The true fill for a :30 round is the open of the 30m bar
    starting at :30. The guesses may only use the containing organizer bar's OHLC.
    """
    b = bars_30m.copy()
    b["hour_start"] = b["start"].dt.floor("h")
    first_half = b[b["start"].dt.minute == 0].set_index(["ticker", "hour_start"])
    second_half = b[b["start"].dt.minute == 30].set_index(["ticker", "hour_start"])
    # 09:30 has no preceding :00 half inside the session, so round 1 drops out here: it
    # fills at the 09:30 open, which both grids observe exactly.
    pair = first_half.join(second_half, how="inner", lsuffix="_a", rsuffix="_b")
    truth = pair["open_b"]
    o, c = pair["open_a"], pair["close_b"]
    h = pair[["high_a", "high_b"]].max(axis=1)
    lo = pair[["low_a", "low_b"]].min(axis=1)
    guesses = {
        "bar_open": o,
        "bar_close": c,
        "mid_open_close": (o + c) / 2,
        "ohlc4": (o + h + lo + c) / 4,
    }
    out = {name: _stats(_bps(g, truth)) for name, g in guesses.items()}
    out["_window"] = [str(b["start"].min()), str(b["start"].max())]
    return out


def feed_self_consistency(bars_60m: pd.DataFrame, bars_30m: pd.DataFrame) -> dict:
    """Yahoo's 60m open at :30 against its own 30m open at :30 -- same vendor, two feeds.

    If these disagree, the 60m opens cannot stand in for execution prices, and the whole
    "public 60m grid = live grid" shortcut is off.
    """
    a = bars_60m.set_index(["ticker", "start"])["open"]
    b = bars_30m.set_index(["ticker", "start"])["open"]
    j = pd.concat([a.rename("m60"), b.rename("m30")], axis=1, join="inner")
    return _stats(_bps(j["m60"], j["m30"]))
