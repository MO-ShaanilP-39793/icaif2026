"""Where prices come from: fills, labels and what a decision may look at.

Alpaca is the organizer's own vendor (reports/alpaca_parity.json: its bars match the
organizer panel to 0 bps). So from 2016 everything reads Alpaca's bars paired into the
live :30 grid. Yahoo 60m bars, on the same grid, take over after Alpaca's last fetch
and are what the live system reads. The organizer panel (:00 grid, 2021-25) is now a
fallback for holes and a cross-check.
"""

import glob

import pandas as pd

from icaif import calendar, data, sim


def latest_public(interval: str) -> pd.DataFrame:
    paths = sorted(glob.glob(str(data.ROOT / "data" / "public" / f"yahoo_{interval}_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no cached yahoo_{interval} snapshot; run tools/data_report.py")
    return pd.read_parquet(paths[-1])


def organizer_exec_prices(organizer: pd.DataFrame) -> pd.DataFrame:
    """Best guess of each round's fill from the organizer grid (execution ts x ticker).

    Round 1 fills at the 09:30 open, which the organizer grid observes exactly. Rounds
    2-7 fill at :30, inside a :00-grid bar; the guess is that bar's (O+H+L+C)/4, the
    lowest-error guess measured in reports/data_parity.json (median 11 bps, p95 45 bps,
    unbiased). That is fine for a 1-3 day label, and not fine for scoring a strategy,
    which is why the simulator fills on public 60m bars instead.
    """
    o = organizer.assign(ohlc4=organizer[["open", "high", "low", "close"]].mean(axis=1))
    opens = o.pivot(index="start", columns="ticker", values="open")
    mids = o.pivot(index="start", columns="ticker", values="ohlc4")
    days = sorted({ts.date() for ts in opens.index})
    rows, idx = [], []
    for d in days:
        for r in calendar.rounds_for(d):
            ex = r["execution"]
            if r["round"] == 1:
                rows.append(opens.reindex([ex]).iloc[0])
            else:
                rows.append(mids.reindex([ex.floor("h")]).iloc[0])
            idx.append(ex)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="execution"))


def public_exec_opens(bars_60m: pd.DataFrame) -> pd.DataFrame:
    """Each round's exact fill from public 60m bars, with holes left as NaN.

    Unlike `sim.market_from_public_60m`, nothing stands in for a missing bar. A stand-in
    (the last close) is a zero return followed by a catch-up move, which is harmless
    for valuing a ledger but writes a fake path into any label that spans it.
    """
    opens = bars_60m.pivot(index="start", columns="ticker", values="open").sort_index()
    days = sorted({ts.date() for ts in opens.index})
    exec_ts = pd.DatetimeIndex([r["execution"] for d in days for r in calendar.rounds_for(d)],
                               name="execution")
    out = opens.reindex(exec_ts)
    everyone_seen = out.notna().cummax().all(axis=1)
    return out[everyone_seen.to_numpy()]


def merge_fills(exact: pd.DataFrame, guessed: pd.DataFrame) -> pd.DataFrame:
    """Exact fills where they exist, the organizer-grid guess only for their holes.

    Never forward-filled: a round with neither stays NaN, so every label whose path
    crosses it is dropped rather than computed on a price that never traded.
    """
    exact = exact.reindex(columns=guessed.columns)
    before = guessed[guessed.index < exact.index.min()]
    during = exact.combine_first(guessed.reindex(exact.index))
    return pd.concat([before, during]).sort_index()


def latest_alpaca_60m() -> pd.DataFrame:
    """Alpaca's 30m bars paired into the live 60m grid (see alpaca.to_60m)."""
    from icaif import alpaca

    paths = sorted(glob.glob(str(data.ROOT / "data" / "public" / "alpaca_30m_2*.parquet")))
    if not paths:
        raise FileNotFoundError("no alpaca_30m snapshot; run tools/alpaca_report.py")
    # The session filter runs again on load, not only at fetch: a snapshot saved before
    # a calendar fix (the 2016-2020 half-days) still holds that day's after-close bars.
    bars, _ = data.regular_session(pd.read_parquet(paths[-1]))
    return alpaca.to_60m(bars)


def label_exec_prices() -> pd.DataFrame:
    """Fill prices for labels: exact Alpaca :30 opens from 2016, holes filled from the rest.

    The organizer panel *is* Alpaca's SIP bars on a :00 grid (reports/alpaca_parity.json:
    0 bps on every open, close, high and low, identical volume). So Alpaca's :30 opens
    are very likely the prices the competition fills at, and they reach back to 2016.
    Yahoo's :30 opens (median 0, p99 21 bps from Alpaca) fill Alpaca's holes after Oct
    2023, then the organizer-grid guess fills what is left. Never a forward fill.
    """
    organizer, _ = data.load_organizer_bars()
    fallback = merge_fills(public_exec_opens(latest_public("60m")), organizer_exec_prices(organizer))
    return merge_fills(public_exec_opens(latest_alpaca_60m()), fallback)


def intraday_info_bars() -> pd.DataFrame:
    """What intraday decisions may look at: Alpaca's :30 grid from 2016, then Yahoo 60m.

    The live system reads Yahoo 60m bars, which sit on the same :30 grid, so a model
    trained here sees one grid throughout. The organizer's :00 grid is no longer in the
    training path, and neither is the grid switch at 2026-01-01.
    """
    a60 = latest_alpaca_60m()
    p60 = latest_public("60m")
    joined_at = a60["end"].max()
    return pd.concat([a60, p60[p60["start"] >= joined_at]], ignore_index=True)


def research_market(fills: str = "alpaca") -> sim.Market:
    """The market strategies are scored on.

    fills="alpaca" (default): execution at Alpaca's :30 opens from 2016, the
    organizer's own vendor, so every scored fill is the price the competition would
    most likely pay, over ~4x the windows Yahoo allows. fills="yahoo": Yahoo's :30
    opens from Nov 2023, the earlier setup, kept for comparison.

    Information bars are `intraday_info_bars()` either way: one :30 grid throughout.
    """
    info = intraday_info_bars()
    if fills == "alpaca":
        market = sim.market_from_public_60m(latest_alpaca_60m(), info)
    elif fills == "yahoo":
        market = sim.market_from_public_60m(latest_public("60m"), info)
    else:
        raise ValueError(f"fills must be 'alpaca' or 'yahoo', not {fills!r}")
    market.issues["fills"] = fills
    return market
