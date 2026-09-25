"""The research market: fills from public 60m bars, information from the organizer panel.

The split agreed in the design doc. Execution prices come from Yahoo's 60m bars, which
sit exactly on the live :30 grid (Nov 2023 on). What a decision may look at is the
organizer panel while it lasts (spin-off adjusted, through 2025-12-31), then Yahoo's
60m bars.

The information grid changes at the join, from bars ending on :00 to bars ending on :30.
Anything that counts bars (a "last 6 closes" signal) means half an hour later after the
join. That is fine for baselines; a model feature must be defined in clock time, not in
bars, or its meaning shifts silently on 2026-01-01.
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


def label_exec_prices() -> pd.DataFrame:
    """Fill prices for labels: organizer guesses through 2025, exact public opens after."""
    organizer, _ = data.load_organizer_bars()
    guessed = organizer_exec_prices(organizer)
    exact = sim.market_from_public_60m(latest_public("60m"), organizer).exec_prices
    joined_at = organizer["end"].max()
    out = pd.concat([guessed, exact[exact.index > joined_at]]).sort_index()
    return out[~out.index.duplicated(keep="first")]


def research_market() -> sim.Market:
    organizer, _ = data.load_organizer_bars()
    p60 = latest_public("60m")
    joined_at = organizer["end"].max()
    info = pd.concat([organizer, p60[p60["start"] >= joined_at]], ignore_index=True)
    market = sim.market_from_public_60m(p60, info)
    market.issues["info_grid_joined_at"] = str(joined_at)
    return market
