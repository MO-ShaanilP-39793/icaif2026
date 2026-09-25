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

from icaif import data, sim


def latest_public(interval: str) -> pd.DataFrame:
    paths = sorted(glob.glob(str(data.ROOT / "data" / "public" / f"yahoo_{interval}_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no cached yahoo_{interval} snapshot; run tools/data_report.py")
    return pd.read_parquet(paths[-1])


def research_market() -> sim.Market:
    organizer, _ = data.load_organizer_bars()
    p60 = latest_public("60m")
    joined_at = organizer["end"].max()
    info = pd.concat([organizer, p60[p60["start"] >= joined_at]], ignore_index=True)
    market = sim.market_from_public_60m(p60, info)
    market.issues["info_grid_joined_at"] = str(joined_at)
    return market
