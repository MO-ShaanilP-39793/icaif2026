"""The daily model's training universe: large caps as they stood on each date.

Membership comes from the public S&P 500 history (github.com/fja05680/sp500, MIT),
snapshotted in `data/external/`. Its symbols are the ones in use on each date, so a
ticker that later changed or delisted appears under its old name.

On each date the universe is the top N members by trailing dollar volume, plus the 30
competition names whatever their rank. Two biases are built into what's left, and
both are measured rather than hidden:

- **Survivorship.** Yahoo keeps history only for symbols still trading, so a member
  that delisted has no prices and can't be ranked. The universe on an old date is then
  the survivors among that date's large caps, which flatters momentum.
  `coverage()` reports the share of members priced, by year.
- **Symbol reuse.** A later company can take over an old ticker, and Yahoo returns the
  later company's history under it. Among the top 100 from 2001 on this should be rare.
  Nothing detects it here.

Dollar volume is close x volume, which is invariant to split adjustment. It is averaged
over sessions strictly before the date. With the date itself included, a name that
spikes on day d would join the universe on the day of its spike.
"""

from pathlib import Path

import pandas as pd

from icaif.data import ROOT, load_universe

EXTERNAL = ROOT / "data" / "external"
TOP_N = 100
LOOKBACK_SESSIONS = 63


def yahoo_symbol(ticker: str) -> str:
    """Membership files write class shares with a dot (BRK.B); Yahoo uses a dash."""
    return ticker.replace(".", "-")


def latest(pattern: str) -> Path:
    paths = sorted(EXTERNAL.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no snapshot matching {pattern} in {EXTERNAL}; "
                                "run tools/enrich_data.py")
    return paths[-1]


def load_membership(path: Path | None = None) -> pd.DataFrame:
    """One row per membership spell: ticker, symbol (Yahoo's), start, end (NaT = current)."""
    m = pd.read_csv(path or latest("sp500_ticker_start_end_*.csv"))
    return pd.DataFrame({
        "ticker": m["ticker"],
        "symbol": m["ticker"].map(yahoo_symbol),
        "start": pd.to_datetime(m["start_date"]),
        "end": pd.to_datetime(m["end_date"]),
    })


def member_mask(membership: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """date x symbol, True while the symbol was in the index (start inclusive, end exclusive).

    The end date is the day the name left, so it is no longer a member that day.
    """
    symbols = sorted(membership["symbol"].unique())
    out = pd.DataFrame(False, index=dates, columns=symbols)
    for row in membership.itertuples():
        live = (dates >= row.start) & ((dates < row.end) if pd.notna(row.end) else True)
        out.loc[live, row.symbol] = True
    return out


def dollar_volume(daily: pd.DataFrame) -> pd.DataFrame:
    """date x symbol trailing mean of close x volume over the prior LOOKBACK sessions."""
    dv = (daily.assign(dv=daily["close"] * daily["volume"])
          .pivot_table(index="date", columns="ticker", values="dv", aggfunc="last")
          .sort_index())
    return dv.rolling(LOOKBACK_SESSIONS, min_periods=LOOKBACK_SESSIONS // 2).mean().shift(1)


def build(daily: pd.DataFrame, membership: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """date x symbol, True if the symbol is in the training universe that day.

    The competition names are always in, member or not. The daily model upweights and
    flags them, and a flag on a name that dropped out of the universe would mark nothing.
    """
    dv = dollar_volume(daily)
    members = member_mask(membership, dv.index).reindex(columns=dv.columns, fill_value=False)
    ranked = dv.where(members).rank(axis=1, ascending=False, method="first")
    universe = ranked.le(top_n)
    for t in load_universe():
        if t in universe.columns:
            universe[t] = dv[t].notna()
    return universe


def coverage(daily: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    """Per year: average members per day, how many have Yahoo prices, and the share."""
    dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
    members = member_mask(membership, dates)
    priced = (daily.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
              .reindex(index=dates, columns=members.columns).notna())
    by_year = pd.DataFrame({"members": members.sum(axis=1),
                            "priced": (members & priced).sum(axis=1)}).groupby(dates.year).mean()
    by_year["share_priced"] = by_year["priced"] / by_year["members"]
    return by_year.round(3)
