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


# Same company, new ticker. Yahoo moves the whole history to the new symbol and returns
# nothing under the old one, so a spell under the old ticker was a member with no prices:
# the company dropped out of the universe for the length of that spell (Fiserv as FI from
# 2023-06 to 2025-11), and the next name by dollar volume took its place. Mapped here,
# both spells price as one name. Checked against the membership file, which records each
# as one spell ending the day the next starts (test_universe). A takeover is not a rename:
# the target's prices are gone, and the acquirer is its own spell.
RENAMES = {
    "BK": "BNY",     # BNY Mellon, 2026-05-21
    "FI": "FISV",    # Fiserv, back to its old ticker on its Nasdaq move, 2025-11-11
    "MMC": "MRSH",   # Marsh McLennan, 2026-01-14
    "SATS": "ECHO",  # EchoStar, 2026-06-24
}

# Index changes announced after the snapshot's source last updated (fja05680/sp500 on
# 2026-09-07). Without them a name that joined is never priced, and BE, trading about
# $3.7bn a day against a top-100 cut near $0.9bn, is missing from the live universe while
# every other name's rank shifts one place. Each row applies only while the snapshot lacks
# it, so a refreshed snapshot that has caught up takes over unchanged. (ticker, date,
# "add" | "drop"): the S&P September 2026 rebalance, announced 2026-09-04.
LATE_CHANGES = [
    ("BE", "2026-09-21", "add"), ("ILMN", "2026-09-21", "add"), ("P", "2026-09-21", "add"),
    ("TAP", "2026-09-21", "drop"), ("TTD", "2026-09-21", "drop"), ("BLDR", "2026-09-21", "drop"),
]


def yahoo_symbol(ticker: str) -> str:
    """Yahoo's symbol for a membership ticker: a rename's current one, and class shares
    with a dash (BRK.B -> BRK-B)."""
    return RENAMES.get(ticker, ticker).replace(".", "-")


def latest(pattern: str) -> Path:
    paths = sorted(EXTERNAL.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no snapshot matching {pattern} in {EXTERNAL}; "
                                "run tools/enrich_data.py")
    return paths[-1]


def load_membership(path: Path | None = None) -> pd.DataFrame:
    """One row per membership spell: ticker, symbol (Yahoo's), start, end (NaT = current)."""
    m = pd.read_csv(path or latest("sp500_ticker_start_end_*.csv"))
    spells = pd.DataFrame({"ticker": m["ticker"], "start": pd.to_datetime(m["start_date"]),
                           "end": pd.to_datetime(m["end_date"])})
    spells = with_late_changes(spells)
    spells.insert(1, "symbol", spells["ticker"].map(yahoo_symbol))
    return spells


def with_late_changes(spells: pd.DataFrame, changes=None) -> pd.DataFrame:
    """`spells` (ticker, start, end) with each of `changes` it does not already record."""
    spells = spells.copy()
    for ticker, day, kind in LATE_CHANGES if changes is None else changes:
        day = pd.Timestamp(day)
        mine = spells["ticker"] == ticker
        if kind == "add":
            if not (mine & (spells["start"] == day)).any():
                spells = pd.concat([spells, pd.DataFrame({
                    "ticker": [ticker], "start": [day],
                    "end": pd.Series([pd.NaT], dtype=spells["end"].dtype)})], ignore_index=True)
        elif kind == "drop":
            open_ = mine & spells["end"].isna() & (spells["start"] < day)
            spells.loc[open_, "end"] = day
        else:
            raise ValueError(f"unknown change {kind!r} for {ticker}")
    return spells.reset_index(drop=True)


def as_of(spells: pd.DataFrame) -> pd.Timestamp:
    """The latest index change the membership records."""
    return max(spells["start"].max(), spells["end"].max())


def last_rebalance(day) -> pd.Timestamp:
    """The latest S&P quarterly rebalance effective on or before `day`: the Monday after
    the third Friday of March, June, September and December."""
    day = pd.Timestamp(day).normalize()
    best = None
    for year in (day.year - 1, day.year):
        for month in (3, 6, 9, 12):
            first = pd.Timestamp(year=year, month=month, day=1)
            third_friday = first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)
            effective = third_friday + pd.Timedelta(days=3)
            if effective <= day:
                best = effective
    return best


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
