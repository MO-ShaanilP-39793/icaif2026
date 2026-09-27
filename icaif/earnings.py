"""Earnings release times from SEC EDGAR: 8-K filings that carry item 2.02.

Item 2.02 ("Results of Operations and Financial Condition") is how US issuers file
their earnings release, so its acceptance time is a point-in-time record of when the
numbers became public. That matters: an earnings calendar scraped today reports the
dates, not when they were knowable, and not whether a release came before the open
or after the close. Coverage starts in 2004, when item 2.02 was introduced.

The SEC requires a User-Agent with contact details on every request, set in the
environment as SEC_USER_AGENT ("name email"). There is no default: a made-up contact
is against the SEC's policy, and an anonymous client gets blocked, which would show
up as an empty result rather than an error.
"""

import os
import time

import numpy as np
import pandas as pd

from icaif import calendar

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
EARNINGS_ITEM = "2.02"
MARKET_OPEN = pd.Timestamp("09:30").time()
# A company that re-registers gets a new CIK, and the current ticker map points only at
# it: history under the old one silently disappears. Found by comparing each name's
# release count with one a quarter since 2004 (tools/enrich_data.py prints the gaps).
FORMER_CIKS = {
    "XOM": (34088,),     # Exxon Mobil Corp, before the 2026 holding company
    "DIS": (1001039,),   # The Walt Disney Co (now TWDC Enterprises 18), before 2019
    "GOOGL": (1288776,), # Google Inc., before Alphabet in 2015
    "GOOG": (1288776,),
}
# Broad-universe names with similar gaps (BLK, LIN, AVGO, MDT, ...) are listed by
# tools/enrich_data.py; their earnings features are NaN before their first recorded
# release rather than a stale "sessions since" count.
CLUSTER_DAYS = 30


def _client():
    import httpx

    agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not agent:
        raise RuntimeError("set SEC_USER_AGENT='<name> <email>' (the SEC requires a contact)")
    return httpx.Client(headers={"User-Agent": agent}, timeout=60, follow_redirects=True)


def cik_map(client) -> dict[str, int]:
    """Current ticker -> CIK. Class shares use a dash here too (BRK-B)."""
    rows = client.get(TICKERS_URL).raise_for_status().json().values()
    return {r["ticker"].upper(): int(r["cik_str"]) for r in rows}


def parse_filings(block: dict) -> pd.DataFrame:
    """Earnings 8-Ks from one EDGAR filings block (the column-array layout it uses)."""
    f = pd.DataFrame({k: block[k] for k in ("form", "items", "acceptanceDateTime")})
    f = f[(f["form"] == "8-K") & f["items"].fillna("").str.split(",").map(
        lambda items: EARNINGS_ITEM in [i.strip() for i in items])]
    # The "Z" is real: EDGAR stamps UTC. Apple's 16:30 ET releases read 20:30Z in
    # summer and 21:30Z in winter. Read as Eastern wall-clock, every after-close
    # release lands mid-session and maps to the wrong open. (This was first written
    # the other way round; the timing check in tools/enrich_data.py caught it.)
    accepted = pd.to_datetime(f["acceptanceDateTime"], utc=True)
    return pd.DataFrame({"accepted": accepted.dt.tz_convert(calendar.TZ)})


def fetch(tickers: list[str], sleep: float = 0.12) -> tuple[pd.DataFrame, list[str]]:
    """(ticker, accepted) for every earnings 8-K, and the tickers EDGAR has no CIK for.

    Delisted names are missing from the current ticker map; they have no Yahoo prices
    either, so they are outside the universe anyway.
    """
    frames, missing = [], []
    with _client() as client:
        ciks = cik_map(client)
        for t in tickers:
            cik = ciks.get(t.upper())
            if cik is None:
                missing.append(t)
                continue
            blocks = []
            for c in (cik, *FORMER_CIKS.get(t.upper(), ())):
                sub = client.get(SUBMISSIONS_URL.format(name=f"CIK{c:010d}.json")).raise_for_status().json()
                blocks.append(sub["filings"]["recent"])
                for extra in sub["filings"].get("files", []):
                    time.sleep(sleep)
                    blocks.append(client.get(SUBMISSIONS_URL.format(name=extra["name"]))
                                  .raise_for_status().json())
                time.sleep(sleep)
            events = pd.concat([parse_filings(b) for b in blocks], ignore_index=True)
            frames.append(events.assign(ticker=t))
            time.sleep(sleep)  # the SEC allows 10 requests a second
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    return out[["ticker", "accepted"]].sort_values(["ticker", "accepted"]).reset_index(drop=True), missing


def quarterly(events: pd.DataFrame, cluster_days: int = CLUSTER_DAYS) -> pd.DataFrame:
    """One earnings event per cluster of item-2.02 filings, keeping the last.

    Item 2.02 also carries pre-announcements and interim updates (Tesla's delivery
    numbers, Chevron's interim updates), a few weeks before the earnings release
    itself. Kept, "sessions to next earnings" would point at the preview. Filings
    less than `cluster_days` apart form one cluster, and the last is the release.
    Duplicates filed under two CIKs on the same day collapse the same way.
    """
    e = events.sort_values(["ticker", "accepted"])
    gap = e.groupby("ticker")["accepted"].diff()
    cluster = (gap.isna() | (gap >= pd.Timedelta(days=cluster_days))).cumsum()
    return e.groupby(cluster).tail(1).reset_index(drop=True)


def reaction_session(accepted: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """The first session whose open reflects the release.

    Before 09:30 it is that day's session; at or after 09:30 (almost always after the
    16:00 close) it is the next one. Weekends and holidays roll forward to the next
    session.
    """
    day = accepted.dt.tz_localize(None).dt.normalize()
    after_open = accepted.dt.time.map(lambda t: t >= MARKET_OPEN).to_numpy()
    pos = sessions.searchsorted(day, side="left")
    # Step past the day's own session only if there was one: a Saturday release is
    # reflected at Monday's open, not Tuesday's.
    is_session = (pos < len(sessions)) & (sessions[pos.clip(max=len(sessions) - 1)] == day)
    pos = pos + (after_open & is_session).astype(int)
    out = pd.Series(pd.NaT, index=accepted.index, dtype="datetime64[ns]")
    # A release before the calendar starts has no session here. Without the second test
    # it would map to the first session and read as a release on that day.
    ok = (pos < len(sessions)) & (day >= sessions[0]).to_numpy()
    out[ok] = sessions[pos[ok]]
    return out


NEXT_KNOWN_SESSIONS = 10


def proximity(events: pd.DataFrame, decisions: pd.DataFrame,
              sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Sessions since the last release and to the next, per (decision row, ticker).

    `events` is (ticker, accepted), already one per quarter; `decisions` is
    (day, deadline), any number per day. Returns columns row (the decision's
    position), ticker, since, to_next.

    A release counts as past only if it was accepted before the decision's deadline.
    One accepted mid-session reacts at the next open, but it is already known to the
    decisions after it that day, so `since` is 0 there rather than negative.

    "Sessions to next" reads the actual future release, which is only fair while its
    date would have been announced: companies publish the date roughly two to four
    weeks ahead. Beyond NEXT_KNOWN_SESSIONS it is NaN, not a count a live system
    could not have known. Both are NaN before a ticker's first recorded release, so a
    name whose EDGAR history starts late (a re-registration) reads as unknown rather
    than as years since its last results.
    """
    pos = pd.Series(range(len(sessions)), index=sessions)
    day_pos = pos.reindex(pd.DatetimeIndex(decisions["day"])).to_numpy()
    deadlines = decisions["deadline"].to_numpy()
    frames = []
    for ticker, e in events.sort_values("accepted").groupby("ticker"):
        acc = e["accepted"].to_numpy()
        react = reaction_session(e["accepted"], sessions)
        r_pos = pos.reindex(pd.DatetimeIndex(react)).to_numpy()
        n_past = acc.searchsorted(deadlines, side="left")  # releases before each deadline
        since = np.full(len(decisions), np.nan)
        has_past = n_past > 0
        since[has_past] = np.maximum(day_pos[has_past] - r_pos[n_past[has_past] - 1], 0)
        to_next = np.full(len(decisions), np.nan)
        has_next = has_past & (n_past < len(acc))
        to_next[has_next] = r_pos[n_past[has_next]] - day_pos[has_next]
        to_next[to_next > NEXT_KNOWN_SESSIONS] = np.nan
        frames.append(pd.DataFrame({"row": np.arange(len(decisions)), "ticker": ticker,
                                    "since": since, "to_next": to_next}))
    return pd.concat(frames, ignore_index=True)
