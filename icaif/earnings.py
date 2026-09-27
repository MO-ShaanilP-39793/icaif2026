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

import pandas as pd

from icaif import calendar

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
EARNINGS_ITEM = "2.02"
MARKET_OPEN = pd.Timestamp("09:30").time()


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
    # EDGAR writes Eastern wall-clock time with a misleading "Z" suffix (Apple's
    # 16:30 ET releases read 16:30:xxZ, not 20:30Z); treating it as UTC would move
    # every after-close release to before the next open.
    accepted = pd.to_datetime(f["acceptanceDateTime"].str.rstrip("Z").str.replace(".000", ""))
    return pd.DataFrame({"accepted": accepted.dt.tz_localize(calendar.TZ)})


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
            sub = client.get(SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json")).raise_for_status().json()
            blocks = [sub["filings"]["recent"]]
            for extra in sub["filings"].get("files", []):
                time.sleep(sleep)
                blocks.append(client.get(SUBMISSIONS_URL.format(name=extra["name"]))
                              .raise_for_status().json())
            events = pd.concat([parse_filings(b) for b in blocks], ignore_index=True)
            frames.append(events.assign(ticker=t))
            time.sleep(sleep)  # the SEC allows 10 requests a second
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    return out[["ticker", "accepted"]].sort_values(["ticker", "accepted"]).reset_index(drop=True), missing


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
    ok = pos < len(sessions)
    out[ok] = sessions[pos[ok]]
    return out
