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
import re
import time
from typing import Optional

import numpy as np
import pandas as pd

from icaif import calendar

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{accession}-index.htm"
_ACCEPTED = re.compile(r'Accepted</div>\s*<div class="info">([^<]+)<')
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


def _client(timeout: float = 60):
    import httpx

    agent = os.environ.get("SEC_USER_AGENT", "").strip()
    if not agent:
        raise RuntimeError("set SEC_USER_AGENT='<name> <email>' (the SEC requires a contact)")
    return httpx.Client(headers={"User-Agent": agent}, timeout=timeout, follow_redirects=True)


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


class EdgarTimeError(RuntimeError):
    """A block's acceptance times disagree with its filing pages in a way not corrected."""


def index_accepted(client, cik: int, accession: str) -> pd.Timestamp:
    """The acceptance time a filing's own index page states, in Eastern time."""
    url = INDEX_URL.format(cik=int(cik), acc=accession.replace("-", ""), accession=accession)
    m = _ACCEPTED.search(client.get(url).raise_for_status().text)
    if not m:
        raise EdgarTimeError(f"no acceptance time on {url}")
    return pd.Timestamp(m.group(1).strip()).tz_localize(calendar.TZ, ambiguous=True)


def _eastern_hours(ts: pd.Timestamp) -> float:
    """Hours Eastern time is behind UTC at `ts`: 4 in summer, 5 in winter."""
    return -ts.tz_convert(calendar.TZ).utcoffset().total_seconds() / 3600


EIGHT_K = ("8-K", "8-K/A")


def _utc(t: str) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _back_by_offset(t: str) -> str:
    """A time EDGAR served late by the Eastern offset, moved back by the offset in force."""
    sent = _utc(t)
    guess = sent - pd.Timedelta(hours=_eastern_hours(sent))
    return (sent - pd.Timedelta(hours=_eastern_hours(guess))).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def checked_times(client, cik: int, block: dict, sleep: float = 0.12,
                  checks: Optional[list] = None) -> dict:
    """The block, its 8-Ks' acceptance times made to agree with EDGAR's own filing pages.

    Since about 2026-10-06 the submissions JSON serves filers' older filings late by
    exactly Eastern's UTC offset (JPM's 06:30:38 ET results as 14:30:38Z, i.e. 10:30
    ET), while each filing's index page still states the true time. On Oct 6 it was
    every filing of AAPL, AMZN, BAC, CVX, GS, JPM, META, NEE, NKE and UNH since 2025;
    by that evening filings accepted that day came back right and the older ones did not,
    so JPM's and BAC's blocks were right at the top and late below. Read as sent, a
    pre-market release lands after the open, on the next session's reaction, and the live
    merge with the snapshot holds every such filing twice, the late copy firing a second
    "new 8-K".

    Only the 8-K entries are checked and corrected, the only ones either reader keeps;
    every other entry is returned as sent. Each read is checked anew, never from a cache:
    a correction kept after EDGAR mends its feed would move filings hours before they
    existed. The newest and oldest 8-K's pages are read every time (a block checked at
    the top only passed JPM's late 8-Ks as right). Both agree: as sent. Both late by the
    offset: every 8-K moved back by the offset in force at it. Newest right and oldest
    late: the boundary is found by bisection, each step a page read, and only the 8-Ks
    below it are moved. Anything else raises: a gap of another size, or a newest late
    over an oldest right, is a fault nobody has measured, and the snapshot (live) or a
    stop (tools) is better than a guess.
    """
    times, accs = block.get("acceptanceDateTime", []), block.get("accessionNumber", [])
    forms = block.get("form", [])
    idx = [i for i in range(len(times))
           if times[i] and i < len(accs) and accs[i] and i < len(forms) and forms[i] in EIGHT_K]
    if not idx:
        return block
    probes = {}

    def kind(j):
        i = idx[j]
        if j not in probes:
            true = index_accepted(client, cik, accs[i]).tz_convert("UTC")
            time.sleep(sleep)
            gap = (_utc(times[i]) - true).total_seconds() / 3600
            if gap == 0:
                probes[j] = "as_sent"
            elif gap == _eastern_hours(true):
                probes[j] = "late_by_offset"
            else:
                raise EdgarTimeError(f"CIK {cik} {accs[i]}: EDGAR JSON {times[i]} is {gap:+.2f}h "
                                     f"from its filing page ({true})")
        return probes[j]

    newest, oldest = kind(0), kind(len(idx) - 1)
    if newest == "late_by_offset" and oldest == "as_sent":
        raise EdgarTimeError(f"CIK {cik}: the newest 8-K is late and the oldest is right; "
                             "not a pattern measured, not corrected")
    if oldest == "as_sent":
        first_late = len(idx)
    elif newest == "late_by_offset":
        first_late = 0
    else:   # right at the top, late below: bisect for the first late 8-K
        lo, hi = 0, len(idx) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if kind(mid) == "late_by_offset":
                hi = mid
            else:
                lo = mid
        first_late = hi
    if checks is not None:
        checks.append({"cik": int(cik), "eight_ks": len(idx), "late": len(idx) - first_late,
                       "probes": len(probes),
                       "times": "as_sent" if first_late == len(idx) else "late_by_offset"})
    if first_late == len(idx):
        return block
    fixed = list(times)
    for i in idx[first_late:]:
        fixed[i] = _back_by_offset(times[i])
    return {**block, "acceptanceDateTime": fixed}


def submission_blocks(client, cik: int, ticker: str, sleep: float = 0.12,
                      recent_only: bool = False, checks: Optional[list] = None,
                      window: Optional[tuple] = None) -> list[dict]:
    """Every filings block EDGAR holds for a name, former CIKs included, each with its
    acceptance times checked against EDGAR's filing pages (`checked_times`) and tagged
    with the CIK it came from (`_cik`): a filing's text lives under that CIK's path.

    `recent_only` reads the current CIK's latest block alone (its last 1,000 filings
    or at least a year): one request a name instead of one per page of history. That
    is all a live round needs on top of a snapshot, and the full history of ~100
    names takes about five minutes, longer than a round's scorer is given.

    `window` (first, last) adds, to the recent block, every block a replay of that span
    needs: the older history pages whose filing dates overlap it, and the former CIKs'.
    The recent block of a heavy filer reaches back only a year or less (JPM's, filing
    ~26,000 prospectuses a year, starts 2025-10-06), so the Apr 2025 window's bank 8-Ks,
    JPM's results among them, replayed as item codes with no text, and XOM's filings
    under its pre-2026 CIK in every window.
    """
    blocks = []
    lo = hi = None
    if window is not None:
        # Filing dates, not acceptance times: an 8-K accepted after 17:30 ET is dated the
        # next business day, so the overlap is widened by a few days on each side.
        lo = (pd.Timestamp(window[0]) - pd.Timedelta(days=4)).strftime("%Y-%m-%d")
        hi = (pd.Timestamp(window[1]) + pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    former = FORMER_CIKS.get(ticker.upper(), ()) if (window is not None or not recent_only) else ()
    for c in (cik, *former):
        sub = client.get(SUBMISSIONS_URL.format(name=f"CIK{c:010d}.json")).raise_for_status().json()
        blocks.append({**checked_times(client, c, sub["filings"]["recent"], sleep, checks), "_cik": c})
        pages = sub["filings"].get("files", [])
        if window is not None:
            pages = [f for f in pages if f.get("filingFrom", "") <= hi and f.get("filingTo", "") >= lo]
        elif recent_only:
            pages = []
        for extra in pages:
            time.sleep(sleep)
            page = client.get(SUBMISSIONS_URL.format(name=extra["name"])).raise_for_status().json()
            blocks.append({**checked_times(client, c, page, sleep, checks), "_cik": c})
        time.sleep(sleep)
    return blocks


def fetch(tickers: list[str], sleep: float = 0.12, recent_only: bool = False,
          checks: Optional[list] = None) -> tuple[pd.DataFrame, list[str]]:
    """(ticker, accepted) for every earnings 8-K, and the tickers EDGAR has no CIK for.

    Delisted names are missing from the current ticker map; they have no Yahoo prices
    either, so they are outside the universe anyway. `recent_only`: see
    `submission_blocks`; the result is then only the last year or so.
    """
    frames, missing = [], []
    with _client() as client:
        ciks = cik_map(client)
        for t in tickers:
            cik = ciks.get(t.upper())
            if cik is None:
                missing.append(t)
                continue
            blocks = submission_blocks(client, cik, t, sleep, recent_only, checks)
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
