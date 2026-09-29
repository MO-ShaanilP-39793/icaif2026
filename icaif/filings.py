"""Company events from SEC EDGAR 8-K filings, point in time by acceptance timestamp.

An 8-K is filed within four business days of a material event and EDGAR stamps when
it was accepted, back decades, so unlike headlines these events can be replayed
fairly. The item codes say what kind of event it was; the ones an agent can act on
are named in ITEMS. Earnings (2.02) stay in `earnings.py`, which clusters them into
one release a quarter; here every 8-K counts, because a departure or a deal filed the
same week as results is a second event, not a duplicate.

Needs SEC_USER_AGENT, as `earnings.py` does.
"""

import pandas as pd

from icaif import calendar, earnings

ITEMS = {
    "1.01": "material agreement",
    "1.02": "agreement terminated",
    "1.03": "bankruptcy or receivership",
    "1.05": "cybersecurity incident",
    "2.01": "acquisition or disposal completed",
    "2.02": "results of operations",
    "2.05": "exit or restructuring costs",
    "2.06": "material impairment",
    "3.01": "delisting notice",
    "4.02": "prior financials unreliable",
    "5.02": "director or officer change",
    "7.01": "Reg FD disclosure",
    "8.01": "other events",
}
# Filed with nearly every 8-K (exhibits) or procedural: not an event.
IGNORED = {"9.01", "5.07", "5.03"}


def parse_events(block: dict) -> pd.DataFrame:
    """(accepted, items) for every 8-K and 8-K/A in one EDGAR filings block."""
    f = pd.DataFrame({k: block[k] for k in ("form", "items", "acceptanceDateTime")})
    f = f[f["form"].isin(["8-K", "8-K/A"])]
    # EDGAR stamps UTC ("Z"); read as Eastern, every after-close filing lands mid-session.
    accepted = pd.to_datetime(f["acceptanceDateTime"], utc=True).dt.tz_convert(calendar.TZ)
    items = f["items"].fillna("").map(
        lambda s: ",".join(i for i in (x.strip() for x in s.split(",")) if i and i not in IGNORED))
    out = pd.DataFrame({"accepted": accepted.to_numpy(), "items": items.to_numpy(),
                        "amended": (f["form"] == "8-K/A").to_numpy()})
    return out[out["items"] != ""]


def fetch(tickers: list[str], sleep: float = 0.12) -> tuple[pd.DataFrame, list[str]]:
    """(ticker, accepted, items, amended) for every event 8-K, and names with no CIK."""
    frames, missing = [], []
    with earnings._client() as client:
        ciks = earnings.cik_map(client)
        for t in tickers:
            cik = ciks.get(t.upper())
            if cik is None:
                missing.append(t)
                continue
            blocks = earnings.submission_blocks(client, cik, t, sleep)
            frames.append(pd.concat([parse_events(b) for b in blocks], ignore_index=True)
                          .assign(ticker=t))
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    cols = ["ticker", "accepted", "items", "amended"]
    return out[cols].sort_values(["ticker", "accepted"]).reset_index(drop=True), missing


def recent(events: pd.DataFrame, deadline: pd.Timestamp, days: int = 7) -> dict[str, list[dict]]:
    """{ticker: events accepted in the `days` before the deadline, newest first}.

    Only filings accepted by the deadline: one accepted at 09:15 is public by 09:30
    but was not known at the 09:10 cut, and the next round will see it.
    """
    deadline = pd.Timestamp(deadline)
    e = events[(events["accepted"] <= deadline)
               & (events["accepted"] > deadline - pd.Timedelta(days=days))]
    out = {}
    for t, g in e.sort_values("accepted", ascending=False).groupby("ticker"):
        out[t] = [{"hours_ago": round((deadline - r.accepted).total_seconds() / 3600, 1),
                   "events": [ITEMS.get(i, f"item {i}") for i in r.items.split(",")],
                   "amendment": bool(r.amended)} for r in g.itertuples()]
    return out
