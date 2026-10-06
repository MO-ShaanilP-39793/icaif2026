"""Company events from SEC EDGAR 8-K filings, point in time by acceptance timestamp.

An 8-K is filed within four business days of a material event and EDGAR stamps when
it was accepted, back decades, so unlike headlines these events can be replayed
fairly. The item codes say what kind of event it was, each named in ITEMS. Earnings
(2.02) also live in `earnings.py`, which clusters them into one release a quarter; here
every 8-K counts, because a departure or a deal filed the same week as results is a
second event, not a duplicate.

**Point in time by acceptance.** A filing is known from its EDGAR acceptance time, and
not before: `recent` and `new` read nothing accepted after the deadline. Replays read
the dated snapshot (`tools/filings_events.py`); a live round adds each name's latest
filings from EDGAR (`fetch(recent_only=True)`), so an 8-K filed this morning is in the
observation by the next round rather than after the next snapshot.

**Filing text needs real names.** A filing's own words name the company and its people,
so anonymised replays show the item codes alone ("director or officer change" names no
one). Live, and in a replay with real names (`tools/replay_sources.py` writes the texts),
`filing_text` reads what the filing says for the Event analyst; it is external text,
cleaned and capped where it is shown (`agents/untrusted.py`).

**An earnings 8-K's numbers are in its exhibit.** Item 2.02's main document says only that
a press release "is furnished as Exhibit 99.1", so read alone it gave the analyst a
boilerplate paragraph on every results day, the one day the text matters most. For 2.02
and 7.01 (Reg FD, the same pattern) `filing_text` reads the press release exhibit when the
filing has one; its first 1,200 characters carry the headline numbers. Filers label it
EX-99.1 or plain EX-99 (GE, NextEra and Pfizer in Jan 2026): matched on 99.1 alone, those
three results days fell back to the cover note.

Needs SEC_USER_AGENT, as `earnings.py` does.
"""

import html
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional

import pandas as pd

from icaif import calendar, data, earnings

ITEMS = {
    "1.01": "material agreement",
    "1.02": "agreement terminated",
    "1.03": "bankruptcy or receivership",
    "1.04": "mine safety",
    "1.05": "cybersecurity incident",
    "2.01": "acquisition or disposal completed",
    "2.02": "results of operations",
    "2.03": "debt or other obligation created",
    "2.04": "obligation accelerated",
    "2.05": "exit or restructuring costs",
    "2.06": "material impairment",
    "3.01": "delisting notice",
    "3.02": "unregistered equity sale",
    "3.03": "security holders' rights modified",
    "4.01": "auditor changed",
    "4.02": "prior financials unreliable",
    "5.01": "change in control",
    "5.02": "director or officer change",
    "5.04": "benefit plan trading suspended",
    "5.05": "code of ethics changed",
    "5.08": "shareholder nomination dates",
    "7.01": "Reg FD disclosure",
    "8.01": "other events",
}
# Filed with nearly every 8-K (exhibits) or procedural: not an event.
IGNORED = {"9.01", "5.07", "5.03"}
# An item code as EDGAR has written it since 2004. Anything else is not shown: a label
# is our text, while a code we never saw would carry EDGAR's string into the prompt.
CODE = re.compile(r"^\d\.\d\d$")


def label(code: str) -> Optional[str]:
    """The event an item code names; None for exhibits and procedure (IGNORED) and for a
    string that is not an item code at all."""
    if not CODE.match(code) or code in IGNORED:
        return None
    return ITEMS.get(code, f"item {code}")


def parse_events(block: dict, documents: bool = False) -> pd.DataFrame:
    """(accepted, items, amended) for every 8-K and 8-K/A in one EDGAR filings block.

    `documents` adds `accession` and `document` (the filing's main file), which a live
    round needs to read the filing's text; the dated snapshot leaves them out.
    """
    cols = ["form", "items", "acceptanceDateTime"]
    if documents:
        cols += ["accessionNumber", "primaryDocument"]
    f = pd.DataFrame({k: block[k] for k in cols})
    f = f[f["form"].isin(["8-K", "8-K/A"])]
    # EDGAR stamps UTC ("Z"); read as Eastern, every after-close filing lands mid-session.
    accepted = pd.to_datetime(f["acceptanceDateTime"], utc=True).dt.tz_convert(calendar.TZ)
    items = f["items"].fillna("").map(
        lambda s: ",".join(i for i in (x.strip() for x in s.split(",")) if i and i not in IGNORED))
    out = pd.DataFrame({"accepted": accepted.to_numpy(), "items": items.to_numpy(),
                        "amended": (f["form"] == "8-K/A").to_numpy()})
    if documents:
        out["accession"] = f["accessionNumber"].to_numpy()
        out["document"] = f["primaryDocument"].fillna("").to_numpy()
    return out[out["items"] != ""]


def fetch(tickers: list[str], sleep: float = 0.12, recent_only: bool = False,
          timeout: float = 60, budget_s: Optional[float] = None,
          checks: Optional[list] = None) -> tuple[pd.DataFrame, list[str]]:
    """(ticker, accepted, items, amended) for every event 8-K, and names with no CIK.

    `recent_only` reads each name's latest block alone (`earnings.submission_blocks`): a
    request a name, the last year or so, with `cik`, `accession` and `document` for the
    live round to read texts from. `budget_s` raises TimeoutError once spent: a live
    round's caller then shows the snapshot, rather than a slow EDGAR eating its deadline.
    """
    frames, missing = [], []
    t0 = time.monotonic()
    with earnings._client(timeout) as client:
        ciks = earnings.cik_map(client)
        for t in tickers:
            if budget_s is not None and time.monotonic() - t0 > budget_s:
                raise TimeoutError(f"EDGAR read past its {budget_s:.0f}s budget at {t}")
            cik = ciks.get(t.upper())
            if cik is None:
                missing.append(t)
                continue
            blocks = earnings.submission_blocks(client, cik, t, sleep, recent_only, checks)
            frames.append(pd.concat([parse_events(b, documents=recent_only) for b in blocks],
                                    ignore_index=True).assign(ticker=t, cik=cik))
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    cols = ["ticker", "accepted", "items", "amended"]
    if recent_only:
        cols += ["cik", "accession", "document"]
    return out[cols].sort_values(["ticker", "accepted"]).reset_index(drop=True), missing


def merge(snapshot: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    """The snapshot's history plus the recent read, each filing once.

    A filing both carry is the same (ticker, accepted, items, amended) row; the recent
    read's copy is kept, since it knows where the filing's text is.
    """
    key = ["ticker", "accepted", "items", "amended"]
    both = pd.concat([recent, snapshot], ignore_index=True)
    return (both.drop_duplicates(key, keep="first").sort_values(["ticker", "accepted"])
            .reset_index(drop=True))


def labels(items: str) -> list[str]:
    """The events one filing's item codes name, in our words."""
    return [x for x in (label(i) for i in str(items).split(",")) if x]


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
        rows = [{"hours_ago": round((deadline - r.accepted).total_seconds() / 3600, 1),
                 "events": labels(r.items), "amendment": bool(r.amended)} for r in g.itertuples()]
        rows = [r for r in rows if r["events"]]
        if rows:
            out[t] = rows
    return out


def new(events: pd.DataFrame, after: pd.Timestamp, deadline: pd.Timestamp) -> pd.DataFrame:
    """Filings accepted in (after, deadline]: the 8-Ks a round sees for the first time."""
    e = events[(events["accepted"] > pd.Timestamp(after)) & (events["accepted"] <= pd.Timestamp(deadline))]
    # `.loc` with a bool Series: an empty list in `[]` selects no columns, not no rows.
    e = e.loc[e["items"].map(lambda i: bool(labels(i))).astype(bool)]
    return e.sort_values(["ticker", "accepted"])


# ----------------------------------------------------------------------------- filing text

ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
# Where the cover page ends and the items begin, in the order 8-Ks are written.
_ITEM = re.compile(r"\bItem\s+\d\.\d\d\b", re.IGNORECASE)
_SKIP = {"script", "style", "head", "title"}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self._skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip += 1
        elif tag in ("p", "div", "br", "tr", "li"):
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in _SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, d):
        if not self._skip:
            self.parts.append(d)


def html_text(doc: str) -> str:
    """The visible text of an 8-K's HTML, from its first item on.

    The cover page (the form's title, check boxes, addresses, the exchange table) says
    nothing about the event and would use the whole cap; the first "Item x.xx" is where
    the filing starts saying what happened. A document with no item heading is kept whole.
    """
    p = _Text()
    p.feed(doc)
    text = " ".join(html.unescape("".join(p.parts)).split())
    m = _ITEM.search(text)
    return text[m.start():] if m else text


def document_text(client, cik: int, accession: str, document: str, sleep: float = 0.12) -> str:
    """The text of one filing's main document, as `html_text` reads it (uncapped)."""
    url = ARCHIVES_URL.format(cik=int(cik), acc=str(accession).replace("-", ""), doc=document)
    body = client.get(url).raise_for_status().text
    time.sleep(sleep)
    return html_text(body) if "<" in body[:2000] else " ".join(body.split())


INDEX_URL = earnings.INDEX_URL
# Items whose main document points at a press release in Exhibit 99.1 instead of saying
# what happened.
EXHIBIT_ITEMS = {"2.02", "7.01"}
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.IGNORECASE)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.IGNORECASE)
_HREF = re.compile(r'href="([^"]+)"', re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")
# EDGAR's SGML wrapper, read as text: "EX-99.1 2 msft-ex99_1.htm EX-99.1 ".
_WRAPPER = re.compile(r"^EX-99(?:\.\d+)?\s+\d+\s+\S+\s+EX-99(?:\.\d+)?\s+", re.IGNORECASE)
_EX99 = re.compile(r"^EX-99(?:\.(\d+))?$")


def press_release(found: dict[str, str]) -> Optional[str]:
    """The press release among a filing's exhibits: EX-99.1, else EX-99, else the
    lowest-numbered EX-99.x."""
    if "EX-99.1" in found:
        return found["EX-99.1"]
    if "EX-99" in found:
        return found["EX-99"]
    numbered = sorted((int(m.group(1)), t) for t in found if (m := _EX99.match(t)) and m.group(1))
    return found[numbered[0][1]] if numbered else None


def exhibits(index_html: str) -> dict[str, str]:
    """{document type: file name} from a filing's index page (EX-99.1 -> its .htm)."""
    out = {}
    for row in _ROW.findall(index_html):
        cells = [html.unescape(_TAGS.sub("", c)).strip() for c in _CELL.findall(row)]
        hrefs = _HREF.findall(row)
        if len(cells) >= 4 and hrefs and cells[3]:
            out.setdefault(cells[3].upper(), hrefs[0].rsplit("/", 1)[-1])
    return out


def filing_text(client, cik: int, accession: str, document: str, items: str,
                sleep: float = 0.12) -> str:
    """What the filing says: the press release for a results or Reg FD 8-K that has one, the
    main document otherwise (and if the index cannot be read, since the codes still say
    what kind of event it was)."""
    codes = {i.strip() for i in str(items).split(",")}
    if codes & EXHIBIT_ITEMS:
        acc = str(accession).replace("-", "")
        url = INDEX_URL.format(cik=int(cik), acc=acc, accession=accession)
        try:
            index = client.get(url).raise_for_status().text
            time.sleep(sleep)
            doc = press_release(exhibits(index))
        except Exception:  # noqa: BLE001 - the main document still says something
            doc = None
        if doc:
            return _WRAPPER.sub("", document_text(client, cik, accession, doc, sleep))
    return document_text(client, cik, accession, document, sleep)


# Replay texts (`tools/replay_sources.py`): one file per fetched span, named by it. Not
# `edgar_8k_*`: the snapshot loaders take the last file of that pattern by name, and
# "edgar_8k_t..." sorts after every dated snapshot, which would swap the 8-K history for
# a few weeks of texts without an error.
TEXTS_DIR = data.ROOT / "data" / "external" / "edgar_texts"


def texts_path(lo: pd.Timestamp, hi: pd.Timestamp, directory: Path = TEXTS_DIR) -> Path:
    return Path(directory) / f"texts_{pd.Timestamp(lo):%Y-%m-%dT%H%M}_{pd.Timestamp(hi):%Y-%m-%dT%H%M}.parquet"


def load_texts(lo: pd.Timestamp, hi: pd.Timestamp, directory: Optional[Path] = None) -> pd.DataFrame:
    """The 8-Ks with text of one fetched span covering [lo, hi]; raises if none does.

    A span never fetched would replay as filings without words, which reads as the
    analyst ignoring the text rather than never being shown it.
    """
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    directory = Path(directory or TEXTS_DIR)
    for f in sorted(directory.glob("texts_*.parquet")):
        a, b = (pd.Timestamp(x).tz_localize(calendar.TZ) for x in f.stem.removeprefix("texts_").split("_"))
        if a <= lo and b >= hi:
            return pd.read_parquet(f)
    raise FileNotFoundError(f"no 8-K texts fetched for {lo:%Y-%m-%d %H:%M} to {hi:%Y-%m-%d %H:%M} "
                            f"in {directory}; run tools/replay_sources.py")
