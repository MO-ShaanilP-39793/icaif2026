import pandas as pd
import pytest

from icaif import earnings, filings, live
from tests.test_news_events import TZ, _snapshot

CIK = 19617
# JPM's results, summer and winter, as their filing pages state them (Eastern).
TRUE = {"0001628280-26-048078": "2026-07-14 06:30:38", "0001628280-26-002211": "2026-01-13 06:45:10"}


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return self

    def json(self):
        return self.payload

    @property
    def text(self):
        return self.payload


class _Edgar:
    """Serves one submissions block, and an index page per filing stating `pages`."""

    def __init__(self, sent: list[str], pages: dict):
        self.block = {"form": ["8-K", "8-K"], "items": ["2.02,9.01", "2.02,9.01"],
                      "acceptanceDateTime": sent, "accessionNumber": list(TRUE),
                      "primaryDocument": ["a.htm", "b.htm"]}
        self.pages, self.asked = pages, []

    def get(self, url):
        self.asked.append(url)
        if "submissions" in url:
            return _Resp({"filings": {"recent": self.block, "files": []}})
        acc = url.rsplit("/", 1)[1].removesuffix("-index.htm")
        return _Resp(f'<div class="infoHead">Accepted</div>\n<div class="info">{self.pages[acc]}</div>')


def _read(client):
    blocks = earnings.submission_blocks(client, CIK, "JPM", sleep=0, recent_only=True)
    return list(earnings.parse_filings(blocks[0])["accepted"].dt.strftime("%Y-%m-%d %H:%M:%S"))


def test_a_filing_time_edgar_sends_late_by_the_eastern_offset_is_read_as_its_page_states():
    """Since Oct 6, 2026 EDGAR's JSON serves JPM's 06:30 ET results as 10:30 ET (4 hours
    late in summer, 5 in winter). Read as sent, a pre-market release lands after the
    open, on the next session's reaction, and live holds it twice beside the snapshot."""
    late = ["2026-07-14T14:30:38.000Z", "2026-01-13T16:45:10.000Z"]
    got = _read(_Edgar(late, TRUE))
    assert got == ["2026-07-14 06:30:38", "2026-01-13 06:45:10"]
    edgar = _Edgar(late, TRUE)
    events = filings.parse_events(earnings.checked_times(edgar, CIK, edgar.block, sleep=0), documents=True)
    assert events["accepted"].dt.strftime("%H:%M").tolist() == ["06:30", "06:45"]


def test_a_block_edgar_sends_right_is_never_moved():
    """The correction must stop the moment EDGAR mends its feed. Kept on a block that is
    already right, it would show every filing four hours before it existed: look-ahead,
    in every replay and every live round."""
    right = ["2026-07-14T10:30:38.000Z", "2026-01-13T11:45:10.000Z"]
    edgar = _Edgar(right, TRUE)
    assert _read(edgar) == ["2026-07-14 06:30:38", "2026-01-13 06:45:10"]
    assert sum("-index.htm" in u for u in edgar.asked) == 2   # both ends, on this read: no cache


def test_a_gap_of_any_other_size_or_a_mixed_block_stops_the_read():
    """Only the measured fault is corrected. A 3-hour gap, or a late newest 8-K over a right
    oldest one, is something nobody has measured; guessed at, it could move filings
    earlier than they were public."""
    off_by_3 = ["2026-07-14T13:30:38.000Z", "2026-01-13T14:45:10.000Z"]
    with pytest.raises(earnings.EdgarTimeError, match="from its filing page"):
        _read(_Edgar(off_by_3, TRUE))
    mixed = ["2026-07-14T14:30:38.000Z", "2026-01-13T11:45:10.000Z"]
    with pytest.raises(earnings.EdgarTimeError, match="not a pattern measured"):
        _read(_Edgar(mixed, TRUE))


def test_a_live_round_falls_back_to_the_snapshot_when_edgar_times_cannot_be_trusted(tmp_path, monkeypatch):
    """A failed check must never drop the round's filings: the snapshot stands in, and the
    meta says EDGAR was not used and why."""
    _snapshot(tmp_path, monkeypatch)
    monkeypatch.setenv("SEC_USER_AGENT", "test test@example.com")

    def fetch(tickers, recent_only):
        raise earnings.EdgarTimeError("CIK 19617: not correcting a mixed block")

    events, meta = live.load_filings(["AAPL"], pd.Timestamp("2026-10-08 10:20", tz=TZ), tmp_path,
                                     fetch=fetch)
    assert meta["source"] == "snapshot" and meta["stale"]
    assert "EdgarTimeError" in meta["edgar_error"]
    assert list(events["ticker"]) == ["AAPL"]


def test_a_block_right_at_the_top_and_late_below_moves_only_the_late_8ks():
    """By the evening of Oct 6, EDGAR served filings accepted that day right and every older
    one late, so JPM's newest filing agreed with its page while its January results did
    not. Checked at the top only, the whole block passed and the late 8-Ks stayed late;
    moved whole, the right ones would have gone four hours before they existed."""
    true = [f"2026-10-0{d} 0{h}:30:00" for d, h in ((6, 9), (6, 8), (5, 7), (2, 6), (1, 6), (1, 5))]
    accs = [f"0000019617-26-00000{i}" for i in range(len(true))]
    sent = [pd.Timestamp(t).tz_localize("America/New_York").tz_convert("UTC") for t in true]
    sent = [(ts + pd.Timedelta(hours=4 if i >= 2 else 0)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            for i, ts in enumerate(sent)]
    edgar = _Edgar(sent, dict(zip(accs, true)))
    edgar.block = {"form": ["8-K"] * 6 + ["424B2"], "items": ["8.01"] * 6 + [""],
                   "acceptanceDateTime": sent + ["2026-01-01T12:00:00.000Z"],
                   "accessionNumber": accs + ["0000019617-26-999999"],
                   "primaryDocument": ["d.htm"] * 7}
    checks = []
    fixed = earnings.checked_times(edgar, CIK, edgar.block, sleep=0, checks=checks)
    got = filings.parse_events(fixed)["accepted"].dt.strftime("%Y-%m-%d %H:%M:%S")
    assert sorted(got) == sorted(true)
    assert fixed["acceptanceDateTime"][6] == "2026-01-01T12:00:00.000Z"   # not an 8-K: as sent
    assert checks[0]["late"] == 4 and checks[0]["probes"] <= 2 + 3


class _History:
    """A name with a recent block, two older history pages and a former CIK's block."""

    def __init__(self):
        def block(acc, at):
            return {"form": ["8-K"], "items": ["2.02"], "acceptanceDateTime": [at],
                    "accessionNumber": [acc], "primaryDocument": ["r.htm"]}
        self.subs = {
            "CIK0000000002.json": {"filings": {
                "recent": block("0000000002-26-000001", "2026-01-20T11:30:00.000Z"),
                "files": [{"name": "CIK0000000002-submissions-001.json", "filingFrom": "2025-07-01", "filingTo": "2025-12-31"},
                          {"name": "CIK0000000002-submissions-002.json", "filingFrom": "2025-01-01", "filingTo": "2025-06-30"}]}},
            "CIK0000000002-submissions-001.json": block("0000000002-25-000009", "2025-10-14T10:30:00.000Z"),
            "CIK0000000002-submissions-002.json": block("0000000002-25-000004", "2025-04-11T10:45:00.000Z"),
            "CIK0000000001.json": {"filings": {"recent": block("0000000001-25-000003", "2025-04-15T20:15:00.000Z"),
                                               "files": []}},
        }
        self.pages = {"0000000002-26-000001": "2026-01-20 06:30:00", "0000000002-25-000009": "2025-10-14 06:30:00",
                      "0000000002-25-000004": "2025-04-11 06:45:00", "0000000001-25-000003": "2025-04-15 16:15:00"}
        self.asked = []

    def get(self, url):
        self.asked.append(url)
        name = url.rsplit("/", 1)[1]
        if name in self.subs:
            return _Resp(self.subs[name])
        acc = name.removesuffix("-index.htm")
        return _Resp(f'<div class="infoHead">Accepted</div>\n<div class="info">{self.pages[acc]}</div>')


def test_a_replay_window_older_than_the_recent_block_reads_the_history_page_that_covers_it(monkeypatch):
    """JPM's recent block starts 2025-10-06, so its Apr 2025 results were never in the
    replay's texts: the desk saw "8-K 2.02" and no press release, in the window where the
    banks open results season. XOM's filings under its old CIK were missing from every
    window. Only the pages that overlap the window are read, and each filing keeps the
    CIK it was listed under, since its text is filed there."""
    monkeypatch.setitem(earnings.FORMER_CIKS, "XYZ", (1,))
    edgar = _History()
    blocks = earnings.submission_blocks(edgar, 2, "XYZ", sleep=0, recent_only=True,
                                        window=(pd.Timestamp("2025-04-11 09:10", tz=TZ),
                                                pd.Timestamp("2025-05-02 15:25", tz=TZ)))
    pages = [u.rsplit("/", 1)[1] for u in edgar.asked if u.endswith(".json")]
    assert pages == ["CIK0000000002.json", "CIK0000000002-submissions-002.json", "CIK0000000001.json"]
    ev = pd.concat([filings.parse_events(b, documents=True).assign(cik=b["_cik"]) for b in blocks])
    by_acc = ev.set_index("accession")
    assert by_acc.loc["0000000002-25-000004", "cik"] == 2 and by_acc.loc["0000000001-25-000003", "cik"] == 1
    assert by_acc.loc["0000000001-25-000003", "accepted"].strftime("%H:%M") == "16:15"

    edgar = _History()   # live (no window): the recent block alone, as before
    earnings.submission_blocks(edgar, 2, "XYZ", sleep=0, recent_only=True)
    assert [u.rsplit("/", 1)[1] for u in edgar.asked if u.endswith(".json")] == ["CIK0000000002.json"]
