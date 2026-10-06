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
    assert sum("-index.htm" in u for u in edgar.asked) == 1   # checked on this read, not cached
    late = _Edgar(["2026-07-14T14:30:38.000Z", "2026-01-13T16:45:10.000Z"], TRUE)
    _read(late)
    assert sum("-index.htm" in u for u in late.asked) == 2    # a correction is confirmed first


def test_a_gap_of_any_other_size_or_a_mixed_block_stops_the_read():
    """Only the measured fault is corrected. A 3-hour gap, or one filing page agreeing and
    another not, is something nobody has measured; guessed at, it could move filings
    earlier than they were public."""
    off_by_3 = ["2026-07-14T13:30:38.000Z", "2026-01-13T14:45:10.000Z"]
    with pytest.raises(earnings.EdgarTimeError, match="from its filing page"):
        _read(_Edgar(off_by_3, TRUE))
    mixed = ["2026-07-14T14:30:38.000Z", "2026-01-13T11:45:10.000Z"]
    with pytest.raises(earnings.EdgarTimeError, match="mixed block"):
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
