import pandas as pd
import pytest

from icaif import earnings_calendar

NOW = pd.Timestamp("2026-09-28 20:00", tz="America/New_York")


def _yahoo(*stamps: str) -> pd.DataFrame:
    """A get_earnings_dates frame: release times (with Yahoo's offsets) as the index."""
    idx = pd.DatetimeIndex([pd.Timestamp(s) for s in stamps], name="Earnings Date")
    return pd.DataFrame({"EPS Estimate": 1.0}, index=idx)


def test_a_placeholder_time_shifted_by_daylight_saving_stays_after_the_close():
    """Yahoo posts after-close slots past the daylight-saving change as 15:00 EST. A
    rule of "after close means 16:00 or later" would call it mid-session: same
    reaction session here, but a mid-session flag is what the timing check treats
    as a time-zone bug, and the side would disagree with EDGAR's on the same event."""
    out = earnings_calendar.parse("NVDA", _yahoo("2026-11-17 15:00-05:00"), NOW)
    assert out["side"].tolist() == ["amc"]


def test_a_release_before_the_open_reacts_that_morning_not_the_next():
    out = earnings_calendar.parse("JPM", _yahoo("2026-10-13 08:00-04:00", "2026-10-13 06:00-04:00"), NOW)
    assert set(out["side"]) == {"bmo"}
    assert (out["date"] == pd.Timestamp("2026-10-13")).all()


def test_a_date_with_no_time_is_unknown_rather_than_before_the_open():
    """Midnight is how a time-less date arrives. As a clock it is before 09:30, which
    would move every after-close reporter's reaction a session early."""
    out = earnings_calendar.parse("XYZ", _yahoo("2026-10-20 00:00-04:00"), NOW)
    assert out["side"].tolist() == ["unknown"]


def test_past_releases_are_left_to_edgar_and_never_duplicated_here():
    """Yahoo's past rows are placeholder-timed copies of events EDGAR already has
    with real timestamps; kept, each would count twice."""
    out = earnings_calendar.parse("AAPL", _yahoo("2026-07-30 16:00-04:00", "2026-10-29 16:00-04:00"), NOW)
    assert out["date"].tolist() == [pd.Timestamp("2026-10-29")]


def test_every_row_records_when_it_was_known():
    out = earnings_calendar.parse("AAPL", _yahoo("2026-10-29 16:00-04:00"), NOW)
    assert (out["fetched_at"] == NOW).all()
    assert list(out.columns) == earnings_calendar.COLUMNS


def test_an_empty_answer_for_every_name_raises_instead_of_reading_as_no_earnings(monkeypatch):
    """All-empty reads downstream as 'nobody reports soon', which removes a risk flag
    exactly when the source is broken."""
    import yfinance as yf

    class Empty:
        def __init__(self, t):
            pass

        def get_earnings_dates(self, limit):
            return None

    monkeypatch.setattr(yf, "Ticker", Empty)
    with pytest.raises(RuntimeError, match="no upcoming earnings date"):
        earnings_calendar.fetch(["AAA", "BBB"], sleep=0, now=NOW)


def test_one_name_failing_does_not_lose_the_others(monkeypatch):
    import yfinance as yf

    class Some:
        def __init__(self, t):
            self.t = t

        def get_earnings_dates(self, limit):
            if self.t == "BAD":
                raise KeyError("no such symbol")
            return _yahoo("2026-10-29 16:00-04:00")

    monkeypatch.setattr(yf, "Ticker", Some)
    rows, issues = earnings_calendar.fetch(["AAA", "BAD"], sleep=0, now=NOW)
    assert rows["ticker"].tolist() == ["AAA"]
    assert issues["failed"] and issues["failed"][0].startswith("BAD")
