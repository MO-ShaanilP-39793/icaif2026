from datetime import date

import pandas as pd
import pytest

from icaif import calendar, data, parity, public_bars

HAVE_PANEL = data.ORGANIZER_PARQUET.exists()
needs_panel = pytest.mark.skipif(not HAVE_PANEL, reason="organizer parquet not present")


@pytest.fixture(scope="module")
def organizer():
    return data.load_organizer_bars()


def test_a_half_day_runs_only_the_rounds_before_its_early_close():
    """Rounds at or after a 13:00 close are cancelled; simulating them trades into bars
    that do not exist in the regular session."""
    assert [r["round"] for r in calendar.rounds_for(date(2025, 11, 28))] == [1, 2, 3, 4]
    assert len(calendar.rounds_for(date(2025, 11, 26))) == 7


@needs_panel
def test_no_extended_hours_bar_survives_on_a_half_day(organizer):
    """The raw panel carries post-close prints on every early-close day; kept, they
    become the half-day's 16:00 close and a fake afternoon return."""
    bars, issues = organizer
    half = bars[bars["start"].dt.date.isin(calendar.EARLY_CLOSES)]
    assert not half.empty
    assert (half["start"].dt.hour < 13).all()
    assert (half["end"] <= half["start"].dt.normalize() + pd.Timedelta(hours=13)).all()
    assert issues.dropped_extended_hours > 0


@needs_panel
def test_organizer_bar_ends_follow_its_own_grid_not_a_fixed_hour(organizer):
    """The 09:30 bar is 30 minutes long. Assuming an hour hands every round-2 decision
    the 10:00-10:30 prices it could not have seen."""
    bars, _ = organizer
    length = (bars["end"] - bars["start"]).dt.total_seconds() / 60
    first = bars["start"].dt.time == calendar.SESSION_OPEN
    assert (length[first] == 30).all()
    assert (length[~first] == 60).all()


@needs_panel
def test_spin_off_ex_dates_no_longer_read_as_twenty_percent_crashes(organizer):
    """Unadjusted, T 2022-04-11 and GE 2023-01-04 / 2024-04-02 each gap down ~20% on a
    distribution, not a loss -- three of the panel's worst days, all fake."""
    bars, _ = organizer
    daily = parity._daily(bars).reset_index()
    daily["gap"] = daily["open"] / daily.groupby("ticker")["close"].shift() - 1
    for action in data.CORPORATE_ACTIONS:
        row = daily[(daily["ticker"] == action["ticker"])
                    & (daily["day"].astype(str) == action["ex_date"])]
        assert abs(float(row["gap"].iloc[0])) < 0.05, action


def test_a_bar_still_in_progress_is_not_treated_as_complete():
    """Yahoo serves the current bar with the latest trade as its close."""
    start = pd.Timestamp("2026-09-24 14:30", tz=calendar.TZ)
    bars = pd.DataFrame({"start": [start - pd.Timedelta(hours=1), start],
                         "end": [start, start + pd.Timedelta(hours=1)]})
    kept = public_bars.completed(bars, as_of=start + pd.Timedelta(minutes=20))
    assert list(kept["start"]) == [start - pd.Timedelta(hours=1)]


def test_the_last_public_hourly_bar_ends_at_the_close_not_an_hour_later():
    start = pd.Series([pd.Timestamp("2026-09-24 15:30", tz=calendar.TZ),
                       pd.Timestamp("2025-11-28 12:30", tz=calendar.TZ)])
    ends = public_bars._end(start, 60)
    assert ends.iloc[0] == pd.Timestamp("2026-09-24 16:00", tz=calendar.TZ)
    assert ends.iloc[1] == pd.Timestamp("2025-11-28 13:00", tz=calendar.TZ)


def test_fill_approximation_is_scored_against_the_open_of_the_half_hour_bar():
    """On a price path whose :30 open differs from the hour's open, `bar_open` must show
    that gap in bps; a scorer that compared the wrong halves would report zero."""
    day = "2026-07-01"
    rows = []
    for hh in range(10, 16):
        a = pd.Timestamp(f"{day} {hh}:00", tz=calendar.TZ)
        rows.append({"ticker": "X", "start": a, "open": 100.0, "high": 100.0,
                     "low": 100.0, "close": 100.0})
        rows.append({"ticker": "X", "start": a + pd.Timedelta(minutes=30), "open": 101.0,
                     "high": 101.0, "low": 101.0, "close": 101.0})
    out = parity.fill_approximation(pd.DataFrame(rows))
    assert out["bar_open"]["n"] == 6
    assert out["bar_open"]["mean_bps"] == pytest.approx((100 / 101 - 1) * 1e4, abs=0.01)
    assert out["bar_close"]["mean_bps"] == 0.0
