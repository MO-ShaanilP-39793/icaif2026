import glob

import pandas as pd
import pytest

from icaif import alpaca, calendar, data


def _bars_30m(day: str, starts: list[str]) -> pd.DataFrame:
    s = pd.to_datetime([f"{day} {t}" for t in starts]).tz_localize(calendar.TZ)
    d = pd.Timestamp(day).date()
    close = calendar.at(d, calendar.session_close(d))
    end = (s + pd.Timedelta(minutes=30)).where(s + pd.Timedelta(minutes=30) < close, close)
    n = len(s)
    return pd.DataFrame({"ticker": "AAA", "start": s, "end": end,
                         "open": [float(i) for i in range(n)], "high": [100.0 + i for i in range(n)],
                         "low": [-float(i) for i in range(n)], "close": [10.0 + i for i in range(n)],
                         "volume": 1.0, "source": "alpaca_30m"})


FULL_DAY = ["09:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00",
            "13:30", "14:00", "14:30", "15:00", "15:30"]


def test_thirty_minute_pairs_become_the_live_hourly_grid():
    """The competition fills at the opens of 09:30, 10:30 ... 15:30. The rebuilt 60m bar
    starting 10:30 must open at the 10:30 bar's open, not the 10:00 bar's."""
    out = alpaca.to_60m(_bars_30m("2021-03-02", FULL_DAY))
    assert [t.strftime("%H:%M") for t in out["start"]] == [
        "09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]
    first = out.iloc[0]
    assert (first["open"], first["close"], first["high"], first["low"], first["volume"]) == \
        (0.0, 11.0, 101.0, -1.0, 2.0)
    assert out.iloc[1]["open"] == 2.0
    last = out.iloc[-1]
    assert last["end"] == pd.Timestamp("2021-03-02 16:00", tz=calendar.TZ) and last["open"] == 12.0


def test_an_hour_missing_either_half_is_dropped_not_filled_from_the_other():
    """Kept, a bucket missing its 10:30 bar would open at 11:00: a fill 30 minutes late."""
    starts = [t for t in FULL_DAY if t not in ("10:30", "12:00")]
    out = alpaca.to_60m(_bars_30m("2021-03-02", starts))
    assert "10:30" not in {t.strftime("%H:%M") for t in out["start"]}
    assert "11:30" not in {t.strftime("%H:%M") for t in out["start"]}
    assert len(out) == 5


def test_a_half_day_ends_with_a_half_hour_bar_at_the_early_close():
    day = next(d for d in sorted(calendar.EARLY_CLOSES) if d.year == 2021)
    starts = ["09:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30"]
    out = alpaca.to_60m(_bars_30m(str(day), starts))
    assert out.iloc[-1]["start"].strftime("%H:%M") == "12:30"
    assert out.iloc[-1]["end"].strftime("%H:%M") == "13:00"


SNAPSHOTS = sorted(glob.glob(str(data.ROOT / "data" / "public" / "alpaca_30m_2*.parquet")))


@pytest.mark.skipif(not SNAPSHOTS, reason="no alpaca_30m snapshot")
def test_the_calendar_knows_every_half_day_in_the_data_and_no_others():
    """A half-day missing from EARLY_CLOSES reads as a full session: rounds after 13:00
    fill on thin after-close prints. It happened for 2016-2020 until this test.
    After 13:30, a half-day trades ~0% of its volume; an ordinary day trades ~30%."""
    raw = pd.read_parquet(SNAPSHOTS[-1], columns=["start", "volume"])
    day = raw["start"].dt.date
    late = raw["start"].dt.hour * 60 + raw["start"].dt.minute >= 13 * 60 + 30
    share = raw[late].groupby(day[late])["volume"].sum() / raw.groupby(day)["volume"].sum()
    in_data = set(share.index)
    looks_half = {d for d, s in share.items() if s < 0.05} | (in_data - set(share.dropna().index))
    assert looks_half == {d for d in calendar.EARLY_CLOSES if d in in_data}
