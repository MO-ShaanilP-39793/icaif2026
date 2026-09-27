import numpy as np
import pandas as pd
import pytest

from icaif import calendar, data, earnings, universe


def _daily(dv_by_symbol: dict[str, list[float]], start="2024-01-02") -> pd.DataFrame:
    n = len(next(iter(dv_by_symbol.values())))
    dates = pd.bdate_range(start, periods=n)
    return pd.concat([pd.DataFrame({"date": dates, "ticker": s, "close": 1.0,
                                    "volume": np.asarray(v, float)})
                      for s, v in dv_by_symbol.items()], ignore_index=True)


def _membership(rows) -> pd.DataFrame:
    return pd.DataFrame({"ticker": [r[0] for r in rows], "symbol": [r[0] for r in rows],
                         "start": pd.to_datetime([r[1] for r in rows]),
                         "end": pd.to_datetime([r[2] for r in rows])})


def test_a_name_ranks_on_dollar_volume_before_the_day_never_including_it():
    """A spike on day d must not lift a name into the universe on d itself: a model
    trained on that universe would pick names on the day they became interesting."""
    n = 80
    quiet, steady = [1.0] * n, [5.0] * n
    spiky = quiet.copy()
    spiky[70] = 1e6
    daily = _daily({"AAA": steady, "BBB": spiky})
    m = _membership([("AAA", "2000-01-01", None), ("BBB", "2000-01-01", None)])
    uni = universe.build(daily, m, top_n=1)
    d = sorted(daily["date"].unique())
    assert uni.loc[d[70], "AAA"] and not uni.loc[d[70], "BBB"]
    assert uni.loc[d[71], "BBB"]


def test_membership_ends_the_day_a_name_leaves_the_index():
    dates = pd.bdate_range("2024-03-01", periods=5)
    m = _membership([("AAA", "2024-03-04", "2024-03-06")])
    mask = universe.member_mask(m, dates)["AAA"]
    assert mask.tolist() == [False, True, True, False, False]


def test_the_competition_names_stay_in_the_universe_whatever_their_rank():
    """The daily model flags and upweights the 30. A flag on a name that fell out of the
    training universe marks nothing, silently."""
    ours = sorted(data.load_universe())[0]
    daily = _daily({"BIG": [1e9] * 80, ours: [1.0] * 80})
    m = _membership([("BIG", "2000-01-01", None)])
    uni = universe.build(daily, m, top_n=1)
    assert uni[ours].iloc[40:].all()


def test_class_shares_map_to_yahoo_symbols():
    assert universe.yahoo_symbol("BRK.B") == "BRK-B"


def test_edgar_times_are_utc_so_an_after_close_release_stays_after_the_close():
    """Apple's 16:30 ET release is stamped 20:30Z. Read as Eastern wall-clock, a pre-open
    release stamped 12:00Z would land mid-session and map to the next day's open."""
    block = {"form": ["8-K"], "items": ["2.02,9.01"],
             "acceptanceDateTime": ["2024-08-01T20:30:40.000Z"]}
    ts = earnings.parse_filings(block)["accepted"].iloc[0]
    assert ts == pd.Timestamp("2024-08-01 16:30:40", tz=calendar.TZ)


def test_only_8ks_carrying_item_2_02_count_as_earnings():
    block = {"form": ["8-K", "8-K", "8-K/A", "10-Q", "8-K"],
             "items": ["2.02,9.01", "7.01,9.01", "2.02", "", "5.02,2.02"],
             "acceptanceDateTime": ["2024-01-01T08:00:00.000Z"] * 5}
    assert len(earnings.parse_filings(block)) == 2


@pytest.mark.parametrize("accepted, reflected", [
    ("2024-08-01 16:30", "2024-08-02"),  # Thursday after the close -> Friday open
    ("2024-08-01 07:00", "2024-08-01"),  # before the open -> that open
    ("2024-08-02 16:05", "2024-08-05"),  # Friday after the close -> Monday
    ("2024-08-03 12:00", "2024-08-05"),  # Saturday, any time -> Monday, not Tuesday
    ("2024-07-04 16:30", "2024-07-05"),  # holiday -> next session
])
def test_a_release_is_reflected_at_the_first_open_after_it(accepted, reflected):
    sessions = pd.DatetimeIndex([d for d in pd.bdate_range("2024-07-01", "2024-08-09")
                                 if d != pd.Timestamp("2024-07-04")])
    got = earnings.reaction_session(pd.Series([pd.Timestamp(accepted, tz=calendar.TZ)]), sessions)
    assert got.iloc[0] == pd.Timestamp(reflected)


def test_a_preannouncement_weeks_before_earnings_is_not_the_earnings_event():
    """Tesla files delivery numbers under item 2.02 about three weeks before results.
    Kept as a separate event, "sessions to next earnings" would point at the preview."""
    ts = pd.to_datetime(["2024-01-02 09:00", "2024-01-24 16:10", "2024-04-02 09:00",
                         "2024-04-23 16:10"]).tz_localize(calendar.TZ)
    got = earnings.quarterly(pd.DataFrame({"ticker": "TSLA", "accepted": ts}))
    assert got["accepted"].tolist() == [ts[1], ts[3]]
