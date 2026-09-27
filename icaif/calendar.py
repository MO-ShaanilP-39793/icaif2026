"""Session calendar: when the market is open, and when each competition round executes.

Everything here is US Eastern. Bars carry tz-aware New York timestamps so that no
comparison between a bar and a round silently mixes naive and aware times -- pandas
refuses that comparison, which is the point.
"""

from datetime import date, time

import pandas as pd

TZ = "America/New_York"

SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)
EARLY_CLOSE = time(13, 0)

# NYSE 13:00 closes inside the organizer data and the competition year. The organizer
# panel carries extended-hours prints after 13:00 on exactly these days (13:00-15:00 bars
# and odd 14:30/15:30 bars); without this list they read as a normal afternoon, and the
# 16:00 "close" of a half-day becomes a thin post-market trade. Nothing downstream would
# look wrong: the day simply gets a fake afternoon return.
EARLY_CLOSES = frozenset(
    date.fromisoformat(d)
    for d in (
        # 2016-2020 were added when Alpaca extended history to 2016. Without them the
        # half-days read as full sessions: rounds after 13:00 scored on thin after-close
        # prints, which is the extended-hours trap the organizer panel had. Confirmed in
        # the data, not just the NYSE calendar: volume from 13:30 on is ~0 on exactly these
        # days (tests/test_alpaca.py).
        "2016-11-25",
        "2017-07-03",
        "2017-11-24",
        "2018-07-03",
        "2018-11-23",
        "2018-12-24",
        "2019-07-03",
        "2019-11-29",
        "2019-12-24",
        "2020-11-27",
        "2020-12-24",
        "2021-11-26",
        "2022-11-25",
        "2023-07-03",
        "2023-11-24",
        "2024-07-03",
        "2024-11-29",
        "2024-12-24",
        "2025-07-03",
        "2025-11-28",
        "2025-12-24",
        "2026-11-27",
        "2026-12-24",
    )
)

# Round -> (upload deadline, execution time). Execution is the open of the hourly bar
# that starts at that time on the organizers' live grid (:30), not the historical grid.
ROUNDS = {
    1: (time(9, 10), time(9, 30)),
    2: (time(10, 25), time(10, 30)),
    3: (time(11, 25), time(11, 30)),
    4: (time(12, 25), time(12, 30)),
    5: (time(13, 25), time(13, 30)),
    6: (time(14, 25), time(14, 30)),
    7: (time(15, 25), time(15, 30)),
}


def session_close(day: date) -> time:
    return EARLY_CLOSE if day in EARLY_CLOSES else SESSION_CLOSE


def at(day: date, t: time) -> pd.Timestamp:
    return pd.Timestamp.combine(day, t).tz_localize(TZ)


def rounds_for(day: date) -> list[dict]:
    """The rounds that actually run on `day`.

    The rules cancel rounds scheduled at or after an early close, so on a half-day only
    rounds 1-4 execute. A simulator that ran all seven would trade into bars that do not
    exist in the regular session.
    """
    close = session_close(day)
    return [
        {"round": n, "deadline": at(day, dl), "execution": at(day, ex)}
        for n, (dl, ex) in ROUNDS.items()
        if ex < close
    ]
