import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from icaif import calendar, filings, macro, news
from icaif.agents.brains import RuleBrain
from icaif.agents.desk import Desk, DeskConfig
from tests.test_agents import DAYS, START, Scripted, _mkt
from tests.test_sim import TICKERS

import icaif.sim as sim

TZ = calendar.TZ


def _context(days, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(pd.to_datetime(days))
    cols = {"SPY": 400, "^VIX": 18, "^IRX": 4.0, "^FVX": 4.1, "^TNX": 4.3, **{s: 80 for s in macro.SECTORS}}
    w = pd.DataFrame({k: v * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))) for k, v in cols.items()},
                     index=idx)
    return w


def test_macro_readings_for_a_day_ignore_that_day_and_every_later_one():
    """A decision at 09:10 cannot know today's close; a cut at <= would hand it over."""
    days = pd.bdate_range("2025-01-01", periods=300)
    w = _context(days)
    d = days[250].date()
    later = w.copy()
    later.loc[later.index >= pd.Timestamp(d)] *= 1.5
    assert macro.readings(w, d) == macro.readings(later, d)


def test_anonymised_macro_carries_no_level_that_could_date_the_window():
    """VIX 13 with a 4.9% ten-year dates a window as surely as a date does."""
    w = _context(pd.bdate_range("2025-01-01", periods=300))
    d = w.index[280].date()
    anon, real = macro.readings(w, d, anonymize=True), macro.readings(w, d)
    levels = {"vix", "yield_3m_pct", "yield_5y_pct", "yield_10y_pct", "curve_10y_3m_pct"}
    assert levels <= set(real) and not levels & set(anon)
    assert "vix_z_1y" in anon and "yield_10y_chg_20d_bp" in anon


FED_PAGE = """
<a>2024 FOMC Meetings</a></h4></div>
<div class="fomc-meeting__month col"><strong>January</strong></div>
<div class="fomc-meeting__date col">30-31</div>
<div class="fomc-meeting__month col"><strong>March</strong></div>
<div class="fomc-meeting__date col">19-20*</div>
<div class="fomc-meeting__month col"><strong>Apr/May</strong></div>
<div class="fomc-meeting__date col">30-1</div>
<div class="fomc-meeting__month col"><strong>August</strong></div>
<div class="fomc-meeting__date col">16 (notation vote)</div>
<a>2023 FOMC Meetings</a></h4></div>
<div class="fomc-meeting__month col"><strong>Dec</strong></div>
<div class="fomc-meeting__date col">12-13*</div>
"""


def test_fomc_parsing_takes_the_last_day_across_a_month_and_skips_notation_votes():
    """A notation vote was never scheduled; flagged, it tells the agent the market knew
    of a decision it did not."""
    d, years = macro.parse_fomc(FED_PAGE)
    assert years == [2023, 2024]
    assert [str(x.date()) for x in d["date"]] == ["2023-12-13", "2024-01-31", "2024-03-20",
                                                   "2024-05-01"]
    assert d.set_index("date")["projections"].to_dict()[pd.Timestamp("2024-03-20")]


def test_a_day_outside_the_calendars_years_reads_unknown_not_no_meeting():
    d, years = macro.parse_fomc(FED_PAGE)
    cal = macro.FomcCalendar(d, years)
    sessions = [x.date() for x in pd.bdate_range("2024-01-25", "2024-02-10")]
    assert cal.block(date(2022, 6, 1), sessions)["fomc_decision_today"] is None
    today = cal.block(date(2024, 1, 31), sessions)
    assert today["fomc_decision_today"] is True and today["fomc_statement_time_et"] == "14:00"
    assert cal.block(date(2024, 1, 29), sessions)["sessions_to_next_fomc"] == 2


RSS = """<?xml version="1.0"?><rss><channel>
<item><title>A</title><description>first</description><guid>g1</guid>
<pubDate>Tue, 29 Sep 2026 11:00:00 +0000</pubDate></item>
<item><title>B</title><description>no time</description><guid>g2</guid><pubDate>soon</pubDate></item>
</channel></rss>"""


def test_a_headline_without_a_parseable_time_is_dropped_not_placed_arbitrarily():
    rows = news.parse_rss(RSS, "AAPL", pd.Timestamp("2026-09-29 09:00", tz=TZ))
    assert list(rows["guid"]) == ["g1"]
    assert rows["published"].iloc[0] == pd.Timestamp("2026-09-29 07:00", tz=TZ)


def test_every_feed_empty_raises_and_one_failed_feed_does_not_lose_the_rest():
    """All-empty read downstream as "no news anywhere": quiet, and in the direction
    that removes a risk flag."""
    empty = '<?xml version="1.0"?><rss><channel></channel></rss>'
    with pytest.raises(RuntimeError):
        news.fetch(["A", "B"], sleep=0, get=lambda t: empty)

    def get(t):
        if t == "B":
            raise OSError("reset")
        return RSS
    frame, issues = news.fetch(["A", "B"], sleep=0, get=get)
    assert set(frame["ticker"]) == {"A"} and issues["failed"][0].startswith("B:")


def test_the_archive_reads_only_what_had_been_fetched_by_the_deadline(tmp_path):
    """A headline published before the deadline but fetched after it is one we did not
    have; replayed as known, the agent reads news it never saw."""
    t0 = pd.Timestamp("2026-10-12 08:00", tz=TZ)
    early = news.parse_rss(RSS, "AAPL", t0)
    news.save(early, t0, tmp_path)
    with pytest.raises(FileExistsError):
        news.save(early, t0, tmp_path)
    late_rss = RSS.replace("g1", "g3").replace("Tue, 29 Sep 2026 11:00:00", "Mon, 12 Oct 2026 12:00:00")
    t1 = pd.Timestamp("2026-10-12 08:30", tz=TZ)
    news.save(news.parse_rss(late_rss, "AAPL", t1), t1, tmp_path)
    news.save(early.assign(fetched_at=t1), t1 + pd.Timedelta(seconds=1), tmp_path)
    got = news.as_of(pd.Timestamp("2026-10-12 08:10", tz=TZ), lookback_days=30, directory=tmp_path)
    assert [h["title"] for h in got["AAPL"]] == ["A"]
    later = news.as_of(pd.Timestamp("2026-10-12 09:00", tz=TZ), lookback_days=30, directory=tmp_path)
    # g3 (published 08:00, fetched 08:30) is now known; g1 appears once, not per snapshot.
    assert [h["published"][:16] for h in later["AAPL"]] == ["2026-10-12 08:00", "2026-09-29 07:00"]


BLOCK = {"form": ["8-K", "10-Q", "8-K/A", "8-K"],
         "items": ["5.02,9.01", "", "2.02", "9.01"],
         "acceptanceDateTime": ["2026-10-12T20:30:00.000Z", "2026-10-13T12:00:00.000Z",
                                "2026-10-14T13:05:00.000Z", "2026-10-14T14:00:00.000Z"]}


def test_8k_events_are_read_in_eastern_time_and_exhibit_only_filings_are_not_events():
    e = filings.parse_events(BLOCK)
    assert list(e["items"]) == ["5.02", "2.02"]
    assert e["accepted"].iloc[0] == pd.Timestamp("2026-10-12 16:30", tz=TZ)
    assert list(e["amended"]) == [False, True]


def test_an_8k_accepted_after_the_deadline_is_not_yet_an_event():
    e = filings.parse_events(BLOCK).assign(ticker="AAPL")
    got = filings.recent(e, pd.Timestamp("2026-10-14 09:00", tz=TZ))
    assert [x["events"] for x in got["AAPL"]] == [["director or officer change"]]
    assert got["AAPL"][0]["hours_ago"] == pytest.approx(40.5, abs=0.1)
    at_cut = filings.recent(e, pd.Timestamp("2026-10-14 09:10", tz=TZ))["AAPL"]
    assert [x["events"] for x in at_cut] == [["results of operations"], ["director or officer change"]]
    assert at_cut[0]["amendment"] is True


def test_a_replayed_observation_gets_macro_and_filings_but_no_levels_and_no_headlines(tmp_path):
    days_ctx = pd.bdate_range(DAYS[0] - pd.Timedelta(days=500), DAYS[-1])
    w = _context(days_ctx)
    ev = pd.DataFrame({"ticker": [TICKERS[0]], "amended": [False], "items": ["5.02"],
                       "accepted": [calendar.at(START, calendar.ROUNDS[1][0]) - pd.Timedelta(hours=20)]})
    t0 = calendar.at(START, calendar.ROUNDS[1][0]) - pd.Timedelta(hours=1)
    news.save(news.parse_rss(RSS, TICKERS[0], t0).assign(published=t0), t0, tmp_path)
    b = Scripted()
    d = Desk(b, DeskConfig(anonymize=True), context=w, filings=ev, news_dir=tmp_path)
    sim.run(d, _mkt(), START, 1)
    obs = b.seen[0][1]
    assert "vix_z_1y" in obs["macro"] and "vix" not in obs["macro"]
    flagged = [r for r in obs["names"] if r["recent_8k_filings"]]
    assert len(flagged) == 1 and flagged[0]["name"] != TICKERS[0]
    assert "headlines" not in json.dumps(obs)
