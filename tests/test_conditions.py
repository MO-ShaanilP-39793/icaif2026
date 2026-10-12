"""The plan's conditions, evaluated by code each round (icaif/agents/conditions.py)."""

import numpy as np
import pandas as pd
import pytest

from icaif import calendar, sim
from icaif.agents import conditions as C
from icaif.agents import triggers as T
from tests.test_quant import _bars, _days
from tests.test_sim import TICKERS, _market

DAYS = _days(150)
E = 120                      # the entry's session
NAME = "AAPL"


def _ctx(m, day, rnd: int) -> sim.RoundContext:
    r = calendar.rounds_for(day)[rnd - 1]
    return sim.RoundContext(day, rnd, r["deadline"], r["execution"], {}, sim.INITIAL_NAV, m)


def _set(bars, t, after, factor):
    """`t`'s every bar ended after `after`, times `factor`."""
    out = bars.copy()
    hit = (out["ticker"] == t) & (out["end"] > after)
    out.loc[hit, ["open", "high", "low", "close"]] *= factor
    return out


def _cond(kind, threshold=None, name=None, action="review", item=None, gross=None):
    return {"kind": kind, "name": name, "threshold": threshold, "item": item,
            "action": action, "gross": gross, "why": "test"}


def _setup(bars, conds, scale=1.0):
    """A watch made at round 1 of the entry session, and the positions bought there."""
    m = _market(DAYS, info_bars=bars)
    ctx = _ctx(m, DAYS[E], 1)
    marks = C.entry_marks(ctx, TICKERS, scale=scale)
    px = m.recent_closes(ctx.deadline, 1).iloc[-1]
    pos = {t: {"entry_price": float(px[t]), "entry_at": ctx.execution.isoformat()} for t in TICKERS}
    return m, C.Watch(conds, marks), pos


def _check(w, m, day, rnd, pos, nav=None, base=None, **kw):
    return w.check(_ctx(m, day, rnd), to_ticker=lambda c: c, positions=pos, nav=nav,
                   base_nav=base, **kw)


def test_a_condition_fires_once_and_stays_fired_until_the_plan_is_rewritten():
    """Re-armed on recovery, a condition sitting on its threshold would reach the PM each
    round it crossed back, and a decline would mean nothing."""
    entry_end = calendar.rounds_for(DAYS[E])[0]["execution"]
    bars = _set(_bars(DAYS, 8), NAME, entry_end + pd.Timedelta(days=2), 0.90)
    m, w, pos = _setup(bars, [_cond("move_from_entry_pct", -5.0, NAME, "exit")])
    before = _check(w, m, DAYS[E + 1], 3, pos)[0]
    assert not before["fired"] and before["to_fire"] > 0
    first = _check(w, m, DAYS[E + 3], 3, pos)[0]
    assert first["fired"] and first["new"] and first["to_fire"] <= 0
    again = _check(w, m, DAYS[E + 4], 3, pos)[0]
    assert again["fired"] and not again["new"] and again["fired_at"] == first["fired_at"]
    w.rewrite(keep=[], add=[_cond("move_from_entry_pct", -5.0, NAME, "exit")])
    fresh = _check(w, m, DAYS[E + 5], 3, pos)[0]
    assert fresh["id"] == "c2" and fresh["new"], "a rewritten condition is new, and its id is too"


@pytest.mark.parametrize("rnd", [1, 4, 7])
def test_no_reading_changes_when_every_price_and_filing_after_the_deadline_is_rewritten(rnd):
    """A condition read past its deadline would fire on the move it is meant to warn of."""
    bars = _bars(DAYS, 8)
    day = DAYS[E + 3]
    deadline = calendar.rounds_for(day)[rnd - 1]["deadline"]
    late = bars.copy()
    hit = late["end"] > deadline
    late.loc[hit, ["open", "high", "low", "close"]] *= 3.0
    filings = pd.DataFrame({"ticker": [NAME], "accepted": [deadline - pd.Timedelta(hours=30)],
                            "items": ["2.05"]})
    later = pd.concat([filings, pd.DataFrame({"ticker": [NAME, NAME],
                                              "accepted": [deadline + pd.Timedelta(minutes=1)] * 2,
                                              "items": ["4.02", "2.05"]})], ignore_index=True)
    conds = [_cond("move_from_entry_pct", -50.0, NAME), _cond("move_from_entry_sigma", 9.0, NAME),
             _cond("give_back_from_peak", 0.5, NAME), _cond("new_8k_item", name=NAME, item="4.02"),
             _cond("basket_move_from_entry_pct", 40.0), _cond("drawdown_from_peak_pct", 3.0),
             _cond("vol_ratio_above", 1.8)]
    got = []
    for b, f in ((bars, filings), (late, later)):
        m, w, pos = _setup(b, conds)
        nav = 1.0 + float(m.recent_closes(deadline, 1).iloc[-1][NAME]) / 1000
        got.append(_check(w, m, day, rnd, pos, nav=nav, base=1.0, filings=f))
    assert got[0] == got[1]
    assert {r["kind"]: r["value"] for r in got[0]}["new_8k_item"] == 0.0


def test_a_name_with_no_price_at_the_deadline_reads_none_and_never_fires():
    """A forward-filled price is a zero move that never happened: a condition watching
    for a fall would read "no fall" through the outage."""
    bars = _set(_bars(DAYS, 8), NAME, calendar.rounds_for(DAYS[E])[0]["execution"], 0.5)
    day = DAYS[E + 2]
    last_end = bars.loc[bars["end"] <= calendar.rounds_for(day)[2]["deadline"], "end"].max()
    bars.loc[(bars["ticker"] == NAME) & (bars["end"] == last_end), "close"] = np.nan
    m, w, pos = _setup(bars, [_cond("move_from_entry_pct", -10.0, NAME, "exit")])
    row = _check(w, m, day, 3, pos)[0]
    assert row["value"] is None and not row["fired"] and "no price" in row["note"]


def test_book_conditions_read_the_book_at_the_pms_scale_under_the_overlay():
    """The PM wrote "down 3% from the peak" for the book it chose. Read at the real scale,
    a 10%-invested book would need a 30% fall in its names, and the condition would
    never fire."""
    m, w, pos = _setup(_bars(DAYS, 8), [_cond("drawdown_from_peak_pct", 3.0),
                                        _cond("drawdown_from_entry_pct", 3.0)], scale=10.0)
    _check(w, m, DAYS[E + 1], 2, pos, nav=1_000_000.0, base=1_000_000.0)
    rows = _check(w, m, DAYS[E + 2], 2, pos, nav=996_000.0, base=1_000_000.0)
    assert [r["value"] for r in rows] == [pytest.approx(4.0), pytest.approx(4.0)]
    assert all(r["fired"] for r in rows)
    m, w1, pos = _setup(_bars(DAYS, 8), [_cond("drawdown_from_peak_pct", 3.0)], scale=1.0)
    _check(w1, m, DAYS[E + 1], 2, pos, nav=1_000_000.0, base=1_000_000.0)
    row = _check(w1, m, DAYS[E + 2], 2, pos, nav=996_000.0, base=1_000_000.0)[0]
    assert row["value"] == pytest.approx(0.4) and not row["fired"]


def test_a_sigma_move_is_measured_in_the_sigma_fixed_at_entry():
    """A sigma read later moves the goalposts: in a falling market vol rises, and a
    2.5-sigma fall measured in it fires later than the one the PM meant."""
    entry_end = calendar.rounds_for(DAYS[E])[0]["execution"]
    bars = _set(_bars(DAYS, 8), NAME, entry_end + pd.Timedelta(days=2), 0.97)
    m, w, pos = _setup(bars, [_cond("move_from_entry_sigma", -2.5, NAME)])
    sd = w.marks["sigma"][NAME]
    assert w.marks["sigma_source"][NAME] == "realised_20d" and sd > 0
    day = DAYS[E + 4]
    last = float(m.recent_closes(calendar.rounds_for(day)[3]["deadline"], 1).iloc[-1][NAME])
    row = _check(w, m, day, 4, pos)[0]
    assert row["value"] == pytest.approx(np.log(last / pos[NAME]["entry_price"]) / sd, abs=1e-3)


def test_entry_marks_take_the_har_forecast_where_there_is_one():
    class Vol:
        def for_day(self, day, deadline):
            return pd.DataFrame({"har_h1": {NAME: 0.0004}, "har_h3": {NAME: 0.0004}})

    m = _market(DAYS, info_bars=_bars(DAYS, 8))
    marks = C.entry_marks(_ctx(m, DAYS[E], 1), TICKERS, vol=Vol())
    assert marks["sigma"][NAME] == pytest.approx(0.02) and marks["sigma_source"][NAME] == "har_1d"
    assert marks["sigma_source"]["MSFT"] == "realised_20d"


@pytest.mark.parametrize("cond", [_cond("drawdown_from_peak_pct", -3.0), _cond("move_from_entry_pct", 0.0, NAME),
                                  _cond("vol_ratio_above", None)])
def test_a_threshold_with_no_readable_direction_never_fires_and_says_why(cond):
    """A drawdown of -3 would read as "fire at once"; guessing the PM meant 3 would be
    code writing the plan."""
    m, w, pos = _setup(_bars(DAYS, 8), [cond])
    row = _check(w, m, DAYS[E + 2], 3, pos, nav=900_000.0, base=1_000_000.0)[0]
    assert not row["fired"] and row["note"].startswith("never fires")


def test_an_8k_fires_only_when_filed_after_the_entry_and_by_the_deadline():
    entry = calendar.rounds_for(DAYS[E])[0]["deadline"]
    day = DAYS[E + 2]
    deadline = calendar.rounds_for(day)[2]["deadline"]
    m, w, pos = _setup(_bars(DAYS, 8), [_cond("new_8k_item", name=NAME, item="2.05", action="exit")])

    def fired(*rows):
        f = pd.DataFrame([{"ticker": NAME, "accepted": a, "items": i} for a, i in rows])
        return C.Watch.from_state(w.state()).check(_ctx(m, day, 3), to_ticker=lambda c: c, positions=pos,
                                                   nav=None, base_nav=None, filings=f)[0]["fired"]

    assert not fired((entry - pd.Timedelta(hours=1), "2.05,9.01")), "known before the entry: in the plan already"
    assert not fired((deadline + pd.Timedelta(minutes=1), "2.05")), "not yet public"
    assert not fired((deadline - pd.Timedelta(hours=1), "2.02,9.01")), "another item"
    assert fired((deadline - pd.Timedelta(hours=1), "2.02, 2.05"))


def test_a_give_back_waits_for_a_gain_worth_giving_back():
    """Half of a 0.1% "gain" given back is a quiet name drifting, not a winner turning."""
    entry_end = calendar.rounds_for(DAYS[E])[0]["execution"]
    bars = _set(_bars(DAYS, 8), NAME, entry_end + pd.Timedelta(days=1), 1.0)
    m, w, pos = _setup(bars, [_cond("give_back_from_peak", 0.5, NAME, "trim_half")])
    sd = w.marks["sigma"][NAME]
    # Prices pinned: flat at entry, a peak, then a fall.
    ep = pos[NAME]["entry_price"]
    rows = bars["ticker"] == NAME
    after = rows & (bars["end"] > entry_end)
    bars.loc[after, "close"] = ep
    peak_day, fall_day = DAYS[E + 2], DAYS[E + 3]
    in_day = lambda d: bars["end"].dt.tz_convert(calendar.TZ).dt.date == d  # noqa: E731
    bars.loc[after & in_day(peak_day), "close"] = ep * (1 + 0.5 * sd)
    bars.loc[after & (bars["end"] > calendar.at(fall_day, calendar.SESSION_OPEN)), "close"] = ep
    m, w, pos = _setup(bars, [_cond("give_back_from_peak", 0.5, NAME, "trim_half")])
    row = _check(w, m, DAYS[E + 4], 3, pos)[0]
    assert row["value"] is None and row["note"].startswith("not armed") and not row["fired"]
    bars.loc[after & in_day(peak_day), "close"] = ep * (1 + 3 * sd)
    bars.loc[after & (bars["end"] > calendar.at(fall_day, calendar.SESSION_OPEN)), "close"] = ep * (1 + 1.2 * sd)
    m, w, pos = _setup(bars, [_cond("give_back_from_peak", 0.5, NAME, "trim_half")])
    row = _check(w, m, DAYS[E + 4], 3, pos)[0]
    assert row["value"] == pytest.approx(0.6, abs=1e-3) and row["fired"]


def test_a_condition_on_a_name_sold_out_is_inactive():
    m, w, pos = _setup(_bars(DAYS, 8), [_cond("move_from_entry_pct", -0.0001, NAME, "exit")])
    row = _check(w, m, DAYS[E + 2], 3, {t: p for t, p in pos.items() if t != NAME})[0]
    assert row["value"] is None and not row["fired"] and row["note"].startswith("inactive")


def test_an_earnings_gap_counts_only_on_the_reaction_to_results_released_after_entry():
    release = calendar.at(DAYS[E + 2], pd.Timestamp("16:30").time())
    events = pd.DataFrame({"ticker": [NAME], "accepted": [release]})
    bars = _set(_bars(DAYS, 8), NAME, calendar.at(DAYS[E + 3], calendar.SESSION_OPEN), 0.94)
    m, w, pos = _setup(bars, [_cond("earnings_gap_pct", -3.0, NAME, "trim_half")])
    hist = T.EarningsHistory.from_market(events, m)
    row = _check(w, m, DAYS[E + 2], 7, pos, history=hist)[0]
    assert row["value"] is None and not row["fired"]
    # The synthetic bars end an hour after each fill: none of the session has ended by round 2.
    assert _check(w, m, DAYS[E + 3], 2, pos, history=hist)[0]["note"] == "no bar of the reaction session yet"
    row = _check(w, m, DAYS[E + 3], 3, pos, history=hist)[0]
    assert row["value"] < -3.0 and row["fired"]
    later = _check(w, m, DAYS[E + 4], 2, pos, history=hist)[0]
    assert later["note"] == "not a reaction session" and later["fired"] and not later["new"]


def test_a_watch_survives_its_state_exactly():
    """Live, each round is its own process: a watch rebuilt without its latches would fire
    the same breach at every round."""
    entry_end = calendar.rounds_for(DAYS[E])[0]["execution"]
    bars = _set(_bars(DAYS, 8), NAME, entry_end + pd.Timedelta(days=2), 0.90)
    m, w, pos = _setup(bars, [_cond("move_from_entry_pct", -5.0, NAME, "exit"),
                              _cond("drawdown_from_peak_pct", 2.0)], scale=10.0)
    _check(w, m, DAYS[E + 3], 3, pos, nav=1_000_000.0, base=1_000_000.0)
    back = C.Watch.from_state(w.state())
    assert back.state() == w.state()
    a = _check(w, m, DAYS[E + 4], 3, pos, nav=999_000.0, base=1_000_000.0)
    b = _check(back, m, DAYS[E + 4], 3, pos, nav=999_000.0, base=1_000_000.0)
    assert a == b and not a[0]["new"]


def test_a_rewrite_that_keeps_an_unknown_condition_is_refused():
    m, w, pos = _setup(_bars(DAYS, 8), [_cond("move_from_entry_pct", -5.0, NAME)])
    with pytest.raises(ValueError, match="not in the plan"):
        w.rewrite(keep=["c9"], add=[])
