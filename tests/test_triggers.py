"""Trigger tags for the v2 desk (icaif/agents/triggers.py)."""

import numpy as np
import pandas as pd
import pytest

from icaif import calendar, sim
from icaif.agents import triggers as T
from icaif.agents.desk import EarningsCalendar
from tests.test_quant import _bars, _days
from tests.test_sim import _market

# Long enough for six quarters: releases must sit 30 days apart or `earnings.quarterly`
# rightly folds them into one.
DAYS = _days(150)
I0 = 137
D0, D1, D2 = DAYS[I0], DAYS[I0 + 1], DAYS[I0 + 2]
NAME = "AAPL"


def _at(day, hhmm: str) -> pd.Timestamp:
    return calendar.at(day, pd.Timestamp(hhmm).time())


def _mkt(bars=None):
    return _market(DAYS, info_bars=_bars(DAYS, 8) if bars is None else bars)


def _ctx(m, day, rnd: int) -> sim.RoundContext:
    r = calendar.rounds_for(day)[rnd - 1]
    return sim.RoundContext(day, rnd, r["deadline"], r["execution"], {}, sim.INITIAL_NAV, m)


def _releases(*rows) -> pd.DataFrame:
    return pd.DataFrame([{"ticker": t, "accepted": a} for t, a in rows])


# AAPL's quarters: five past releases after the close, each reacting at the next open,
# and one after the close of D0, reacting at D1's open.
PAST = [_at(DAYS[i], "16:30") for i in (22, 45, 68, 91, 114)]
EVENTS = _releases(*[(NAME, a) for a in PAST], (NAME, _at(D0, "16:30")))
assert all(b - a >= pd.Timedelta(days=30) for a, b in zip(PAST, PAST[1:] + [_at(D0, "16:30")]))


def _tags(m, events=EVENTS):
    return T.TriggerTags(T.EarningsHistory.from_market(events, m),
                         EarningsCalendar(events, m.days))


def test_an_event_public_by_the_deadline_is_always_already_in_the_next_fill():
    """Fills land at :30 and every deadline is before its execution, so nothing public
    can be traded ahead of. Tagged "upcoming", a filing seen at 10:20 or results out
    overnight would invite a sale meant to beat a reaction the fill already contains."""
    m, tags = _mkt(), T.TriggerTags()
    for day in (D0, D1):
        for r in calendar.rounds_for(day):
            ctx = _ctx(m, day, r["round"])
            for at in (r["deadline"] - pd.Timedelta(minutes=5), r["deadline"],
                       _at(day, "07:00"), _at(DAYS[DAYS.index(day) - 1], "17:45"),
                       _at(DAYS[DAYS.index(day) - 3], "12:00")):
                if at <= r["deadline"]:
                    assert tags.tag(ctx, NAME, "8k", at=at)["status"] == T.REACTED


def test_results_released_overnight_react_at_round_ones_fill_and_the_tag_says_so():
    """The v1 failure: a name sold at the open after its results had gapped it down. Round
    1 fills at the open, so the tag must say the sale comes after the reaction."""
    m = _mkt()
    t = _tags(m).earnings(_ctx(m, D1, 1), NAME)
    assert t["status"] == T.REACTED and t["minutes_traded"] == 0
    assert t["move_since_close_sigmas"] is None        # nothing has traded since the close
    assert "after it, not before" in t["note"]
    later = _tags(m).earnings(_ctx(m, D1, 3), NAME)
    assert later["status"] == T.REACTED and later["minutes_traded"] == 120
    assert later["move_since_close_sigmas"] is not None


def test_results_not_yet_public_are_upcoming_with_the_sessions_to_their_reaction():
    m = _mkt()
    for rnd in (1, 7):
        t = _tags(m).earnings(_ctx(m, D0, rnd), NAME)
        assert t["status"] == T.UPCOMING and t["sessions_to_reaction"] == 1
        assert "before it" in t["note"]
    assert _tags(m).earnings(_ctx(m, DAYS[I0 - 2], 1), NAME) is None   # 3 sessions: outside 2
    assert _tags(m).earnings(_ctx(m, DAYS[I0 - 1], 1), NAME)["sessions_to_reaction"] == 2


def test_an_event_dated_after_the_deadline_is_refused_not_tagged():
    """A caller that passed a filing accepted after the deadline would hand a decision
    the future; the tag raises rather than describe it."""
    m = _mkt()
    ctx = _ctx(m, D0, 2)
    with pytest.raises(ValueError, match="after the deadline"):
        T.TriggerTags().tag(ctx, NAME, "8k", at=ctx.deadline + pd.Timedelta(seconds=1))
    with pytest.raises(ValueError):
        T.TriggerTags().tag(ctx, NAME, "earnings")      # neither public nor scheduled


def test_a_reaction_counts_only_once_its_session_has_closed():
    """D1's reaction to the D0 release is still happening at D1's last round; counted
    then, today's outcome would be evidence for today's decision."""
    m = _mkt()
    h = T.EarningsHistory.from_market(EVENTS, m)
    during = h.past(NAME, _ctx(m, D1, 7).deadline)
    after = h.past(NAME, _ctx(m, D2, 1).deadline)
    assert len(during) == 5 and len(after) == 6
    assert after.iloc[0]["session"] == D1
    s = h.summary(NAME, _ctx(m, D2, 1).deadline)
    assert s["quarters"] == 6 and len(s["recent_pct"]) == 6


def test_past_reactions_are_the_close_to_close_move_on_each_reaction_session():
    m = _mkt()
    h = T.EarningsHistory.from_market(EVENTS, m)
    daily = T.daily_panel(m)
    row = h.past(NAME, _ctx(m, D2, 1).deadline).iloc[-1]          # the oldest: DAYS[23]
    i = [ts.date() for ts in daily.index].index(DAYS[23])
    want = daily[NAME].iloc[i] / daily[NAME].iloc[i - 1] - 1
    sd = np.log(daily[NAME]).diff().iloc[i - T.SIGMA_DAYS:i].std()
    assert row["session"] == DAYS[23]
    assert row["pct"] == pytest.approx(want)
    assert row["sigmas"] == pytest.approx(np.log(1 + want) / sd)


def _rewrite_after(bars: pd.DataFrame, deadline) -> pd.DataFrame:
    out = bars.copy()
    late = out["end"] > deadline
    out.loc[late, ["open", "high", "low", "close"]] *= 3.0
    return out


@pytest.mark.parametrize("rnd", [1, 3, 7])
def test_no_tag_changes_when_every_price_and_release_after_the_deadline_is_rewritten(rnd):
    """Each tag reads prices, past reactions and releases; one that read past the
    deadline would show a decision the move it is deciding about."""
    bars = _bars(DAYS, 8)
    ctx_day = D1
    deadline = calendar.rounds_for(ctx_day)[rnd - 1]["deadline"]
    # MSFT's lies past the announced horizon (`earnings.NEXT_KNOWN_SESSIONS`): inside it, a
    # date is public by design, as in v1's calendar.
    later = _releases((NAME, deadline + pd.Timedelta(minutes=1)), (NAME, _at(DAYS[I0 + 9], "16:30")),
                      ("MSFT", _at(DAYS[I0 + 12], "07:00")))
    assert I0 + 12 - DAYS.index(ctx_day) > T.earnings.NEXT_KNOWN_SESSIONS
    a, b = _mkt(bars), _mkt(_rewrite_after(bars, deadline))
    ta, tb = _tags(a), _tags(b, pd.concat([EVENTS, later], ignore_index=True))
    for t in (NAME, "MSFT"):
        ca, cb = _ctx(a, ctx_day, rnd), _ctx(b, ctx_day, rnd)
        assert ta.earnings(ca, t) == tb.earnings(cb, t)
        assert ta.tag(ca, t, "move") == tb.tag(cb, t, "move")
        assert ta.tag(ca, t, "8k", at=_at(D0, "12:00")) == tb.tag(cb, t, "8k", at=_at(D0, "12:00"))


def test_the_move_since_the_close_is_in_the_v1_triggers_sigmas():
    m = _mkt()
    ctx = _ctx(m, D1, 4)
    closes = m.recent_closes(pd.Timestamp("2100-01-01", tz="UTC"), 10 ** 6)
    daily = closes.groupby(closes.index.date).tail(1)
    daily = daily[[ts.date() < D1 for ts in daily.index]].tail(T.SIGMA_DAYS + 1)
    last = m.recent_closes(ctx.deadline, 1).iloc[-1][NAME]
    z = np.log(last / daily[NAME].iloc[-1]) / np.log(daily[NAME]).diff().std()
    pct, got = T.move_since_close(ctx, NAME)
    assert got == pytest.approx(z) and pct == pytest.approx(last / daily[NAME].iloc[-1] - 1)
    assert T.move_since_close(_ctx(m, D1, 1), NAME) == (None, None)


def test_a_reaction_on_a_missing_price_is_left_out_never_filled():
    """A stand-in close is a zero return that never happened: the name would look like
    one that does not move on results."""
    bars = _bars(DAYS, 8)
    react = DAYS[46]
    gone = (bars["ticker"] == NAME) & (bars["end"] == calendar.at(react, calendar.session_close(react)))
    bars.loc[gone, "close"] = np.nan
    h = T.EarningsHistory.from_market(EVENTS, _mkt(bars))
    assert react not in set(h.table["session"])
    assert len(h.table) == 5


def test_a_session_still_trading_has_no_close_in_the_daily_panel():
    """Live, today's latest bar would otherwise stand as a close, and a results day half
    traded would enter the history as a whole reaction."""
    bars = _bars(DAYS, 8)
    bars = bars[bars["end"] <= _at(DAYS[-1], "12:30")]
    daily = T.daily_panel(_mkt(bars))
    assert daily.index[-1].date() == DAYS[-2]
