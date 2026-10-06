"""Trigger tags (desk v2): what code says about an event before any role reads it.

The v1 desks were told an event had fired and left to work out the rest. Two failures
followed. The Event analyst sold names that had already gapped down, which locks in the
fall and pays the fee for it. And the free desk sold every name before its results, 16
of them on Jan 21 - Feb 10, 2026, never asking how far that name usually moves on
results. So every trigger now carries three numbers, computed here, not estimated by a
role:

- **status**: "upcoming" or "already_reacted", against the fill this round's decision
  would get, not against the clock. Fills land at :30 and every deadline is before its
  execution, so an event public by the deadline is always already in the next fill's
  price, overnight results included: round 1 fills at the open, which is the reaction.
  Only a scheduled event not yet public can be traded ahead of. `minutes_traded` says
  how long the session had traded on a public event before the fill; 0 means the
  reaction happens at the fill itself, so selling "on the news" sells after it.
- **the move since the last close**, in percent and in daily sigmas (the standard
  deviation of the last `SIGMA_DAYS` daily log returns, the v1 trigger's own unit).
  None when no bar has ended since that close: before the open nothing has traded, and
  a 0.0 would read as a quiet name.
- **the name's past earnings reactions** (`EarningsHistory`): close-to-close moves on
  each reaction session of the last `PAST_REACTIONS` quarters, from our EDGAR item-2.02
  releases. A reaction counts only once its session's official close has passed the
  deadline. Before that, the move is still happening, and counting it would put today's
  outcome into the evidence for today's decision.

Point in time: prices come through `Market.recent_closes` and `qs.daily_closes`, both
cut at the deadline. A tag refuses an event dated after the deadline instead of tagging
it. The one forward-looking input is `EarningsCalendar`'s next reaction, read within
`earnings.NEXT_KNOWN_SESSIONS` as v1 does: companies announce the date weeks ahead.
"""

from bisect import bisect_left
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from icaif import calendar, earnings, quant_strategies as qs

SIGMA_DAYS = 20        # = the v1 trigger's window, so a "3-sigma move" means one thing
PAST_REACTIONS = 8     # two years of quarters
UPCOMING, REACTED = "upcoming", "already_reacted"
KINDS = ("earnings", "8k", "move")


def _r(x, nd: int) -> Optional[float]:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def daily_panel(market) -> pd.DataFrame:
    """Each session's official close (rows = the close's timestamp, columns = tickers).

    Only a day whose last bar ends at its official close has one. A live market holds a
    session still trading, and its latest bar would otherwise stand as that day's close:
    a results day half traded would enter the history as a full reaction.
    """
    bars = market.recent_closes(pd.Timestamp("2100-01-01", tz="UTC"), len(market.info_bars))
    last = bars.groupby(bars.index.date).tail(1)
    keep = [ts == calendar.at(ts.date(), calendar.session_close(ts.date())) for ts in last.index]
    return last[keep]


class EarningsHistory:
    """Each name's earnings releases and the reactions to them, each known from its close.

    `events`: (ticker, accepted) item-2.02 8-Ks, as `earnings.fetch` returns. `daily`:
    `daily_panel(market)`. A reaction whose own close or prior close is missing, or whose
    prior `SIGMA_DAYS` returns are not all there, is left out: a stand-in price would be a
    move that never happened.

    **One release a quarter, chosen from the past only.** Item 2.02 also carries previews
    (Tesla's deliveries), and `earnings.quarterly` keeps the last filing of each 30-day
    cluster. Applied to the whole history, a filing after the deadline would join a
    cluster and drop a reaction the decision had already seen, or hide a release already
    public. So every filing is kept here and clustered at query time, among those known
    by then.
    """

    def __init__(self, events: pd.DataFrame, daily: pd.DataFrame, sigma_days: int = SIGMA_DAYS):
        self.releases = (events[["ticker", "accepted"]].drop_duplicates()
                         .sort_values(["ticker", "accepted"]).reset_index(drop=True))
        sessions = pd.DatetimeIndex([pd.Timestamp(ts.date()) for ts in daily.index])
        react = earnings.reaction_session(self.releases["accepted"], sessions)
        logret = np.log(daily).diff()
        rows = []
        for t, acc, r in zip(self.releases["ticker"], self.releases["accepted"], react):
            if pd.isna(r) or t not in daily.columns:
                continue
            i = sessions.get_loc(r)
            if i < sigma_days + 1:
                continue
            prev, now = daily[t].iloc[i - 1], daily[t].iloc[i]
            hist = logret[t].iloc[i - sigma_days:i]
            if not (np.isfinite(prev) and np.isfinite(now)) or hist.isna().any():
                continue
            sd = float(hist.std())
            rows.append({"ticker": t, "accepted": acc, "session": r.date(),
                         "known_at": daily.index[i], "pct": now / prev - 1.0,
                         "sigmas": float(np.log(now / prev)) / sd if sd > 0 else np.nan})
        cols = ["ticker", "accepted", "session", "known_at", "pct", "sigmas"]
        self.table = pd.DataFrame(rows, columns=cols).sort_values("known_at").reset_index(drop=True)

    @classmethod
    def from_market(cls, events: pd.DataFrame, market) -> "EarningsHistory":
        return cls(events[events["ticker"].isin(market.tickers)], daily_panel(market))

    def past(self, ticker: str, as_of: pd.Timestamp, n: int = PAST_REACTIONS) -> pd.DataFrame:
        """The name's last `n` reactions whose session had closed by `as_of`, newest first."""
        t = self.table
        rows = t[(t["ticker"] == ticker) & (t["known_at"] <= pd.Timestamp(as_of))]
        if len(rows):
            rows = earnings.quarterly(rows)
        return rows.iloc[::-1].head(n)

    def summary(self, ticker: str, as_of: pd.Timestamp) -> Optional[dict]:
        p = self.past(ticker, as_of)
        if not len(p):
            return None
        return {"quarters": int(len(p)),
                "median_abs_pct": _r(100 * p["pct"].abs().median(), 2),
                "median_abs_sigmas": _r(p["sigmas"].abs().median(), 1),
                "largest_abs_pct": _r(100 * p["pct"].abs().max(), 2),
                "recent_pct": [_r(100 * x, 1) for x in p["pct"]]}

    def release(self, ticker: str, as_of: pd.Timestamp) -> Optional[pd.Timestamp]:
        """The name's latest release public by `as_of`, or None."""
        r = self.releases
        acc = r.loc[(r["ticker"] == ticker) & (r["accepted"] <= pd.Timestamp(as_of)), "accepted"]
        return None if acc.empty else pd.Timestamp(acc.max())


def priced_from(at: pd.Timestamp, days: list[date]) -> Optional[pd.Timestamp]:
    """The first moment a regular session can trade on news public at `at`: `at` inside a
    session, else the next session's open; None if no session we know of follows."""
    at = pd.Timestamp(at).tz_convert(calendar.TZ)
    d = at.date()
    i = bisect_left(days, d)
    if i < len(days) and days[i] == d:
        if at < calendar.at(d, calendar.SESSION_OPEN):
            return calendar.at(d, calendar.SESSION_OPEN)
        if at < calendar.at(d, calendar.session_close(d)):
            return at
        i += 1
    return calendar.at(days[i], calendar.SESSION_OPEN) if i < len(days) else None


def move_since_close(ctx, ticker: str, sigma_days: int = SIGMA_DAYS) -> tuple:
    """(fraction, daily sigmas) from the last close to the latest bar by the deadline, or
    (None, None) when no bar has ended since that close or a price is missing."""
    closes = qs.daily_closes(ctx, sigma_days + 1)
    bars = ctx.recent_closes(1)
    if len(closes) <= sigma_days or not len(bars) or bars.index[-1] <= closes.index[-1]:
        return None, None
    last, prev = float(bars.iloc[-1][ticker]), float(closes.iloc[-1][ticker])
    sd = float(np.log(closes[ticker]).diff().std())
    if not (np.isfinite(last) and np.isfinite(prev) and np.isfinite(sd)) or sd <= 0:
        return None, None
    return last / prev - 1.0, float(np.log(last / prev)) / sd


class TriggerTags:
    """Tags events for a round. `history` and `calendar_` are optional: without them a tag
    says it has no earnings evidence rather than inventing any."""

    def __init__(self, history: Optional[EarningsHistory] = None, calendar_=None):
        self.history, self.calendar = history, calendar_

    def tag(self, ctx, ticker: str, kind: str, at: Optional[pd.Timestamp] = None,
            sessions_to: Optional[int] = None) -> dict:
        """One event's tag. `at`: when it became public (a filing's acceptance, a release);
        `sessions_to`: sessions to a scheduled reaction not yet public. A move has neither:
        it is a reaction by definition."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
        out = {"kind": kind}
        if at is not None:
            at = pd.Timestamp(at)
            if at > pd.Timestamp(ctx.deadline):
                raise ValueError(f"{ticker} {kind} at {at} is after the deadline {ctx.deadline}: "
                                 f"no decision may be tagged with an event it could not see")
            p = priced_from(at, ctx.market.days)
            reacted = p is not None and p <= pd.Timestamp(ctx.execution)
            out["status"] = REACTED if reacted else UPCOMING
            out["hours_since_public"] = _r((pd.Timestamp(ctx.deadline) - at) / pd.Timedelta(hours=1), 1)
            if reacted:
                out["minutes_traded"] = int((pd.Timestamp(ctx.execution) - p) / pd.Timedelta(minutes=1))
        elif sessions_to is not None:
            if sessions_to < 1:
                raise ValueError("a scheduled reaction is at least one session away")
            out["status"] = UPCOMING
            out["sessions_to_reaction"] = int(sessions_to)
        elif kind == "move":
            out["status"] = REACTED
        else:
            raise ValueError(f"a {kind} tag needs when it became public or when it is due")
        pct, z = move_since_close(ctx, ticker)
        out["move_since_close_pct"] = _r(100 * pct, 2) if pct is not None else None
        out["move_since_close_sigmas"] = _r(z, 1)
        out["past_earnings_reactions"] = (self.history.summary(ticker, ctx.deadline)
                                          if self.history is not None else None)
        out["note"] = _note(out)
        return out

    def earnings(self, ctx, ticker: str, horizon: int = 2) -> Optional[dict]:
        """The name's earnings tag now, or None: a release public by the deadline whose
        reaction session is today, else a reaction due within `horizon` sessions."""
        if self.history is not None:
            acc = self.history.release(ticker, ctx.deadline)
            if acc is not None:
                p = priced_from(acc, ctx.market.days)
                if p is not None and p.date() == ctx.day:
                    return self.tag(ctx, ticker, "earnings", at=acc)
        if self.calendar is not None:
            n = self.calendar.to_next(ctx.day).get(ticker)
            if n is not None and n <= horizon:
                return self.tag(ctx, ticker, "earnings", sessions_to=n)
        return None


def _note(t: dict) -> str:
    """One sentence for the role, so the number is read the way it was meant."""
    if t["status"] == UPCOMING:
        n = t.get("sessions_to_reaction")
        when = "at the next session's open" if n == 1 else f"in {n} sessions" if n else "later"
        return (f"not yet public: the reaction comes {when}, after this round's fill, so a "
                f"trade now is made before it")
    if t["kind"] == "move":
        return "a price move: the next fill is at the moved price"
    if t.get("minutes_traded") == 0:
        return ("public before this session opened: this round's fill is the open, which "
                "prices the reaction, so a trade now is made after it, not before")
    return (f"public and traded on for {t['minutes_traded']} minutes before this round's fill, "
            f"which prices what the market has made of it so far")
