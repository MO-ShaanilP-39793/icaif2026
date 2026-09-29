"""Macro readings for the agents: volatility, the market, rates, sectors, the Fed.

Everything is as of the close *before* the decision day: the context series are daily
closes, and a decision at 09:10 cannot know today's. `readings` takes the rows dated
strictly before the day, and a test rewrites every later row and requires the same
readings.

**Anonymised mode drops levels.** A replay hides names and dates so the model can't
recall what happened next. A 10-year yield of 4.9% with VIX at 13 dates a window to
late 2023 just as surely as a date would. So levels (VIX, yields, the curve) are
replaced by where they sit in their own trailing year (z-scores) and by changes, which
say "rates rose sharply" without saying "October 2023".

**FOMC decisions** come from the Fed's calendar page, which covers the current year
and the five before it. A day outside the covered years reads `null` ("unknown"),
never "no meeting": absence of a record is not absence of an event. Statements are
released at 14:00 ET, between rounds 5 and 6.
"""

import re
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from icaif import calendar, data

FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
EXTERNAL = data.ROOT / "data" / "external"
SECTORS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
YEAR = 252


def wide(context: pd.DataFrame) -> pd.DataFrame:
    """date x symbol closes from the long context frame (`external.load("yahoo_daily_context")`).

    Kept on SPY's sessions only: Yahoo prints ^VIX on some market holidays, and a union
    of dates would put a NaN SPY row on each (the bug e7e1dc8 fixed in the features).
    """
    w = context.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    w.index = pd.DatetimeIndex(w.index).normalize()
    return w[w["SPY"].notna()].sort_index()


def _z(s: pd.Series) -> Optional[float]:
    s = s.dropna().tail(YEAR)
    if len(s) < 60 or s.std() == 0:
        return None
    return round(float((s.iloc[-1] - s.mean()) / s.std()), 2)


def _r(x, nd=4):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def readings(w: pd.DataFrame, day: date, anonymize: bool = False) -> dict:
    """The macro block of an observation, from sessions strictly before `day`."""
    h = w[w.index < pd.Timestamp(day)]
    if len(h) < 30:
        return {}
    spy = h["SPY"]
    lr = np.log(spy).diff()
    vix = h["^VIX"].dropna()
    t3m, t10y, t5y = h["^IRX"].dropna(), h["^TNX"].dropna(), h["^FVX"].dropna()
    curve = (h["^TNX"] - h["^IRX"]).dropna()
    out = {
        "spy_ret_1d": _r(lr.iloc[-1]),
        "spy_ret_5d": _r(lr.tail(5).sum()),
        "spy_ret_20d": _r(lr.tail(20).sum()),
        "spy_vol_20d_ann": _r(lr.tail(20).std() * np.sqrt(YEAR)),
        "spy_drawdown_from_1y_high": _r(1 - spy.iloc[-1] / spy.tail(YEAR).max()),
        "vix_z_1y": _z(vix),
        "vix_chg_5d_pct": _r(vix.iloc[-1] / vix.iloc[-6] - 1 if len(vix) > 6 else np.nan, 3),
        "yield_10y_chg_20d_bp": _r((t10y.iloc[-1] - t10y.iloc[-21]) * 100 if len(t10y) > 21 else np.nan, 1),
        "yield_3m_chg_20d_bp": _r((t3m.iloc[-1] - t3m.iloc[-21]) * 100 if len(t3m) > 21 else np.nan, 1),
        "yield_10y_z_1y": _z(t10y),
        "curve_10y_3m_chg_20d_bp": _r((curve.iloc[-1] - curve.iloc[-21]) * 100 if len(curve) > 21 else np.nan, 1),
        "sector_ret_20d_vs_spy": {
            s: _r(np.log(h[s]).diff().tail(20).sum() - lr.tail(20).sum())
            for s in SECTORS if s in h and h[s].tail(21).notna().all()},
    }
    if not anonymize:
        out.update({"vix": _r(vix.iloc[-1], 2), "yield_3m_pct": _r(t3m.iloc[-1], 3),
                    "yield_5y_pct": _r(t5y.iloc[-1], 3), "yield_10y_pct": _r(t10y.iloc[-1], 3),
                    "curve_10y_3m_pct": _r(curve.iloc[-1], 3)})
    return out


# ----------------------------------------------------------------------------- the Fed

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
                                      "sep", "oct", "nov", "dec"], start=1)}


def parse_fomc(html: str) -> tuple[pd.DataFrame, list[int]]:
    """(decision dates with a projections flag, years the page covers).

    A meeting reads "27-28" under its month, "30-1" under "April/May", or "17-18*" when
    it carries the Summary of Economic Projections. The decision is the last day. Rows
    marked "(notation vote)" or "(unscheduled)" are not scheduled decisions, so a
    calendar built on them would flag days no one knew about in advance.
    """
    rows, years = [], []
    panels = re.split(r"(\d{4}) FOMC Meetings", html)
    for i in range(1, len(panels) - 1, 2):
        year, body = int(panels[i]), panels[i + 1]
        years.append(year)
        # No trailing context in the pattern: a greedy tail swallowed the next meeting
        # whenever entries were short, dropping every other decision without error.
        for month, days in re.findall(
                r'fomc-meeting__month[^>]*><strong>([^<]+)</strong></div>\s*'
                r'<div class="fomc-meeting__date[^>]*>([^<]+)</div>', body):
            if "notation" in days.lower() or "unscheduled" in days.lower():
                continue
            m = re.match(r"\s*(\d+)(?:-(\d+))?(\*?)", days)
            if not m:
                continue
            last_day = int(m.group(2) or m.group(1))
            # Month labels come full ("January") or short and paired ("Jan/Feb"); a range
            # that wraps ("30-1") ends in the second month.
            names = [x.strip().lower()[:3] for x in month.split("/")]
            wraps = bool(m.group(2)) and int(m.group(2)) < int(m.group(1))
            mon = MONTHS[names[-1] if wraps else names[0]]
            rows.append({"date": pd.Timestamp(year, mon, last_day), "projections": bool(m.group(3))})
    out = pd.DataFrame(rows, columns=["date", "projections"]).drop_duplicates("date")
    return out.sort_values("date").reset_index(drop=True), sorted(set(years))


def fetch_fomc() -> tuple[pd.DataFrame, list[int]]:
    import httpx

    from icaif import net

    with httpx.Client(verify=net.ssl_context(), timeout=30, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0"}) as c:
        html = c.get(FOMC_URL).raise_for_status().text
    dates, years = parse_fomc(html)
    if dates.empty:
        raise RuntimeError("the Fed calendar parsed to no meetings; the page layout changed")
    return dates, years


class FomcCalendar:
    """Sessions until the next scheduled FOMC decision, or None where not covered."""

    def __init__(self, decisions: pd.DataFrame, years: list[int]):
        self.dates = sorted(pd.to_datetime(decisions["date"]).dt.date)
        self.sep = {d for d, p in zip(pd.to_datetime(decisions["date"]).dt.date,
                                      decisions["projections"]) if p}
        self.years = set(years)

    @classmethod
    def load(cls, directory=EXTERNAL) -> Optional["FomcCalendar"]:
        """The latest dated snapshot, or None when none has been fetched."""
        import json

        files = sorted(directory.glob("fomc_decisions_*.json"))
        if not files:
            return None
        blob = json.loads(files[-1].read_text())
        return cls(pd.DataFrame(blob["decisions"]), blob["years"])

    @staticmethod
    def save(decisions: pd.DataFrame, years: list[int], directory=EXTERNAL) -> str:
        """A dated JSON snapshot; the covered years travel with the dates, since a
        date outside them must read "unknown", not "no meeting"."""
        import json

        path = directory / f"fomc_decisions_{pd.Timestamp.now(tz=calendar.TZ):%Y-%m-%d}.json"
        rows = [{"date": str(d.date()), "projections": bool(p)}
                for d, p in zip(pd.to_datetime(decisions["date"]), decisions["projections"])]
        path.write_text(json.dumps({"source": FOMC_URL, "years": years, "decisions": rows},
                                   indent=1))
        return str(path)

    def block(self, day: date, sessions: list) -> dict:
        """{"fomc_decision_today": bool | None, "sessions_to_next_fomc": int | None}."""
        if day.year not in self.years:
            return {"fomc_decision_today": None, "sessions_to_next_fomc": None}
        ahead = [d for d in sessions if d >= day]
        nxt = next((d for d in self.dates if d >= day), None)
        n = ahead.index(nxt) if nxt in ahead else None
        return {"fomc_decision_today": nxt == day,
                "fomc_statement_time_et": "14:00" if nxt == day else None,
                "sessions_to_next_fomc": n,
                "next_fomc_has_projections": (nxt in self.sep) if nxt else None}
