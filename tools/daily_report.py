"""Official's four metrics, estimated each evening and pushed to the owner's phone (ntfy).

    .venv/bin/python tools/daily_report.py            # compute and send now
    .venv/bin/python tools/daily_report.py --boot     # the VM's boot hook (see below)
    .venv/bin/python tools/daily_report.py --dry      # print, send nothing

The server keeps Official's metrics hidden until results are released (HTTP 409), and
its NAV lags the rounds by hours. So this marks the held book to market on Alpaca's
30m opens (the organizer's vendor, the prices rounds fill at) at every round's
execution time and applies the formulas of starter-kit/docs/evaluation.md: period
returns between executions, Sharpe = 42 x mean / sample sd, drawdown over the endpoints
and the daily closes, turnover as the mean of each round's notional / NAV. The server's
reported NAV is printed beside it as the cross-check.

**Valid only while the book is the one bought at round 1.** Shares are read from the
server's book now and applied to the whole phase, which is exact for a buy-and-hold
entry and wrong the moment a later round trades.

The VM wakes on a GCP instance schedule at 21:30 ET on weekdays. `--boot` reports only
inside that window, and exits 3 outside it. The boot hook powers the VM off on any
other exit code, so a VM started by hand for other work is never shut under its user.
A report that fails still sends a short failure message, never silence.
"""

import argparse
import json
import math
import sys
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

ET = ZoneInfo("America/New_York")
SCHEDULE = ROOT / "starter-kit" / "schedule.json"
B1 = 1_000_000.0
WINDOW = (dtime(21, 15), dtime(22, 30))   # ET, around the schedule's 21:30 start
OUTSIDE = 3
HALF_HOUR = timedelta(minutes=30)


def metrics(cash: float, shares: dict, opens, closes, executions, final_close=None) -> dict:
    """The four metrics of evaluation.md for a book held since the first execution.

    `opens`/`closes` are DataFrames (index: bar start, UTC; columns: tickers).
    `executions` are the phase's execution times (UTC) with a bar in `opens`. A period
    ends at the next execution; the last one ends at `final_close` when the phase has
    closed. Turnover counts round 1's notional only: the book is held, so every later
    round trades nothing.
    """
    names = [t for t, s in shares.items() if abs(s) > 1e-12]

    def value(frame, t):
        return cash + sum(shares[n] * float(frame.loc[t, n]) for n in names)

    ex = [t for t in executions if t in opens.index]
    ends = [value(opens, t) for t in ex[1:]]
    if final_close is not None and final_close in closes.index:
        ends.append(value(closes, final_close))
    k = len(ends)
    if k == 0:
        return {"periods": 0}
    starts = [B1] + ends[:-1]
    r = [e / b - 1 for e, b in zip(ends, starts)]
    mean = sum(r) / k
    sd = math.sqrt(sum((x - mean) ** 2 for x in r) / (k - 1)) if k > 1 else 0.0
    sharpe = 42 * mean / sd if sd > 0 else 0.0
    last_end = ex[k] if k < len(ex) else final_close
    # A daily close is the 15:30 bar's close, valued at 16:00: half an hour after round 7
    # fills at the same bar's open, so it sorts after it. A day still trading has none.
    day_closes = [(t + HALF_HOUR, value(closes, t)) for t in closes.index
                  if t.tz_convert(ET).time() == dtime(15, 30)]
    seq = sorted([(ex[0], B1)] + list(zip(ex[1:k + 1], ends)) + day_closes)
    peak, dd = 0.0, 0.0
    for _, v in seq:
        peak = max(peak, v)
        dd = max(dd, (peak - v) / peak)
    invested = sum(shares[n] * float(opens.loc[ex[0], n]) for n in names)
    return {"periods": k, "through": last_end, "nav": ends[-1],
            "cumulative_return": ends[-1] / B1 - 1, "sharpe": sharpe, "max_drawdown": dd,
            "turnover": (invested / B1) / k, "invested_at_entry": invested,
            "book_return": (ends[-1] - cash) / invested - 1 if invested else 0.0}


def moves(shares: dict, opens, closes, first) -> dict:
    """Each held name's move from its fill (the first execution's open) to the last close."""
    return {n: float(closes[n].dropna().iloc[-1]) / float(opens.loc[first, n]) - 1
            for n, s in shares.items() if abs(s) > 1e-12}


def message(m: dict, mv: dict, server_nav, day_no: int) -> dict:
    sign = lambda x: f"{x:+.2%}"  # noqa: E731
    pnl = m["nav"] - B1
    ranked = sorted(mv.items(), key=lambda x: -x[1])
    body = "\n".join([
        f"Through {m['through'].tz_convert(ET):%a %b %d %H:%M} ET, {m['periods']} periods (estimate)",
        f"Return    {m['cumulative_return']:+.6%}  (NAV ${m['nav']:,.2f})",
        f"Sharpe    {m['sharpe']:.2f}",
        f"Max DD    {m['max_drawdown']:.6%}",
        f"Turnover  {m['turnover']:.6f}",
        f"Book      {sign(m['book_return'])} on ${m['invested_at_entry']:,.0f} invested",
        f"Server NAV ${server_nav:,.2f}" if server_nav is not None else "Server NAV unavailable",
        "",
        "  ".join(f"{n} {sign(v)}" for n, v in ranked),
    ])
    return {"title": f"Official day {day_no}: NAV ${m['nav']:,.2f} ({pnl:+,.2f})",
            "body": body, "priority": "default", "tags": "chart_with_upwards_trend"}


def gather():
    """(cash, shares, server NAV, opens, closes, executions, final close) from the
    server's book, Alpaca's bars and the kit's schedule."""
    import httpx
    import pandas as pd

    from icaif import alpaca, net, runner
    from icaif import portfolio as P

    with runner.kit_session(readonly=True) as s:
        book = P.parse(s.portfolio("official"))
    rows = sorted((r for r in json.loads(SCHEDULE.read_text())["rounds"]
                   if r["phase"] == "official" and r.get("status") != "CANCELLED"),
                  key=lambda r: r["execution_time"])
    executions = [pd.Timestamp(r["execution_time"]).tz_convert("UTC") for r in rows]
    close = pd.Timestamp(rows[-1]["close_time"]).tz_convert("UTC") - pd.Timedelta(minutes=30)
    names = [t for t, s in book.shares.items() if abs(s) > 1e-12]
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=16)).isoformat()
    limiter = alpaca._RateLimiter()
    with httpx.Client(headers=alpaca._credentials(), verify=net.ssl_context(), timeout=60) as c:
        raw = pd.concat([alpaca._fetch_symbol(c, limiter, n, executions[0].isoformat(), end)
                         for n in names])
    raw["t"] = pd.to_datetime(raw["t"], utc=True)
    local = raw["t"].dt.tz_convert(ET).dt.time
    raw = raw[(local >= dtime(9, 30)) & (local < dtime(16, 0))]   # regular session only
    opens = raw.pivot(index="t", columns="ticker", values="open")
    closes = raw.pivot(index="t", columns="ticker", values="close")
    return (book.cash, dict(book.shares), book.reported_nav, opens, closes, executions,
            close if close in closes.index else None)


def main(argv=None) -> int:
    from alert_watch import send, setting

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--boot", action="store_true", help="the VM's boot hook: report only in the window")
    p.add_argument("--dry", action="store_true", help="print the report, send nothing")
    args = p.parse_args(argv)
    now = datetime.now(timezone.utc).astimezone(ET)
    if args.boot and not (now.weekday() < 5 and WINDOW[0] <= now.time() <= WINDOW[1]):
        print(f"{now:%F %T %Z}: outside the report window; nothing sent, VM left running")
        return OUTSIDE
    topic, email = setting("NTFY_TOPIC"), setting("NTFY_EMAIL")
    def post(m):
        """The push first, then the email on its own: ntfy.sh refuses anonymous email
        (HTTP 400, 2026-10-16), and sent together a refused email lost the push too."""
        if args.dry:
            print(m["title"], m["body"], sep="\n")
            return
        send(m, topic)
        if email:
            try:
                send(m, topic, email=email)
            except Exception as err:  # noqa: BLE001 - the push already went
                print(f"{now:%F %T %Z}: email not sent: {type(err).__name__}: {err}")
    try:
        cash, shares, server_nav, opens, closes, executions, final = gather()
        m = metrics(cash, shares, opens, closes, executions, final)
        if not m["periods"]:
            raise RuntimeError("no completed period yet")
        days = len({t.tz_convert(ET).date() for t in executions if t <= m["through"]})
        post(message(m, moves(shares, opens, closes, executions[0]), server_nav, days))
    except Exception as err:  # noqa: BLE001 - the owner hears of a failure, never silence
        post({"title": "Official daily report failed", "priority": "high", "tags": "warning",
              "body": f"{type(err).__name__}: {str(err)[:300]}"})
        print(f"{now:%F %T %Z}: report failed: {type(err).__name__}: {err}")
        return 1
    print(f"{now:%F %T %Z}: report sent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
