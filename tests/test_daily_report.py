"""The daily Official report (tools/daily_report.py): the organizer's formulas on a held book."""

import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import daily_report as D  # noqa: E402

ET = "America/New_York"


def _bars(day, prices_by_bar):
    """30m bars from 09:30 ET; prices_by_bar[i] = (open, close) for one ticker 'A'."""
    t0 = pd.Timestamp(f"{day} 09:30", tz=ET).tz_convert("UTC")
    idx = [t0 + pd.Timedelta(minutes=30 * i) for i in range(len(prices_by_bar))]
    return (pd.DataFrame({"A": [o for o, _ in prices_by_bar]}, index=idx),
            pd.DataFrame({"A": [c for _, c in prices_by_bar]}, index=idx))


def _day(prices):
    """A full session (13 bars), executions at 09:30, 10:30 ... 15:30 (bars 0, 2 ... 12)."""
    opens, closes = _bars("2026-10-12", prices)
    return opens, closes, list(opens.index[::2])


def test_a_book_of_cash_scores_zero_on_every_metric():
    """Zero volatility is a Sharpe of 0 by the rules, not a division error, and an all-cash
    book has no return, drawdown or turnover."""
    opens, closes, ex = _day([(100.0, 100.0)] * 13)
    m = D.metrics(B := 1_000_000.0, {"A": 0.0}, opens, closes, ex)
    assert m["periods"] == 6 and m["cumulative_return"] == 0 and m["sharpe"] == 0
    assert m["max_drawdown"] == 0 and m["turnover"] == 0 and m["nav"] == B


def test_period_returns_run_from_one_execution_to_the_next_with_the_fee_in_the_first():
    """B_1 is $1M before the fee, so the fee lands in period 1's return (evaluation.md).
    Starting from the post-fee NAV would drop the fee from the score entirely."""
    prices = [(100.0, 100.0)] * 13
    for i, p in zip(range(2, 13, 2), (110, 99, 110, 121, 110, 121)):
        prices[i] = (float(p), float(p))
    opens, closes, ex = _day(prices)
    cash, shares = 998_999.0, {"A": 10.0}   # $1,000 of A at 100, $1 fee
    m = D.metrics(cash, shares, opens, closes, ex)
    navs = [cash + 10 * p for p in (110, 99, 110, 121, 110, 121)]
    r = [e / b - 1 for e, b in zip(navs, [1_000_000.0] + navs[:-1])]
    mean, sd = sum(r) / 6, math.sqrt(sum((x - sum(r) / 6) ** 2 for x in r) / 5)
    assert m["periods"] == 6 and m["sharpe"] == pytest.approx(42 * mean / sd)
    assert m["cumulative_return"] == pytest.approx(navs[-1] / 1e6 - 1)
    assert m["turnover"] == pytest.approx((1000 / 1e6) / 6)


def test_a_dip_at_the_daily_close_counts_toward_drawdown_though_no_period_ends_there():
    """The close is a drawdown observation inside round 7's overnight period; measured on
    period endpoints alone, a crash into the close would never show."""
    prices = [(100.0, 100.0)] * 13
    prices[12] = (100.0, 50.0)          # round 7 fills at 100, the close is 50
    opens, closes, ex = _day(prices)
    m = D.metrics(999_000.0, {"A": 10.0}, opens, closes, ex)
    assert m["max_drawdown"] == pytest.approx(500 / 1_000_000)


def test_the_last_period_ends_at_the_phases_close_once_it_has_closed():
    prices = [(100.0, 100.0)] * 12 + [(100.0, 120.0)]
    opens, closes, ex = _day(prices)
    final = closes.index[-1]
    m = D.metrics(999_000.0, {"A": 10.0}, opens, closes, ex, final_close=final)
    assert m["periods"] == 7 and m["nav"] == pytest.approx(999_000 + 1200)


def test_the_boot_hook_leaves_a_vm_started_by_hand_running(monkeypatch):
    """The hook powers the VM off unless this exits 3. Started at noon for other work, a
    VM that reported and shut down would cut off whoever started it."""
    class Noon(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 19, 16, 0, tzinfo=tz)   # 12:00 ET

    monkeypatch.setattr(D, "datetime", Noon)
    monkeypatch.setattr(D, "gather", lambda: pytest.fail("reported outside the window"))
    assert D.main(["--boot"]) == D.OUTSIDE


def test_a_failed_report_is_sent_as_a_failure_not_silence(monkeypatch):
    sent = []
    import alert_watch

    def broken():
        raise ConnectionError("Alpaca unreachable")

    monkeypatch.setattr(D, "gather", broken)
    monkeypatch.setattr(alert_watch, "send", lambda m, topic, email=None: sent.append(m))
    monkeypatch.setattr(alert_watch, "setting", lambda name: "x")
    assert D.main([]) == 1
    assert sent[0]["title"] == "Official daily report failed" and "Alpaca unreachable" in sent[0]["body"]


def test_a_refused_email_never_costs_the_push(monkeypatch):
    """ntfy.sh refuses anonymous email with HTTP 400. Sent in one request with the push,
    the refusal took the push down with it, and the owner heard nothing that evening."""
    import alert_watch

    pushes = []

    def send(m, topic, email=None):
        if email:
            raise OSError("HTTP 400: anonymous email sending is not allowed")
        pushes.append(m)

    monkeypatch.setattr(D, "gather", lambda: (_ for _ in ()).throw(ConnectionError("x")))
    monkeypatch.setattr(alert_watch, "send", send)
    monkeypatch.setattr(alert_watch, "setting", lambda name: "someone@example.com")
    assert D.main([]) == 1 and len(pushes) == 1
