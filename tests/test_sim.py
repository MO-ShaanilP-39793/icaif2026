import json
from datetime import date

import pandas as pd
import pytest

from icaif import calendar, data, kit, sim

TICKERS = sorted(data.load_universe())
DAY1, DAY2 = date(2026, 9, 21), date(2026, 9, 22)


def _market(days, exec_px=None, close_px=None, info_bars=None):
    """Every ticker at 100 unless overridden: {(day, round): {ticker: px}}."""
    idx = [r["execution"] for d in days for r in calendar.rounds_for(d)]
    ex = pd.DataFrame(100.0, index=idx, columns=TICKERS)
    for (d, rnd), px in (exec_px or {}).items():
        ts = calendar.rounds_for(d)[rnd - 1]["execution"]
        for t, p in px.items():
            ex.loc[ts, t] = p
    cidx = [calendar.at(d, calendar.session_close(d)) for d in days]
    cl = pd.DataFrame(100.0, index=cidx, columns=TICKERS)
    for d, px in (close_px or {}).items():
        for t, p in px.items():
            cl.loc[calendar.at(d, calendar.session_close(d)), t] = p
    if info_bars is None:
        info_bars = pd.DataFrame({"end": pd.Series([], dtype=f"datetime64[ns, {calendar.TZ}]")})
    return sim.Market(ex, cl, info_bars)


def _weights(**kw):
    return {t: kw.get(t, 0) for t in TICKERS}


def _once(weights):
    """Trade `weights` in round 1 of the first day, then hold."""
    def strategy(ctx):
        return weights if (ctx.day == DAY1 and ctx.round == 1) else None
    return strategy


def test_one_trade_then_holding_matches_the_ledger_worked_by_hand():
    """Buy 30% AAPL at 100, mark at 110 through the day, close at 120.

    By hand: 3,000 shares, notional 300,000, fee 300, cash 699,700. Every later round
    is 699,700 + 3,000 x 110 = 1,029,700; the close is 699,700 + 360,000 = 1,059,700.
    """
    later = {(DAY1, r): {"AAPL": 110.0} for r in range(2, 8)}
    m = _market([DAY1], exec_px=later, close_px={DAY1: {"AAPL": 120.0}})
    res = sim.run(_once(_weights(AAPL=0.3)), m, DAY1, 1)

    assert [p["nav_before"] for p in res.periods] == [1_000_000.0] + [1_029_700.0] * 6
    assert res.periods[-1]["nav_after_period"] == pytest.approx(1_059_700.0)
    assert res.periods[0]["traded_notional"] == 300_000.0
    assert all(p["traded_notional"] == 0 for p in res.periods[1:])
    got = res.metrics()
    assert got["cumulative_return"] == pytest.approx(0.0597)
    assert got["turnover"] == pytest.approx(0.3 / 7)
    assert got["maximum_drawdown"] == 0


def test_the_fee_lands_in_the_trading_rounds_own_return():
    """Round 1 buys at 100 and nothing moves, so its whole return is the 0.1% fee on
    300,000 of notional: -300 on 1,000,000."""
    res = sim.run(_once(_weights(AAPL=0.3)), _market([DAY1]), DAY1, 1)
    first = res.periods[0]
    assert first["nav_after_period"] / first["nav_before"] - 1 == pytest.approx(-0.0003)


def test_a_float_weight_just_over_the_cap_holds_the_round_instead_of_trading():
    """0.1 + 0.2 is 0.30000000000000004 in binary float, which the backend's Decimal
    check rejects as over 0.30. The round is silently held live; here it is recorded."""
    res = sim.run(_once(_weights(AAPL=0.1 + 0.2)), _market([DAY1]), DAY1, 1)
    assert res.periods[0]["held"] and res.periods[0]["traded_notional"] == 0
    assert len(res.invalid_rounds) == 1 and "0.30" in res.invalid_rounds[0]["reason"]


def test_an_overnight_dip_at_the_close_counts_for_drawdown_but_not_as_a_return():
    """Round 7's period runs to the next 09:30, so a crash at 16:00 that recovers by the
    open adds no period return. The rules still make that close a drawdown point.

    By hand: fully invested at pre-fee sizing, cash is -1,000; at the close the stock is
    worth 500,000, so NAV is 499,000 against the 1,000,000 starting peak: 0.501."""
    m = _market([DAY1, DAY2], close_px={DAY1: {t: 50.0 for t in TICKERS}})
    res = sim.run(_once({t: 1 / 30 for t in TICKERS}), m, DAY1, 2)
    r7 = res.periods[6]
    assert r7["nav_after_period"] == pytest.approx(r7["nav_before"])
    assert res.metrics()["maximum_drawdown"] == pytest.approx(0.501)


def test_pre_fee_sizing_overdraws_cash_by_the_fee_and_post_fee_does_not():
    """Fully invested at pre-fee sizing leaves cash at minus 0.1% of notional. Which one
    the backend does is unconfirmed until Validation receipts show share counts."""
    w = {t: 1 / 30 for t in TICKERS}
    pre = sim.run(_once(w), _market([DAY1]), DAY1, 1, sizing="pre_fee")
    post = sim.run(_once(w), _market([DAY1]), DAY1, 1, sizing="post_fee")
    assert pre.ledger["cash"].iloc[0] == pytest.approx(-1_000.0, abs=1e-6)
    assert post.ledger["cash"].iloc[0] == pytest.approx(0.0, abs=0.01)


def test_a_bar_that_ends_after_the_deadline_is_invisible_to_the_decision():
    """The 10:00-11:00 organizer bar starts before round 3's 11:25 deadline but is
    complete by then; the 11:00-12:00 bar starts before it and is not."""
    s = [pd.Timestamp(f"2026-09-21 {h}:00", tz=calendar.TZ) for h in (10, 11)]
    info = pd.DataFrame({"start": s, "end": [x + pd.Timedelta(hours=1) for x in s]})
    m = _market([DAY1], info_bars=info)
    seen = {}

    def strategy(ctx):
        seen[ctx.round] = list(ctx.history()["start"])
        return None

    sim.run(strategy, m, DAY1, 1)
    assert seen[2] == []
    assert seen[3] == [s[0]]
    assert seen[4] == s


def test_a_half_day_window_runs_four_rounds_and_ends_at_the_early_close():
    half = date(2025, 11, 28)
    m = _market([half], close_px={half: {"AAPL": 120.0}})
    res = sim.run(_once_on(half, _weights(AAPL=0.3)), m, half, 1)
    assert len(res.periods) == 4
    assert res.valuation_times[-1] == calendar.at(half, calendar.EARLY_CLOSE)


def _once_on(day, weights):
    return lambda ctx: weights if (ctx.day == day and ctx.round == 1) else None


def test_the_kit_calculator_reproduces_its_own_published_example():
    root = data.ROOT / "starter-kit" / "examples"
    ex = json.loads((root / "evaluation.json").read_text())
    want = json.loads((root / "evaluation_expected.json").read_text())["metrics"]
    got = kit.metrics(ex["periods"], ex["valuation_points"], float(ex["initial_nav"]))
    assert got == {k: pytest.approx(float(v)) for k, v in want.items()}


def _public_bars(days, skip=()):
    """Yahoo-shaped 60m bars at 100 for every ticker; `skip` = {(ticker, 'HH:MM', day)}."""
    rows = []
    for d in days:
        for r in calendar.rounds_for(d):
            s = r["execution"]
            for t in TICKERS:
                if (t, s.strftime("%H:%M"), d) in skip:
                    continue
                rows.append({"ticker": t, "start": s, "open": 100.0, "high": 100.0,
                             "low": 100.0, "close": 100.0, "volume": 1.0})
    bars = pd.DataFrame(rows)
    from icaif import public_bars
    bars["end"] = public_bars._end(bars["start"], 60)
    return bars


def test_a_missing_public_bar_is_patched_but_its_day_is_marked_degraded():
    """A stand-in price keeps the ledger valued, but it is a zero return that never
    happened; a window scored across it would look calmer than the market was."""
    days = [DAY1, DAY2]
    bars = _public_bars(days, skip={("AAPL", "12:30", DAY2)})
    m = sim.market_from_public_60m(bars, pd.DataFrame({"end": bars["end"]}))
    assert m.issues["degraded_days"] == [str(DAY2)]
    assert m.issues["stand_in_prices"] == 1


def test_days_before_a_ticker_first_trades_are_dropped_not_backfilled():
    """TMO's public 60m history starts a week after everyone else's. There is no earlier
    price to stand in, and a backfilled one would invent that week."""
    days = [DAY1, DAY2]
    skip = {("TMO", r["execution"].strftime("%H:%M"), DAY1) for r in calendar.rounds_for(DAY1)}
    bars = _public_bars(days, skip=skip)
    m = sim.market_from_public_60m(bars, pd.DataFrame({"end": bars["end"]}))
    assert m.days == [DAY2]
    assert m.issues["dropped_leading_days"] == [str(DAY1)]
