"""The live book: the server's portfolio read strictly, and the paper book filled by sim's rules."""

from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from icaif import calendar, data, sim
from icaif import portfolio as P
from tests.test_sim import _market, _once, _weights

TICKERS = sorted(data.load_universe())


@pytest.mark.parametrize("positions", [
    {"AAPL": 100, "MSFT": Decimal("50.5")},
    {"AAPL": {"shares": 100, "price": 200.0}, "MSFT": {"shares": 50.5}},
    [{"symbol": "AAPL", "shares": 100, "market_value": 20000}, {"symbol": "MSFT", "quantity": 50.5}],
])
def test_each_declared_shape_reads_the_same_book(positions):
    book = P.parse({"cash": Decimal("950000.25"), "positions": positions, "nav": 1_000_000,
                    "as_of": "2026-10-08T20:00:00+00:00"})
    assert book.shares["AAPL"] == 100 and book.shares["MSFT"] == 50.5
    assert set(book.shares) == set(TICKERS) and sum(v > 0 for v in book.shares.values()) == 2
    assert book.cash == 950000.25 and not book.all_cash and book.source == "server"


def test_a_response_nested_under_portfolio_reads_the_same():
    book = P.parse({"phase": "validation", "portfolio": {"cash": 1e6, "holdings": {}}})
    assert book.all_cash and book.cash == 1e6


@pytest.mark.parametrize("response,match", [
    ({"positions": {}}, "cash"),
    ({"cash": 1e6}, "positions"),
    ({"cash": 1e6, "positions": {}, "holdings": {}}, "both"),
    ({"cash": 1e6, "positions": {"BRK.B": 10}}, "outside the 30"),
    ({"cash": 1e6, "positions": {"AAPL": -5}}, "negative"),
    ({"cash": 1e6, "positions": [{"symbol": "AAPL", "shares": 1}, {"symbol": "AAPL", "shares": 2}]}, "twice"),
    ({"cash": 1e6, "positions": [{"symbol": "AAPL", "value": 1}]}, "shares"),
    ({"cash": "lots", "positions": {}}, "number"),
    ({"cash": -100_000, "positions": {"AAPL": 5000}, "nav": 1_000_000}, "far below zero"),
])
def test_a_portfolio_the_reader_does_not_know_raises_instead_of_reading_as_cash(response, match):
    """Read as all cash, an unknown shape looks like a fresh phase, and the rule's answer
    to a fresh phase is to buy the whole entry again over the book we hold."""
    with pytest.raises(P.PortfolioFormatError, match=match):
        P.parse(response)


def test_the_fee_leaves_cash_slightly_negative_and_that_still_reads():
    """pre_fee sizing leaves cash -0.1% x notional when fully invested; rejecting that
    would refuse every fully invested book the backend can produce."""
    book = P.parse({"cash": -950.0, "positions": {"AAPL": 4000}, "nav": 999_050.0})
    assert book.cash == -950.0


def test_the_printed_shape_carries_no_value():
    shape = P.keys_only({"cash": 123.0, "team_token": "secret", "positions": [{"symbol": "AAPL", "shares": 5}]})
    assert "secret" not in str(shape) and "123" not in str(shape) and "AAPL" not in str(shape)


def test_weights_need_a_price_only_for_names_held():
    book = P.Book(cash=500_000.0, shares={**{t: 0.0 for t in TICKERS}, "AAPL": 2500.0}, source="paper")
    px = pd.Series(np.nan, index=TICKERS)
    px["AAPL"] = 200.0
    w = book.weights(px)
    assert w["AAPL"] == pytest.approx(0.5) and w.drop("AAPL").eq(0).all()
    px["AAPL"] = np.nan
    assert book.weights(px).isna().all()


# ----------------------------------------------------------------------------- paper

DAY = date(2026, 9, 21)


def _opens(ts, prices):
    return pd.DataFrame([prices], index=pd.DatetimeIndex([ts]))


def test_a_paper_fill_is_the_simulators_fill_to_the_cent():
    """A second copy of the fee or sizing rule would make the shadow's P&L differ from
    the submitted book's by our arithmetic, not by its decisions."""
    ex = calendar.rounds_for(DAY)[0]["execution"]
    pb = P.PaperBook()
    pb.order("r1", ex, _weights(AAPL=0.3, MSFT=0.2))
    assert pb.settle(_opens(ex, {t: 100.0 for t in TICKERS}), ex) == []
    ref = sim.run(_once(_weights(AAPL=0.3, MSFT=0.2)), _market([DAY]), DAY, 1)
    first = ref.ledger.iloc[0]
    assert pb.cash == pytest.approx(first["cash"], abs=1e-6)
    assert pb.shares["AAPL"] == pytest.approx(first["AAPL"]) and pb.shares["MSFT"] == pytest.approx(first["MSFT"])
    assert pb.pending == [] and pb.traded_notional == pytest.approx(500_000.0)


def test_an_order_waits_for_its_own_open_rather_than_filling_at_another_price():
    """A fill at the last close is a zero return that never happened, and the shadow's
    P&L would carry it."""
    ex = calendar.rounds_for(DAY)[0]["execution"]
    pb = P.PaperBook()
    pb.order("r1", ex, _weights(AAPL=0.3))
    later = ex + pd.Timedelta(hours=1)
    issues = pb.settle(_opens(later, {t: 100.0 for t in TICKERS}), later)
    assert pb.pending and pb.cash == sim.INITIAL_NAV and "waits" in issues[0]
    assert pb.settle(None, later) and pb.pending


def test_a_name_neither_held_nor_ordered_needs_no_price():
    """0 x NaN is NaN: one unpriced bystander would otherwise blank the whole fill."""
    ex = calendar.rounds_for(DAY)[0]["execution"]
    prices = {t: 100.0 for t in TICKERS}
    prices["XOM"] = np.nan
    pb = P.PaperBook()
    pb.order("r1", ex, _weights(AAPL=0.3))
    assert pb.settle(_opens(ex, prices), ex) == []
    assert np.isfinite(pb.cash) and pb.shares["AAPL"] == pytest.approx(3000.0)


def test_an_order_not_yet_executed_is_left_alone():
    ex = calendar.rounds_for(DAY)[0]["execution"]
    pb = P.PaperBook()
    pb.order("r1", ex, _weights(AAPL=0.3))
    assert pb.settle(_opens(ex, {t: 100.0 for t in TICKERS}), ex - pd.Timedelta(minutes=1)) == []
    assert pb.pending and pb.cash == sim.INITIAL_NAV


def test_a_paper_book_survives_its_json():
    ex = calendar.rounds_for(DAY)[0]["execution"]
    pb = P.PaperBook()
    pb.order("r1", ex, _weights(AAPL=0.3))
    pb.settle(_opens(ex, {t: 100.0 for t in TICKERS}), ex)
    again = P.PaperBook.from_json(pb.to_json())
    assert again == pb
