import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import build_holdout_space as bhs  # noqa: E402

from icaif import calendar, data, holdout, sim, suites  # noqa: E402

TICKERS = sorted(data.load_universe())
# A fixed window (days 0-2), two days no suite needs, then a rolling stretch (days 5-7).
DAYS = [date(2026, 11, 16), date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 19),
        date(2026, 11, 20), date(2026, 11, 23), date(2026, 11, 24), date(2026, 11, 25)]
TEST_SUITES = {
    "holdout": suites.Suite("holdout", "test", (str(DAYS[5]), str(DAYS[7])), n_days=2),
    "fixed1": suites.Suite("fixed1", "test", (str(DAYS[0]), str(DAYS[2])),
                           ((str(DAYS[0]), str(DAYS[2])),), n_days=3),
}


@pytest.fixture(autouse=True)
def _suites(monkeypatch):
    monkeypatch.setattr(suites, "SUITES", TEST_SUITES)
    monkeypatch.setattr(bhs, "PRICES_FROM", str(DAYS[5]))


def _market(drop_days=(), drop_rounds=(), exec_prices=None):
    idx = [r["execution"] for d in DAYS for r in calendar.rounds_for(d)]
    ex = pd.DataFrame({t: [100 * (1 + 0.001 * (j % 5 - 1)) ** i for i in range(len(idx))]
                       for j, t in enumerate(TICKERS)}, index=idx)
    cl = pd.DataFrame(1.0005 * ex.groupby(ex.index.date).last().to_numpy(),
                      index=[calendar.at(d, calendar.session_close(d)) for d in DAYS], columns=TICKERS)
    if exec_prices is not None:
        ex = exec_prices
    ex = ex[[ts.date() not in drop_days and ts not in drop_rounds for ts in ex.index]]
    cl = cl[[ts.date() not in drop_days for ts in cl.index]]
    info = pd.DataFrame({"end": pd.Series([], dtype=f"datetime64[ns, {calendar.TZ}]")})
    return sim.Market(ex, cl, info)


def _shipped(full, drop_days=(), **kw):
    keep = set(bhs.shipped_days(full)) - set(drop_days)
    return _market(drop_days=[d for d in DAYS if d not in keep], **kw)


def test_the_price_file_holds_each_fixed_window_and_not_the_sessions_between():
    """Shipping only from the holdout's start left a fixed suite's 2025 windows unpriced;
    shipping everything between would publish months of prices no score reads."""
    full = _market()
    assert bhs.shipped_days(full) == DAYS[:3] + DAYS[5:]
    assert bhs.price_file_problems(_shipped(full), full) == []


def test_a_price_file_missing_a_session_of_a_fixed_window_fails_the_build():
    """A fixed window is an island in the price file. Short a session, the page would
    refuse every file for that suite, after the deploy rather than before it."""
    full = _market()
    problems = bhs.price_file_problems(_shipped(full, drop_days=[DAYS[1]]), full)
    assert any("suite fixed1" in p and "holds 2 trading days" in p for p in problems), problems


def test_a_price_file_missing_one_round_of_a_window_fails_the_build():
    """With its day present, a window still has 15 sessions; the missing round would only
    fail when a file traded at it, or never, for a file that held there."""
    full = _market()
    gone = calendar.rounds_for(DAYS[6])[3]["execution"]
    problems = bhs.price_file_problems(_shipped(full, drop_rounds=[gone]), full)
    assert any("suite holdout" in p and "lacks 1 of its rounds" in p for p in problems), problems


def test_a_rolling_window_cut_short_by_the_price_file_fails_the_build():
    """A rolling suite raises nothing on a missing day: it scores one window fewer, or a
    window whose sessions skip the day, under the same suite name."""
    full = _market()
    problems = bhs.price_file_problems(_shipped(full, drop_days=[DAYS[7]]), full)
    assert any("suite holdout" in p for p in problems), problems


def test_a_shipped_price_one_ulp_off_fails_the_build():
    """Every fill in the page would drift by a hair, and the page's numbers would no
    longer be the CLI's, which is the claim the scorer makes."""
    full = _market()
    ex = full.exec_prices.copy()
    ex.iloc[5, 3] = np.nextafter(ex.iloc[5, 3], np.inf)
    problems = bhs.price_file_problems(_shipped(full, exec_prices=ex), full)
    assert problems == ["exec_prices: not bit-identical to the full market's"]


def test_a_holdout_window_spanning_the_shipped_gap_is_not_scored_across_it():
    """The shipped market has a two-day hole between the fixed window and the holdout. A
    rolling suite reaching back over it would join sessions a week apart into one window."""
    full = _market()
    spans, _ = holdout.suite_spans(_shipped(full), suites.get("holdout"))
    assert [d for span in spans.values() for d in span if d < DAYS[5]] == []


# ------------------------------------------------------------------ the scorer's entry point

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "space"))
import webapp  # noqa: E402

from icaif import weights as W  # noqa: E402


@pytest.fixture
def page(monkeypatch):
    monkeypatch.setattr(webapp, "_MARKET", (_market(), {"snapshot": "snap"}))
    return webapp


def _template_file(tmp_path, page, suite, rebalance="once"):
    p = tmp_path / f"{suite}.json"
    p.write_text(page.template(suite, rebalance))
    return p


def test_the_pages_template_scores_as_its_suite_and_exactly_as_ew_hold(tmp_path, page):
    """The page's download is the first file most people score. If its keys or round_ids
    were the page's own guess, the scorer would reject the file it handed out; if its
    weights drifted from the reference's, "my file scores as ew_hold" would stop being true."""
    for name in TEST_SUITES:
        doc = json.loads(page.template(name, "once"))
        assert doc["suite"] == name
        dec = holdout.load_decisions(_template_file(tmp_path, page, name), _market(), name, strict=False)
        got, _ = holdout.rolling(dec, _market())
        from icaif import baselines
        want, _ = holdout.rolling_runs(baselines.EqualWeightHold, _market(), name)
        for k in holdout.METRICS:
            assert list(got[k]) == pytest.approx(list(want[k]), rel=1e-12)


def test_the_template_weights_are_the_strategy_codes_own_grid():
    """holdout.template floors 1/30 itself, since weights.py does not ship. A different
    rounding would leave cash a hair off and the template would fail its own cash check."""
    tickers = _market().tickers
    doc = holdout.template(_market(), "fixed1", "every")
    w = next(iter(doc["windows"].values()))[0]["weights"]
    assert w == W.safe({t: 1 / len(tickers) for t in tickers}, tickers)
    rounds = sum(len(calendar.rounds_for(d)) for d in DAYS[:3])
    assert len(doc["windows"][str(DAYS[0])]) == rounds


def test_the_page_offers_an_entry_for_a_whole_suite_only_and_stamps_its_suite(tmp_path, page):
    """An entry without its suite would rank as a holdout entry; one from a narrowed span
    would rank a month's run beside whole-suite runs."""
    fixed = json.loads(page.score(str(_template_file(tmp_path, page, "fixed1")), "", "", False,
                                  "pre_fee", "fixed1"))
    assert fixed["entry"]["suite"] == "fixed1" and fixed["report"]["suite"] == "fixed1"
    assert fixed["report"]["trading_days"] == 3

    rolling = _template_file(tmp_path, page, "holdout")
    one = json.loads(rolling.read_text())
    one["windows"] = {str(DAYS[5]): one["windows"][str(DAYS[5])]}
    rolling.write_text(json.dumps(one))
    narrowed = json.loads(page.score(str(rolling), str(DAYS[5]), str(DAYS[6]), False, "pre_fee", "holdout"))
    assert narrowed["entry"] is None and narrowed["report"]["windows"] == 1
    rolling = _template_file(tmp_path, page, "holdout")
    whole = json.loads(page.score(str(rolling), "", "", False, "pre_fee", "holdout"))
    assert whole["entry"]["suite"] == "holdout"
    post = json.loads(page.score(str(rolling), "", "", False, "post_fee", "holdout"))
    assert post["entry"] is None


def test_the_page_rejects_a_file_scored_as_another_suite_with_no_numbers(tmp_path, page):
    """The selector and the file can disagree. A partial score of a file the harness
    disagrees with would read as a real result."""
    out = json.loads(page.score(str(_template_file(tmp_path, page, "holdout")), "", "", False,
                                "pre_fee", "fixed1"))
    assert set(out) == {"rejected"} and "for suite 'holdout'" in out["rejected"]
    bad = json.loads(page.score(str(_template_file(tmp_path, page, "fixed1")), "2026-11-16",
                                "2026-11-17", False, "pre_fee", "fixed1"))
    assert set(bad) == {"rejected"} and "fixed windows" in bad["rejected"]
