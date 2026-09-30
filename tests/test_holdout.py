import json
from datetime import date

import pandas as pd
import pytest

from icaif import calendar, data, holdout, sim
from icaif import weights as W
from icaif.rankplay import _Replay

TICKERS = sorted(data.load_universe())
# Wednesday, the Friday half-day (rounds 1-4 only), then a full week.
DAYS = [date(2026, 11, 25), date(2026, 11, 27), date(2026, 11, 30), date(2026, 12, 1),
        date(2026, 12, 2), date(2026, 12, 3)]
START, END = DAYS[0], DAYS[-1]


def _market(issues=None):
    """Prices drift up 0.1% per round per ticker, with a ticker-specific tilt, so
    every rebalance trades and every metric is non-trivial."""
    idx = [r["execution"] for d in DAYS for r in calendar.rounds_for(d)]
    ex = pd.DataFrame({t: [100 * (1 + 0.001 * (j % 5 - 1)) ** i for i in range(len(idx))]
                       for j, t in enumerate(TICKERS)}, index=idx)
    cidx = [calendar.at(d, calendar.session_close(d)) for d in DAYS]
    last = [ex.index.get_indexer([max(t for t in idx if t.date() == d)])[0] for d in DAYS]
    cl = pd.DataFrame(ex.iloc[last].to_numpy() * 1.0005, index=cidx, columns=TICKERS)
    info = pd.DataFrame({"end": pd.Series([], dtype=f"datetime64[ns, {calendar.TZ}]")})
    return sim.Market(ex, cl, info, issues=issues or {})


def _alternating():
    """Round-by-round weights that differ, so a replay that dropped or shuffled a round
    would change the numbers."""
    a = W.safe({t: (0.05 if i < 10 else 0.01) for i, t in enumerate(TICKERS)}, TICKERS)
    b = W.safe({t: (0.01 if i < 10 else 0.03) for i, t in enumerate(TICKERS)}, TICKERS)
    return {(d, r["round"]): (a if k % 2 else b)
            for k, (d, r) in enumerate((d, r) for d in DAYS for r in calendar.rounds_for(d))}


def _entry(key, w, cash=None):
    if cash is None:
        cash = 1 - sum(w.values())
    return {"round_id": holdout.round_id(*key), "cash": cash, "weights": w}


def _write(tmp_path, entries, name="agent"):
    p = tmp_path / "decisions.json"
    p.write_text(json.dumps({"strategy": name, "decisions": entries}))
    return p


def _load(tmp_path, entries, **kw):
    return holdout.load_decisions(_write(tmp_path, entries), _market(), START, END, **kw)


def test_the_continuous_run_matches_a_direct_sim_run_of_the_same_weights(tmp_path):
    """The harness's parse of the JSON is the only new step between an agent and the
    ledger. If it reordered, dropped or rounded a round, the harness would score a
    different book than the one the agent wrote, and still print four clean numbers."""
    plan = _alternating()
    dec = _load(tmp_path, [_entry(k, w) for k, w in plan.items()])
    got, _ = holdout.continuous(dec, _market(), START, END)
    want = sim.run(_Replay(plan), _market(), START, len(DAYS)).metrics()

    for k in holdout.METRICS:
        assert got[k] == pytest.approx(want[k], rel=1e-12)
    assert got["turnover"] > 0 and got["cumulative_return"] != 0
    assert got["missing_rounds"] == 0 and got["invalid_rounds"] == 0
    assert got["rounds"] == 7 * 5 + 4


def test_a_cash_weight_that_disagrees_with_the_stock_weights_rejects_the_file(tmp_path):
    """An agent whose cash says 25% while its stocks sum to 90% has a bug in one of the
    two. Scoring the stocks alone would hide it."""
    plan = _alternating()
    entries = [_entry(k, w) for k, w in plan.items()]
    entries[3]["cash"] += 0.01
    with pytest.raises(holdout.DecisionFileError, match=entries[3]["round_id"]):
        _load(tmp_path, entries)


def test_a_round_that_is_not_in_the_calendar_rejects_the_file(tmp_path):
    """A round on a weekend, or round 5 on a half-day, means the agent's clock is not the
    competition's. Dropping it silently would score a schedule the agent never followed."""
    w = next(iter(_alternating().values()))
    for bad in [(date(2026, 11, 28), 1), (date(2026, 11, 27), 5), (date(2026, 12, 4), 1)]:
        with pytest.raises(holdout.DecisionFileError, match=holdout.round_id(*bad)):
            _load(tmp_path, [_entry((START, 1), w), _entry(bad, w)])


def test_a_duplicate_round_id_rejects_the_file_instead_of_keeping_the_last(tmp_path):
    """Two decisions for one round are two different books; either choice is a guess."""
    plan = _alternating()
    entries = [_entry(k, w) for k, w in plan.items()]
    entries.append(entries[0])
    with pytest.raises(holdout.DecisionFileError, match="second decision"):
        _load(tmp_path, entries)


def test_a_round_missing_a_symbol_rejects_the_file(tmp_path):
    w = dict(next(iter(_alternating().values())))
    w.pop(TICKERS[0])
    with pytest.raises(holdout.DecisionFileError, match=TICKERS[0]):
        _load(tmp_path, [_entry((START, 1), w)])


def test_a_missing_round_holds_positions_and_is_reported(tmp_path):
    """The backend holds a round with no decision. The harness must too, and it must
    say so, because a quietly held round scores as deliberate low turnover."""
    plan = _alternating()
    gone = (DAYS[2], 3)
    del plan[gone]
    dec = _load(tmp_path, [_entry(k, w) for k, w in plan.items()])
    got, res = holdout.continuous(dec, _market(), START, END)

    assert dec.missing == [holdout.round_id(*gone)]
    assert got["missing_rounds"] == 1
    held = [p for p in res.periods if p["held"]]
    assert len(held) == 1 and held[0]["traded_notional"] == 0


def test_an_over_cap_weight_holds_the_round_like_the_backend(tmp_path):
    """0.1 + 0.2 is 0.30000000000000004, over the cap. Live, the round holds. If the
    harness snapped it to the grid, it would score a trade the backend never makes."""
    plan = _alternating()
    key = (DAYS[1], 2)
    bad = dict(plan[key])
    bad[TICKERS[0]] = 0.1 + 0.2
    plan[key] = bad
    dec = _load(tmp_path, [_entry(k, w) for k, w in plan.items()])
    got, _ = holdout.continuous(dec, _market(), START, END)

    assert [i["round_id"] for i in dec.invalid] == [holdout.round_id(*key)]
    assert got["invalid_rounds"] == 1


def test_strict_mode_makes_a_missing_round_fatal(tmp_path):
    plan = _alternating()
    del plan[(DAYS[0], 1)]
    with pytest.raises(holdout.DecisionFileError, match="strict"):
        _load(tmp_path, [_entry(k, w) for k, w in plan.items()], strict=True)


def test_every_rolling_window_restarts_from_one_million_in_cash(tmp_path):
    """A window that inherited the previous window's book would start invested, which
    changes its day-1 turnover and its return base. Each window must equal a fresh run."""
    plan = _alternating()
    dec = _load(tmp_path, [_entry(k, w) for k, w in plan.items()])
    wins, skipped = holdout.rolling(dec, _market(), START, END, n_days=3)

    assert list(wins["window_start"]) == [str(d) for d in DAYS[:4]] and skipped == []
    for _, row in wins.iterrows():
        fresh = sim.run(_Replay(plan), _market(), date.fromisoformat(row["window_start"]), 3)
        for k in holdout.METRICS:
            assert row[k] == pytest.approx(fresh.metrics()[k], rel=1e-12)


def test_a_window_touching_a_degraded_day_is_skipped_and_named(tmp_path):
    """A degraded day carries stand-in prices, a zero return that never happened. A
    window across it looks calmer than the market was."""
    plan = _alternating()
    path = _write(tmp_path, [_entry(k, w) for k, w in plan.items()])
    market = _market(issues={"degraded_days": [str(DAYS[3])]})
    dec = holdout.load_decisions(path, market, START, END)
    wins, skipped = holdout.rolling(dec, market, START, END, n_days=3)

    assert skipped == [str(d) for d in DAYS[1:4]]
    assert list(wins["window_start"]) == [str(DAYS[0])]
    got, _ = holdout.continuous(dec, market, START, END)
    assert got["degraded_days"] == [str(DAYS[3])]


def test_independent_windows_counts_windows_that_share_no_day(tmp_path):
    """150 overlapping windows are ~10 samples; the summary must not read as 150."""
    plan = _alternating()
    dec = _load(tmp_path, [_entry(k, w) for k, w in plan.items()])
    wins, _ = holdout.rolling(dec, _market(), START, END, n_days=3)
    assert holdout.summarise_rolling(wins).attrs == {"windows": 4, "independent_windows": 2}


def test_a_market_that_ends_before_the_span_rejects_rather_than_scoring_less(tmp_path):
    plan = _alternating()
    path = _write(tmp_path, [_entry(k, w) for k, w in plan.items()])
    with pytest.raises(holdout.DecisionFileError, match="ends"):
        holdout.load_decisions(path, _market(), START, date(2026, 12, 31))
