import json
from datetime import date

import pandas as pd
import pytest

from icaif import baselines, calendar, data, holdout, sim
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


N = 3   # window length here: six days give four windows
STARTS = DAYS[:4]


def _rounds(days):
    return [(d, r["round"]) for d in days for r in calendar.rounds_for(d)]


def _span(start):
    i = DAYS.index(start)
    return DAYS[i:i + N]


def _plan():
    """Per window, round-by-round weights that differ by round and by window, so a replay
    that dropped or shuffled a round, or scored a window with another's plan, would
    change the numbers."""
    plans = {}
    for k, ws in enumerate(STARTS):
        a = W.safe({t: (0.05 if i < 10 + k else 0.01) for i, t in enumerate(TICKERS)}, TICKERS)
        b = W.safe({t: (0.01 if i < 10 else 0.03) for i, t in enumerate(TICKERS)}, TICKERS)
        plans[ws] = {key: (a if j % 2 else b) for j, key in enumerate(_rounds(_span(ws)))}
    return plans


def _entry(key, w, cash=None):
    if cash is None:
        cash = 1 - sum(w.values())
    return {"round_id": holdout.round_id(*key), "cash": cash, "weights": w}


def _doc(plans):
    return {str(ws): [_entry(k, w) for k, w in plan.items()] for ws, plan in plans.items()}


def _write(tmp_path, windows, name="agent"):
    p = tmp_path / "decisions.json"
    p.write_text(json.dumps({"strategy": name, "windows": windows}))
    return p


def _load(tmp_path, windows, market=None, **kw):
    return holdout.load_decisions(_write(tmp_path, windows), market or _market(), START, END,
                                  n_days=N, **kw)


def test_each_window_scores_its_own_decisions_from_one_million_in_cash(tmp_path):
    """A window run with another window's plan, or one that inherited a book, would print
    four clean numbers for a run the agent never made."""
    plans = _plan()
    dec = _load(tmp_path, _doc(plans))
    wins, skipped = holdout.rolling(dec, _market())

    assert list(wins["window_start"]) == [str(d) for d in STARTS] and skipped == []
    for ws, (_, row) in zip(STARTS, wins.iterrows()):
        want = sim.run(_Replay(plans[ws]), _market(), ws, N).metrics()
        for k in holdout.METRICS:
            assert row[k] == pytest.approx(want[k], rel=1e-12)
        assert row["missing_rounds"] == 0 and row["invalid_rounds"] == 0
    assert wins["cumulative_return"].nunique() == len(STARTS)


def test_an_equal_weight_hold_file_scores_exactly_as_the_ew_hold_reference(tmp_path):
    """The point of one run per window: a buy-and-hold buys 1/30 each at every window's
    first round. Replaying one six-month run instead bought the weights that had drifted
    since day one, and the same idea scored differently as a file and as a reference."""
    w = W.safe({t: 1 / len(TICKERS) for t in TICKERS}, TICKERS)
    windows = {str(ws): [_entry(_rounds(_span(ws))[0], w)] for ws in STARTS}
    dec = _load(tmp_path, windows)
    got, _ = holdout.rolling(dec, _market())
    want, _ = holdout.rolling_runs(baselines.EqualWeightHold, _market(), START, END, n_days=N)

    for k in holdout.METRICS:
        assert list(got[k]) == pytest.approx(list(want[k]), rel=1e-12)


def test_the_old_one_run_file_is_rejected_by_name_not_replayed(tmp_path):
    """Accepted alongside, old and new files would rank side by side with nothing to say
    one of them scored a different strategy."""
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"strategy": "x", "decisions": []}))
    with pytest.raises(holdout.DecisionFileError, match="old one-run format"):
        holdout.load_decisions(p, _market(), START, END, n_days=N)


def test_a_missing_window_rejects_the_file(tmp_path):
    """A window with no run is not a window of holds: the agent never ran there."""
    windows = _doc(_plan())
    del windows[str(STARTS[2])]
    with pytest.raises(holdout.DecisionFileError, match=f"window {STARTS[2]}: no decisions"):
        _load(tmp_path, windows)


def test_a_key_that_starts_no_window_rejects_the_file(tmp_path):
    """A window keyed on a weekend, or too near the end for 15 days, means the agent's
    windows are not the board's."""
    w = next(iter(_plan()[START].values()))
    for bad in ["2026-11-28", str(DAYS[4]), "not-a-date"]:
        windows = _doc(_plan())
        windows[bad] = [_entry((START, 1), w)]
        with pytest.raises(holdout.DecisionFileError, match=f"window '?{bad}"):
            _load(tmp_path, windows)


def test_a_round_outside_its_own_window_rejects_the_file(tmp_path):
    """A decision for day 4 inside the window that starts on day 1 is a round that window
    never has; scoring it would graft another run's trade onto this one."""
    windows = _doc(_plan())
    w = next(iter(_plan()[START].values()))
    windows[str(START)].append(_entry((DAYS[3], 1), w))
    with pytest.raises(holdout.DecisionFileError, match="outside the window"):
        _load(tmp_path, windows)


def test_a_cash_weight_that_disagrees_with_the_stock_weights_rejects_the_file(tmp_path):
    """An agent whose cash says 25% while its stocks sum to 90% has a bug in one of the
    two. Scoring the stocks alone would hide it."""
    windows = _doc(_plan())
    bad = windows[str(STARTS[1])][3]
    bad["cash"] += 0.01
    with pytest.raises(holdout.DecisionFileError, match=bad["round_id"]):
        _load(tmp_path, windows)


def test_a_round_that_is_not_in_the_calendar_rejects_the_file(tmp_path):
    """A round on a weekend, or round 5 on a half-day, means the agent's clock is not the
    competition's. Dropping it silently would score a schedule the agent never followed."""
    w = next(iter(_plan()[START].values()))
    for bad in [(date(2026, 11, 28), 1), (date(2026, 11, 27), 5)]:
        windows = _doc(_plan())
        windows[str(START)].append(_entry(bad, w))
        with pytest.raises(holdout.DecisionFileError, match=holdout.round_id(*bad)):
            _load(tmp_path, windows)


def test_a_duplicate_round_id_rejects_the_file_instead_of_keeping_the_last(tmp_path):
    """Two decisions for one round are two different books; either choice is a guess."""
    windows = _doc(_plan())
    windows[str(START)].append(windows[str(START)][0])
    with pytest.raises(holdout.DecisionFileError, match="second decision"):
        _load(tmp_path, windows)


def test_a_round_missing_a_symbol_rejects_the_file(tmp_path):
    windows = _doc(_plan())
    windows[str(START)][0]["weights"].pop(TICKERS[0])
    with pytest.raises(holdout.DecisionFileError, match=TICKERS[0]):
        _load(tmp_path, windows)


def test_a_systematic_error_is_listed_once_per_kind_not_thousands_of_times(tmp_path):
    """The same mistake repeats in every window of a real file (~11,000 rounds); a message
    that long hides the first line, which is the one that explains it."""
    windows = _doc(_plan())
    for entries in windows.values():
        for e in entries:
            e["cash"] += 0.01
    with pytest.raises(holdout.DecisionFileError) as err:
        _load(tmp_path, windows)
    lines = str(err.value).splitlines()
    assert len(lines) <= holdout.MAX_LISTED_ERRORS + 2 and "more" in lines[-1]


def test_a_missing_round_holds_positions_and_is_reported_against_its_window(tmp_path):
    """The backend holds a round with no decision. The harness must too, and it must
    say so, because a quietly held round scores as deliberate low turnover."""
    plans = _plan()
    gone = (DAYS[2], 3)
    del plans[STARTS[1]][gone]
    dec = _load(tmp_path, _doc(plans))
    wins, _ = holdout.rolling(dec, _market())

    assert dec.missing == [f"{STARTS[1]} {holdout.round_id(*gone)}"]
    assert list(wins["missing_rounds"]) == [0, 1, 0, 0]
    # The same round in the other windows that contain it is still traded.
    assert (DAYS[2], 3) in plans[STARTS[0]] and (DAYS[2], 3) in plans[STARTS[2]]


def test_an_over_cap_weight_holds_the_round_like_the_backend(tmp_path):
    """0.1 + 0.2 is 0.30000000000000004, over the cap. Live, the round holds. If the
    harness snapped it to the grid, it would score a trade the backend never makes."""
    plans = _plan()
    key = (DAYS[1], 2)
    bad = dict(plans[START][key])
    bad[TICKERS[0]] = 0.1 + 0.2
    plans[START][key] = bad
    dec = _load(tmp_path, _doc(plans))
    wins, _ = holdout.rolling(dec, _market())

    assert [(i["window"], i["round_id"]) for i in dec.invalid] == [(str(START), holdout.round_id(*key))]
    assert list(wins["invalid_rounds"]) == [1, 0, 0, 0]


def test_strict_mode_makes_a_missing_round_fatal(tmp_path):
    plans = _plan()
    del plans[START][(DAYS[0], 1)]
    with pytest.raises(holdout.DecisionFileError, match="strict"):
        _load(tmp_path, _doc(plans), strict=True)


def test_a_window_touching_a_degraded_day_is_skipped_and_named(tmp_path):
    """A degraded day carries stand-in prices, a zero return that never happened. A
    window across it looks calmer than the market was. Its key in the file is ignored,
    since the agent cannot know which days the harness will call degraded."""
    market = _market(issues={"degraded_days": [str(DAYS[3])]})
    dec = _load(tmp_path, _doc(_plan()), market=market)
    wins, skipped = holdout.rolling(dec, market)

    assert skipped == [str(d) for d in DAYS[1:4]]
    assert list(wins["window_start"]) == [str(DAYS[0])]


def test_independent_windows_counts_windows_that_share_no_day(tmp_path):
    """150 overlapping windows are ~10 samples; the summary must not read as 150."""
    dec = _load(tmp_path, _doc(_plan()))
    wins, _ = holdout.rolling(dec, _market())
    assert holdout.summarise_rolling(wins).attrs == {"windows": 4, "independent_windows": 2}


def test_a_market_that_ends_before_the_span_rejects_rather_than_scoring_less(tmp_path):
    path = _write(tmp_path, _doc(_plan()))
    with pytest.raises(holdout.DecisionFileError, match="ends"):
        holdout.load_decisions(path, _market(), START, date(2026, 12, 31), n_days=N)
