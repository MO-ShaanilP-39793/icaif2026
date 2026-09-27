import numpy as np
import pandas as pd
import pytest

from icaif import calendar, compiler, kit, ranking, sim, windows
from icaif.compiler import Levers, compile_weights, plan
from tests.test_sim import DAY1, DAY2, TICKERS, _market

# Scores fall with ticker order, so TICKERS[i] has selection rank i + 1 under flat vol.
SCORES = pd.Series(np.linspace(1.0, 0.0, len(TICKERS)), index=TICKERS)
FLAT_VOL = pd.Series(0.02, index=TICKERS)
NONE = pd.Series(0.0, index=TICKERS)


def _book(w):
    return {t for t, v in w.items() if v > 0}


def test_every_compiled_decision_passes_the_organizers_own_validator():
    """One weight like 0.1 + 0.2 fails the backend's Decimal check and the round holds
    silently; random levers, NaNs and drifted books must never produce one."""
    rng = np.random.default_rng(7)
    for _ in range(300):
        s = pd.Series(rng.random(len(TICKERS)), index=TICKERS).mask(rng.random(len(TICKERS)) < 0.1)
        v = pd.Series(rng.uniform(0.005, 0.06, len(TICKERS)), index=TICKERS)
        cur = pd.Series(rng.dirichlet(np.ones(len(TICKERS))) * rng.uniform(0, 1.02), index=TICKERS)
        cur[rng.random(len(TICKERS)) < 0.5] = 0.0
        lv = Levers(exposure=float(rng.choice([0.3, 0.75, 1.0])), top_k=int(rng.integers(1, 31)),
                    gamma=float(rng.choice([0, 0.5, 1, 2])),
                    weighting=str(rng.choice(["equal", "inverse_vol"])),
                    buffer=int(rng.integers(0, 5)), band=float(rng.choice([0, 0.02, 0.1])))
        w = compile_weights(s, v, cur, lv, int(rng.integers(1, 8)))
        kit.validate_weights(w)
        assert set(w) == set(TICKERS)


def test_two_names_at_full_exposure_are_capped_at_30pct_with_the_rest_in_cash():
    """Sizing 1.0 over two names gives 0.50 each, which the backend rejects outright
    and the round holds. The excess must go to cash, not over the cap."""
    w = compile_weights(SCORES, FLAT_VOL, NONE, Levers(exposure=1.0, top_k=2), 1)
    assert _book(w) == set(TICKERS[:2])
    assert w[TICKERS[0]] == w[TICKERS[1]] == pytest.approx(0.30)
    assert sum(w.values()) == pytest.approx(0.60)
    kit.validate_weights(w)


def test_a_capped_inverse_vol_favourite_passes_its_excess_to_the_other_names():
    """Clipping alone would leave the book under its exposure lever whenever a
    low-vol name hits the cap, and the agent would be running a smaller book than it chose."""
    vol = FLAT_VOL.copy()
    vol[TICKERS[0]] = 0.002
    w = compile_weights(SCORES, vol, NONE, Levers(exposure=1.0, top_k=5, gamma=0), 1)
    assert w[TICKERS[0]] == pytest.approx(0.30)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-5)
    assert max(w.values()) <= 0.30


def test_a_held_name_just_outside_top_k_is_kept_and_one_past_the_buffer_is_dropped():
    """Without hysteresis a name wobbling between rank 10 and 11 is bought and sold
    day after day: turnover and fees for a ranking difference that is noise."""
    lv = Levers(top_k=3, buffer=2, band=0.0, weighting="equal")
    kept = compile_weights(SCORES, FLAT_VOL, {TICKERS[3]: 0.25}, lv, 1)  # rank 4 = top_k + 1
    assert _book(kept) == {TICKERS[0], TICKERS[1], TICKERS[3]}
    edge = compile_weights(SCORES, FLAT_VOL, {TICKERS[4]: 0.25}, lv, 1)  # rank 5 = top_k + buffer
    assert TICKERS[4] in _book(edge)
    dropped = compile_weights(SCORES, FLAT_VOL, {TICKERS[5]: 0.25}, lv, 1)  # top_k + buffer + 1
    assert _book(dropped) == set(TICKERS[:3])


def test_the_band_suppresses_resizes_smaller_than_itself():
    """Turnover is a ranked metric: re-sizing every name by 1% each morning pays a rank
    for nothing. A drifted book inside the band must produce no trade at all."""
    lv = Levers(top_k=3, band=0.02, weighting="equal", exposure=0.75)
    drifted = {TICKERS[0]: 0.26, TICKERS[1]: 0.24, TICKERS[2]: 0.255}  # target 0.25 each
    target, changed = plan(SCORES, FLAT_VOL, drifted, lv, 1)
    assert len(changed) == 0
    assert target[TICKERS[0]] == 0.26
    moved = {TICKERS[0]: 0.28, TICKERS[1]: 0.24, TICKERS[2]: 0.255}
    target, changed = plan(SCORES, FLAT_VOL, moved, lv, 1)
    assert list(changed) == [TICKERS[0]] and target[TICKERS[0]] == pytest.approx(0.25)


def test_the_band_never_blocks_an_exit():
    """A 1% position dropped from the book but kept by the band is a stray the
    selection no longer owns; nothing would ever sell it."""
    lv = Levers(top_k=3, buffer=0, band=0.02)
    w = compile_weights(SCORES, FLAT_VOL, {TICKERS[20]: 0.01}, lv, 1)
    assert w[TICKERS[20]] == 0.0


def test_a_round_outside_rebalance_rounds_never_adds_a_name():
    """Re-selecting every hour churns the book seven times a day on scores that only
    change once a day."""
    assert _book(compile_weights(SCORES, FLAT_VOL, NONE, Levers(), 2)) == set()
    held = {TICKERS[25]: 0.2, TICKERS[29]: 0.1}
    for r in range(2, 8):
        target, changed = plan(SCORES, FLAT_VOL, held, Levers(), r)
        assert len(changed) == 0 and _book(target) == set(held)


def test_a_nan_score_or_vol_never_sells_a_held_name():
    """A NaN is a feed gap, not a bad score. Reading it as 'not selected' dumps the
    position on a data hiccup and buys it back the next day."""
    s, v = SCORES.copy(), FLAT_VOL.copy()
    s[TICKERS[29]] = np.nan
    v[TICKERS[28]] = np.nan
    held = {TICKERS[29]: 0.07, TICKERS[28]: 0.05}
    for r in (1, 2, 4):
        w = compile_weights(s, v, held, Levers(), r)
        assert w[TICKERS[29]] == pytest.approx(0.07) and w[TICKERS[28]] == pytest.approx(0.05)
    w = compile_weights(s, v, held, Levers(exposure=0.75, top_k=10), 1)
    assert len(_book(w)) == 10  # the two frozen names take two of the ten slots
    assert sum(w.values()) == pytest.approx(0.75, abs=1e-5)


def test_a_nan_score_is_never_selected():
    s = SCORES.copy()
    s[TICKERS[0]] = np.nan
    assert TICKERS[0] not in _book(compile_weights(s, FLAT_VOL, NONE, Levers(top_k=3), 1))


def test_an_avoided_name_is_never_bought_and_is_sold_in_any_round():
    lv = Levers(top_k=3, avoid={TICKERS[0]})
    assert TICKERS[0] not in _book(compile_weights(SCORES, FLAT_VOL, NONE, lv, 1))
    w = compile_weights(SCORES, FLAT_VOL, {TICKERS[0]: 0.2, TICKERS[1]: 0.2}, lv, 5)
    assert w[TICKERS[0]] == 0.0 and w[TICKERS[1]] == pytest.approx(0.2)


def test_the_stop_fires_only_past_its_threshold():
    """A stop that fired on a -1.9 sigma move at a 2-sigma setting, or on a missing
    entry price, would sell on noise and gaps and look like prudence."""
    lv = Levers(stop_sigma=2.0)
    held = {TICKERS[0]: 0.2}
    entry = pd.Series(100.0, index=TICKERS)
    inside = pd.Series(100.0 * np.exp(-0.039), index=TICKERS)
    past = pd.Series(100.0 * np.exp(-0.041), index=TICKERS)
    assert compile_weights(SCORES, FLAT_VOL, held, lv, 3, entry, inside)[TICKERS[0]] > 0
    assert compile_weights(SCORES, FLAT_VOL, held, lv, 3, entry, past)[TICKERS[0]] == 0.0
    assert compile_weights(SCORES, FLAT_VOL, held, lv, 3, None, past)[TICKERS[0]] > 0
    assert compile_weights(SCORES, FLAT_VOL, held, Levers(), 3, entry, past)[TICKERS[0]] > 0


def test_an_out_of_range_lever_raises_rather_than_being_clamped():
    """A clamped exposure of 1.2 runs a different book from the one the agent reports."""
    for bad in ({"exposure": 1.2}, {"top_k": 0}, {"weighting": "risk_parity"},
                {"rebalance_rounds": (0,)}, {"stop_sigma": -1.0}):
        with pytest.raises(ValueError):
            Levers(**bad)


def test_a_strategy_asking_for_another_days_scores_fails_loudly():
    """Tomorrow's prediction read by an off-by-one looks like a brilliant model."""
    panel = compiler.DailyPanel(pd.DataFrame([SCORES, SCORES], index=pd.to_datetime([DAY1, DAY2])),
                                TICKERS)
    deadline = calendar.rounds_for(DAY1)[0]["deadline"]
    assert panel.for_day(DAY1, deadline).equals(SCORES)
    with pytest.raises(compiler.LookAheadError):
        panel.for_day(DAY2, deadline)


def test_trailing_vol_for_a_day_excludes_that_days_own_close():
    """Unshifted, the vol a decision divides by already knows whether today is a big day."""
    dates = pd.bdate_range("2026-06-01", periods=40)
    rng = np.random.default_rng(0)
    rows = []
    for t in TICKERS:
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        px[30:] *= 1.5 if t == TICKERS[0] else 1.0  # a +50% close on dates[30]
        rows += [{"date": d, "ticker": t, "open": p, "high": p, "low": p, "close": p,
                  "volume": 1e6} for d, p in zip(dates, px)]
    vol = compiler.trailing_daily_vol(pd.DataFrame(rows)).frame[TICKERS[0]]
    assert vol[dates[30]] < 0.02 < vol[dates[31]]


def _info_bars(days):
    rows = []
    for d in days:
        for r in calendar.rounds_for(d):
            for t in TICKERS:
                rows.append({"ticker": t, "start": r["execution"],
                             "end": r["execution"] + pd.Timedelta(hours=1),
                             "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0})
    return pd.DataFrame(rows)


def test_a_compiled_strategy_trades_only_in_its_rebalance_rounds_and_never_invalidly():
    """Resubmitting unchanged weights every hour re-sizes the book to each round's price
    and pays the fee on it: turnover that the metric counts and the book never chose."""
    days = [DAY1, DAY2]
    m = _market(days, info_bars=_info_bars(days))
    idx = pd.to_datetime(days)
    scores = compiler.DailyPanel(pd.DataFrame([SCORES.values, SCORES.values[::-1]], index=idx,
                                              columns=TICKERS), TICKERS)
    vol = compiler.DailyPanel(pd.DataFrame([FLAT_VOL, FLAT_VOL], index=idx), TICKERS)
    res = sim.run(compiler.compiled(scores, vol, Levers(top_k=3, buffer=0))(), m, DAY1, 2)
    assert res.invalid_rounds == []
    traded = [(p["execution"].date(), p["execution"].hour) for p in res.periods
              if p["traded_notional"] > 0]
    assert traded == [(DAY1, 9), (DAY2, 9)]  # day 2's reversed scores rotate the book
    last = res.ledger.iloc[-1]
    assert {t for t in TICKERS if last[t] > 0} == set(TICKERS[-3:])


def _field_window(rng, names):
    vals = rng.choice([0.0, 0.01, 0.02, -0.01], size=(len(names), 4))
    return pd.DataFrame(vals, index=names, columns=list(ranking.METRICS))


def test_a_candidate_identical_to_a_field_member_ties_it():
    """If a copy of a field member scored differently from it, the sweep's scores would
    be an artefact of how the candidate was inserted, not of what it did."""
    rng = np.random.default_rng(3)
    names = ["a", "b", "c", "d"]
    field_rows, cand_rows = [], []
    for w in range(20):
        m = _field_window(rng, names)
        r = ranking.rank_window(m)
        field_rows += [{"window": w, "strategy": n, **m.loc[n], **r.loc[n]} for n in names]
        cand_rows.append({"window": w, "strategy": "copy_of_b", **m.loc["b"]})
    field = pd.DataFrame(field_rows)
    got = windows.rank_against_field(pd.DataFrame(cand_rows), field).set_index("window")
    for w in range(20):
        m = field[field.window == w].set_index("strategy")[list(ranking.METRICS)]
        joint = ranking.rank_window(pd.concat([m, m.loc[["b"]].rename(index={"b": "copy"})]))
        assert got.loc[w, "overall_score"] == joint.loc["copy", "overall_score"]
        assert got.loc[w, "overall_score"] == joint.loc["b", "overall_score"]
        assert got.loc[w, "position"] == joint.loc["b", "position"]


def test_ranking_one_candidate_against_the_field_matches_a_full_rerank():
    """The shortcut must agree with `rank_window` on the field plus the candidate,
    ties included, or the sweep optimises a scorer the competition never uses."""
    rng = np.random.default_rng(11)
    names = list("abcdefg")
    for w in range(200):
        m = _field_window(rng, names)
        cand = _field_window(rng, ["x"]).iloc[0]
        field = pd.DataFrame([{"window": w, "strategy": n, **m.loc[n]} for n in names])
        got = windows.rank_against_field(
            pd.DataFrame([{"window": w, "strategy": "x", **cand}]), field).iloc[0]
        want = ranking.rank_window(pd.concat([m, cand.to_frame("x").T])).loc["x"]
        for col in want.index:
            assert got[col] == want[col], (w, col)
