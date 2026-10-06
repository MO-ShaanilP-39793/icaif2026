import json

import pandas as pd
import pytest

from icaif import boardrank, leaderboard as lb, replay_entry as re_, suites

O4 = [tuple(w) for w in suites.SUITES["official4"].windows]
M = lb.METRICS
HOLD = (0.03, 5.0, 0.01, 0.007)


def _ref(name, per_window):
    wins = pd.DataFrame([{"window_start": s, "window_end": e, **dict(zip(M, m))}
                         for (s, e), m in zip(O4, per_window)])
    return lb.make_entry(name, lb.REFERENCE, wins, span=suites.SUITES["official4"].span,
                         sizing="pre_fee", market_snapshot="snap", submitted_at="", suite="official4")


def _holdout_ref(name):
    wins = pd.DataFrame([{"window_start": "2026-01-02", "window_end": "2026-01-23",
                          **dict(zip(M, (0, 0, 0, 0)))}])
    return lb.make_entry(name, lb.REFERENCE, wins, span=("2026-01-02", "2026-06-30"),
                         sizing="pre_fee", market_snapshot="snap", submitted_at="")


REFS = [_holdout_ref("cash"), _ref("cash", [(0, 0, 0, 0)] * 4), _ref(boardrank.ANCHOR, [HOLD] * 4)]
BOARD = lb.boards(REFS)


def _out(*lines, fallbacks=None):
    text = "summary...\n" + (f"fallbacks: {fallbacks}\n" if fallbacks is not None else "")
    return text + "".join(f"\n{m}: {c} calls, ${d:.2f}; cache hits {h}, misses {c - h}\n"
                          for m, c, d, h in lines) + "\nlog: x   (400s)\n"


def _run(tag, windows, desk="free_blank_claude", model="gemini:gemini-2.5-pro:high:wire1",
         cost=0.7, hold=HOLD, log=None, out=None):
    rows = []
    for w in windows:
        rows.append({"window": w, "strategy": boardrank.ANCHOR, **dict(zip(M, hold)), "invalid_rounds": 0})
        rows.append({"window": w, "strategy": "q_riskparity_entry_regime", **dict(zip(M, HOLD))})
        rows.append({"window": w, "strategy": desk, **dict(zip(M, (0.02, 3.0, 0.01, 0.04))),
                     "invalid_rounds": 2})
    return re_.read_run(tag, pd.DataFrame(rows), out or _out((model, 15, cost, 0)), log)


def _four(**kw):
    return [_run(f"t{i}", [s], **kw) for i, (s, _) in enumerate(O4)]


def _build(runs, **kw):
    args = dict(name="v1", desk="free desk", window_choice="suite", snapshot="snap", author="a",
                submitted_at="2026-10-06T12:00:00Z")
    args.update(kw)
    return re_.build_entry(runs, "official4", BOARD, **args)


def test_four_one_window_runs_make_one_entry_that_ranks_in_the_suites_field():
    """The Earnings season was run one window at a time. Submitted run by run, each would
    cover one of four windows and be named "not ranked" on the board."""
    e = _build(_four())
    assert e["suite"] == "official4" and e["kind"] == lb.AGENTIC
    assert [(w["window_start"], w["window_end"]) for w in e["windows"]] == O4
    assert e["agent"]["calls"] == 60 and e["agent"]["cost_usd"] == pytest.approx(2.8)
    assert e["agent"]["replay"] == ["t0", "t1", "t2", "t3"]
    assert e["invalid_rounds"] == 8
    o4 = lb.boards(REFS + [e])["suites"]["official4"]
    assert "v1" in {r["strategy"] for r in o4["rows"]} and o4["excluded"] == []


def test_a_window_no_run_covers_or_two_runs_cover_refuses_before_upload():
    """Three of four windows would be excluded on the board after the upload; a window
    run twice leaves the choice of run to whichever came first."""
    with pytest.raises(re_.ReplayError, match=f"no run covers {O4[1][0]}"):
        _build([r for i, r in enumerate(_four()) if i != 1])
    with pytest.raises(re_.ReplayError, match=f"window {O4[0][0]} is in both t0 and again"):
        _build(_four() + [_run("again", [O4[0][0]])])


def test_runs_of_another_desk_or_model_never_combine():
    """Four windows from two desks, or Pro in one and Flash in another, would rank as one
    method that never ran."""
    other_desk = _four()
    other_desk[2] = _run("v2", [O4[2][0]], desk="v2_claude")
    with pytest.raises(re_.ReplayError, match="different desks"):
        _build(other_desk)
    other_model = _four()
    other_model[3] = _run("flash", [O4[3][0]], model="gemini:gemini-2.5-flash:medium:wire1")
    with pytest.raises(re_.ReplayError, match="different models"):
        _build(other_model)


def test_a_window_priced_differently_from_the_board_is_refused():
    """The replay and the board price on separate snapshots. A window they disagree on
    would rank the agent against a field priced on other data, and still look like a place."""
    runs = _four()
    runs[1] = _run("t1", [O4[1][0]], hold=(HOLD[0] + 1e-5, *HOLD[1:]))
    with pytest.raises(boardrank.AnchorMismatch, match="priced this window differently"):
        _build(runs)


def test_every_models_spend_is_summed_and_cached_answers_are_named():
    """A v2 desk prints one spend line per model. Reading the first reported about half the
    cost; a cached answer is not paid again, so a cost without the cache count overstates
    how cheap the run was to reproduce."""
    out = _out(("gemini:gemini-2.5-flash:medium:wire1", 136, 1.64, 0),
               ("gemini:gemini-2.5-pro:high:wire1", 42, 1.72, 30),
               fallbacks="bear:failed 10, pm:hold 2, trader:failed 8")
    runs = [_run(f"t{i}", [s], desk="v2_claude", out=out) for i, (s, _) in enumerate(O4)]
    a = _build(runs)["agent"]
    assert a["model"] == "gemini:gemini-2.5-flash:medium:wire1 + gemini:gemini-2.5-pro:high:wire1"
    assert a["calls"] == 4 * 178 and a["cost_usd"] == pytest.approx(4 * 3.36)
    assert a["cache_hits"] == 4 * 30 and a["fallbacks"] == 4 * 20


def test_a_run_without_a_spend_line_is_refused():
    """A run with no spend line did not finish; a cost typed in by hand is the one number
    on the board nobody could check against the run."""
    with pytest.raises(re_.ReplayError, match="no spend line"):
        _run("t0", [O4[0][0]], out="Traceback (most recent call last):\n")


def test_the_entry_carries_metrics_only_never_decisions_or_reasoning():
    """The decision log holds the model's rationale and every weight it chose. The board
    is public, and an entry is ranked on metrics alone."""
    log = pd.DataFrame([{"source": "brain", "decision": {"rationale": "SECRET-REASONING",
                                                          "weights": [{"name": "GS", "weight": 0.15}]}},
                        {"source": "fallback", "decision": {}}])
    runs = [_run(f"t{i}", [s], log=log) for i, (s, _) in enumerate(O4)]
    e = _build(runs)
    text = json.dumps(e)
    assert "SECRET-REASONING" not in text and '"GS"' not in text
    assert set(e["windows"][0]) == {"window_start", "window_end", *M}
    assert e["agent"]["fallbacks"] == 4


def test_a_suite_the_board_has_no_references_for_is_refused():
    board = lb.boards([_holdout_ref("cash")])
    with pytest.raises(re_.ReplayError, match="no official4 references"):
        re_.build_entry(_four(), "official4", board, name="v1", desk="d", window_choice="w",
                        snapshot="s", author="a")
