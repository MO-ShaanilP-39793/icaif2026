"""Stage 1's decision rules (tools/doe_report.py, stage1_doe.md)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import doe_report as R  # noqa: E402
from icaif.agents.v3 import STREAMS, STREAMS16  # noqa: E402

WINDOWS = [f"2025-{m:02d}-01" for m in range(1, 13)] + ["2025-12-15"]   # 13 windows


def _runs(effects: dict, noise=0.0, seed=0) -> dict:
    """Per-window scores for the 16 rows: a market swing every arm shares, plus each
    included stream's planted effect, plus optional arm-level noise."""
    rng = np.random.default_rng(seed)
    swing = pd.Series(rng.normal(3.0, 0.8, len(WINDOWS)), index=WINDOWS)
    out = {}
    for r, streams in enumerate(STREAMS16, 1):
        planted = sum(effects.get(s, 0.0) for s in streams)
        out[r] = swing + planted + rng.normal(0, noise, len(WINDOWS))
    return out


def test_the_factorial_recovers_each_planted_stream_effect_exactly_and_no_other():
    """Orthogonal columns: with no noise, each stream's estimate is its planted effect and
    the others read exactly zero, whatever the shared market swing."""
    planted = {"model_rank": -0.4, "headlines": 0.3}
    eff = R.stream_effects(_runs(planted), {r: set(s) for r, s in enumerate(STREAMS16, 1)})
    for s in STREAMS:
        assert eff.loc[s, "effect"] == pytest.approx(planted.get(s, 0.0), abs=1e-12)


def test_a_stream_is_kept_only_when_it_helps_by_more_than_one_se():
    """A tie drops the stream: one that can't show its worth isn't earning its prompt."""
    eff = pd.DataFrame({"effect": [-0.30, -0.04, 0.02, 0.20, -0.05],
                        "se": [0.10, 0.05, 0.05, 0.05, np.nan]},
                       index=["model_rank", "macro", "regime", "headlines", "universe"])
    assert R.keep_streams(eff) == ["model_rank"]


def test_a_cheaper_architecture_within_one_se_of_the_best_is_chosen():
    s = pd.Series([3.0, 3.2, 2.8, 3.1], index=list("abcd"))
    scores = {"reports_raw": s - 0.05, "none": s + np.array([0.2, -0.2, 0.1, -0.1]), "reports_only": s + 0.5}
    pick, why = R.choose_architecture(scores, R.COST)
    assert pick == "none" and "cheaper" in why
    clear = {"reports_raw": s - 0.6, "none": s, "reports_only": s + 0.5}
    assert R.choose_architecture(clear, R.COST)[0] == "reports_raw"


def test_a_cash_floor_is_chosen_only_when_it_beats_no_floor_by_more_than_one_se():
    s = pd.Series([3.0, 3.2, 2.8, 3.1, 2.9], index=list("abcde"))
    tied = {0.0: s, 0.4: s - np.array([0.1, -0.1, 0.05, -0.02, 0.0])}
    assert R.choose_floor(tied)[0] == 0.0
    clear = {0.0: s, 0.4: s - 0.5, 0.9: s - 0.2}
    assert R.choose_floor(clear)[0] == 0.4


def test_book_agreement_reads_shares_not_gross():
    """A capped book is the same names at a lower gross; it must read as identical."""
    a = {"X": 0.3, "Y": 0.2, "Z": 0.1}
    half = {k: v / 2 for k, v in a.items()}
    got = R.book_agreement(a, half)
    assert got["jaccard"] == 1.0 and got["share_corr"] == pytest.approx(1.0)
    other = R.book_agreement(a, {"X": 0.1, "W": 0.4})
    assert other["jaccard"] == pytest.approx(1 / 4)


def test_with_too_few_windows_to_measure_a_difference_the_simpler_choice_stands():
    """One window has no SE; a rule reading a NaN SE as "beaten by more than one SE"
    would crown whatever won that window."""
    one = {0.0: pd.Series([4.1], index=["a"]), 0.75: pd.Series([3.2], index=["a"])}
    assert R.choose_floor(one)[0] == 0.0
    arch = {"reports_raw": pd.Series([3.0], index=["a"]), "none": pd.Series([3.6], index=["a"])}
    assert R.choose_architecture(arch, R.COST)[0] == "none"
