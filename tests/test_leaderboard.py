import pandas as pd
import pytest

from icaif import leaderboard as lb
from icaif import ranking

WINDOWS = [("2026-01-02", "2026-01-23"), ("2026-01-05", "2026-01-26"), ("2026-01-06", "2026-01-27")]
SPAN = ("2026-01-02", "2026-08-31")


def _entry(name, per_window, kind=lb.SUBMITTED, at="2026-09-30T10:00:00Z", **kw):
    """`per_window`: one (return, sharpe, mdd, turnover) tuple per window."""
    wins = pd.DataFrame([{"window_start": s, "window_end": e,
                          **dict(zip(lb.METRICS, m))} for (s, e), m in zip(WINDOWS, per_window)])
    args = dict(span=SPAN, sizing="pre_fee", market_snapshot="snap", author="a",
                submitted_at=at)
    args.update(kw)
    return lb.make_entry(name, kind, wins, **args)


CASH = _entry("cash", [(0, 0, 0, 0)] * 3, kind=lb.REFERENCE, submitted_at="")
EW = _entry("ew_hold", [(0.01, 1.0, 0.02, 0.01)] * 3, kind=lb.REFERENCE, submitted_at="")


def test_the_board_ranks_each_window_exactly_as_the_contest_rules_do():
    """The board's score must be the rules' score. A private ranking formula would order
    strategies by something the leaderboard never computes."""
    a = _entry("a", [(0.02, 2.0, 0.01, 0.05), (-0.01, -1.0, 0.03, 0.05), (0.03, 3.0, 0.01, 0.05)])
    got = {r["strategy"]: r for r in lb.standings([CASH, EW, a])["rows"]}

    for name in ("cash", "ew_hold", "a"):
        want = []
        for i in range(3):
            m = pd.DataFrame({e["strategy"]: e["windows"][i] for e in (CASH, EW, a)}).T
            want.append(ranking.rank_window(m[lb.METRICS].astype(float))
                        .loc[name, "overall_score"])
        assert got[name]["mean_overall_score"] == pytest.approx(sum(want) / 3)


def test_only_the_newest_version_ranks_and_every_version_is_counted():
    """Ranking every version would let one strategy's near-copies crowd the ranks of
    everything else. Dropping the old ones would hide how often the holdout was looked at."""
    old = _entry("a", [(0.05, 5.0, 0.0, 0.0)] * 3, at="2026-09-29T10:00:00Z")
    new = _entry("a", [(0.0, 0.0, 0.1, 0.5)] * 3, at="2026-09-30T10:00:00Z")
    b = lb.standings([CASH, EW, new, old])
    row = next(r for r in b["rows"] if r["strategy"] == "a")

    assert [r["strategy"] for r in b["rows"]].count("a") == 1
    assert row["versions"] == 2
    assert row["mean_window_cumulative_return"] == 0.0
    assert [h["status"] for h in b["history"]] == ["ranked", "superseded"]


def test_an_entry_on_different_windows_is_left_off_and_named_not_ranked_on_a_subset():
    """In a subset of windows an entrant faces a different field, so its mean score is not
    comparable. Ranking it anyway would print a clean number for a different contest."""
    short = _entry("short", [(0.01, 1.0, 0.01, 0.01)] * 2)
    other_span = _entry("span", [(0.01, 1.0, 0.01, 0.01)] * 3, span=("2026-02-02", "2026-08-31"))
    post = _entry("post", [(0.01, 1.0, 0.01, 0.01)] * 3, sizing="post_fee")
    b = lb.standings([CASH, EW, short, other_span, post])

    assert {r["strategy"] for r in b["rows"]} == {"cash", "ew_hold"}
    assert {e["strategy"] for e in b["excluded"]} == {"short", "span", "post"}


def test_a_submission_cannot_take_a_reference_strategys_name():
    """A submission named `cash` would shadow the anchor every other entry is read against."""
    fake = _entry("cash", [(0.05, 5.0, 0.0, 0.0)] * 3)
    b = lb.standings([CASH, EW, fake])
    assert [e["strategy"] for e in b["excluded"]] == ["cash"]
    assert next(r for r in b["rows"] if r["strategy"] == "cash")["kind"] == lb.REFERENCE


def test_float_dust_does_not_split_a_tie_the_kits_decimals_would_call():
    """Two strategies equal to 1e-15 are one tie under the kit's 40-digit Decimals. An
    unrounded float rank would split them and hand one a better score for nothing."""
    a = _entry("a", [(0.01, 1.0, 0.02, 0.01)] * 3)
    b_ = _entry("b", [(0.01 + 1e-15, 1.0, 0.02, 0.01)] * 3)
    rows = {r["strategy"]: r for r in lb.standings([CASH, EW, a, b_])["rows"]}
    assert rows["a"]["mean_overall_score"] == rows["b"]["mean_overall_score"]


def test_a_board_without_references_refuses_rather_than_ranking_submissions_alone():
    with pytest.raises(lb.EntryError, match="reference"):
        lb.standings([_entry("a", [(0, 0, 0, 0)] * 3)])


def test_every_rows_histogram_uses_the_same_bins_and_counts_every_window():
    """Per-row bins would draw a tight and a wide distribution as the same shape, so a
    column of histograms would compare nothing. A dropped window would understate a tail."""
    a = _entry("a", [(0.02, 2.0, 0.01, 0.05), (-0.01, -1.0, 0.03, 0.05), (0.03, 3.0, 0.01, 0.05)])
    b = lb.standings([CASH, EW, a])
    for k in lb.METRICS:
        edges = b["bins"][k]
        assert len(edges) == lb.HIST_BINS + 1 and edges == sorted(edges)
        for r in b["rows"]:
            assert sum(r["hist"][k]) == 3
    # The max lands in the last bin (closed on the right), the min in the first.
    ra = next(r for r in b["rows"] if r["strategy"] == "a")
    assert ra["hist"]["cumulative_return"][-1] == 1
    assert next(r for r in b["rows"] if r["strategy"] == "cash")["hist"]["turnover"][0] == 3


def test_the_non_overlapping_windows_carry_the_boards_own_ranks_and_no_shared_day():
    """A window sharing days with the one before it would count the same market twice in
    a table read as independent results. A rank recomputed on the subset alone would
    disagree with the score it sits beside, against the same field."""
    wins = [("2026-01-02", "2026-01-23"), ("2026-01-05", "2026-01-26"),
            ("2026-01-26", "2026-02-13"), ("2026-01-27", "2026-02-17")]

    def entry(name, per_window, kind=lb.SUBMITTED):
        df = pd.DataFrame([{"window_start": s, "window_end": e, **dict(zip(lb.METRICS, m))}
                           for (s, e), m in zip(wins, per_window)])
        return lb.make_entry(name, kind, df, span=SPAN,
                             sizing="pre_fee", market_snapshot="snap",
                             submitted_at="" if kind == lb.REFERENCE else "2026-09-30T10:00:00Z")

    cash = entry("cash", [(0, 0, 0, 0)] * 4, lb.REFERENCE)
    a = entry("a", [(0.02, 2.0, 0.01, 0.05), (-0.01, -1.0, 0.03, 0.05),
                    (0.03, 3.0, 0.01, 0.05), (-0.02, -2.0, 0.04, 0.05)])
    b = lb.standings([cash, a])
    row = next(r for r in b["rows"] if r["strategy"] == "a")

    # Window 3 starts on window 2's last day, so it is not disjoint from it; but window 2
    # overlaps window 1, so the tiling is windows 1 and 3.
    assert [w["window_start"] for w in row["disjoint"]] == ["2026-01-02", "2026-01-26"]
    assert len(row["disjoint"]) == b["independent_windows"]
    for w, i in zip(row["disjoint"], (0, 2)):
        m = pd.DataFrame({e["strategy"]: e["windows"][i] for e in (cash, a)}).T
        want = ranking.rank_window(m[lb.METRICS].astype(float)).loc["a"]
        assert (w["position"], w["overall_score"]) == (want["position"], want["overall_score"])
        assert w["cumulative_return"] == a["windows"][i]["cumulative_return"]
    assert row["mean_disjoint_overall_score"] == pytest.approx(
        sum(w["overall_score"] for w in row["disjoint"]) / 2)


def _old_format(entry):
    """A schema-1 entry as the board holds them: a six-month run replayed into windows."""
    return {**entry, "schema": 1, "continuous": dict.fromkeys(lb.METRICS, 0.0)}


def test_an_entry_scored_by_replaying_one_run_is_listed_but_never_ranked():
    """A replayed run scored a different strategy from the agent for anything that decides
    from its own book. Ranked beside per-window entries, the two would look alike."""
    old = _old_format(_entry("a", [(0.05, 5.0, 0.0, 0.0)] * 3))
    b = lb.standings([CASH, EW, old])

    assert {r["strategy"] for r in b["rows"]} == {"cash", "ew_hold"}
    assert [e["strategy"] for e in b["excluded"]] == ["a"]
    assert "resubmit" in b["excluded"][0]["reason"]
    assert [h["status"] for h in b["history"]] == ["old format"]


def test_a_resubmission_ranks_and_its_old_format_versions_still_count_as_looks():
    """Dropping old versions from the count would hide how often the holdout was looked
    at; still naming them as excluded would say the strategy is off the board when it is on it."""
    old = _old_format(_entry("a", [(0.05, 5.0, 0.0, 0.0)] * 3, at="2026-09-29T10:00:00Z"))
    new = _entry("a", [(0.01, 1.0, 0.01, 0.01)] * 3, at="2026-10-03T10:00:00Z")
    b = lb.standings([CASH, EW, old, new])
    row = next(r for r in b["rows"] if r["strategy"] == "a")

    assert row["versions"] == 2 and row["mean_window_cumulative_return"] == 0.01
    assert b["excluded"] == []
    assert [h["status"] for h in b["history"]] == ["ranked", "old format"]
