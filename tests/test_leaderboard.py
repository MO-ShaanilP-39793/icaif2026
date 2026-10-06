import pandas as pd
import pytest

from icaif import leaderboard as lb
from icaif import ranking, suites

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



def test_a_picked_window_shows_the_same_ranks_the_board_score_averages():
    """The by-window view is read as "how the contest would have scored this window". If its
    places came from anywhere but the ranks behind the score, a strategy could read first in
    every window it is picked in and still sit low on the board, with nothing to say why."""
    a = _entry("a", [(0.02, 2.0, 0.01, 0.05), (-0.01, -1.0, 0.03, 0.05), (0.03, 3.0, 0.01, 0.05)])
    b = lb.standings([CASH, EW, a])
    assert [(w["window_start"], w["window_end"]) for w in b["by_window"]] == WINDOWS
    for i, w in enumerate(b["by_window"]):
        m = pd.DataFrame({e["strategy"]: e["windows"][i] for e in (CASH, EW, a)}).T
        want = ranking.rank_window(m[lb.METRICS].astype(float))
        assert [r["position"] for r in w["rows"]] == sorted(want["position"])
        for r in w["rows"]:
            assert r["position"] == want.loc[r["strategy"], "position"]
            assert r["overall_score"] == pytest.approx(want.loc[r["strategy"], "overall_score"])
            assert r["cumulative_return"] == m.loc[r["strategy"], "cumulative_return"]
    for row in b["rows"]:
        scores = [r["overall_score"] for w in b["by_window"] for r in w["rows"]
                  if r["strategy"] == row["strategy"]]
        assert row["mean_overall_score"] == pytest.approx(sum(scores) / len(scores))

# ----------------------------------------------------------------------------- agentic panel

AGENT = {"model": "grok-4.7", "desk": "free", "calls": 15, "cost_usd": 1.15,
         "window_choice": "picked for its earnings"}


def _agentic(name, windows, metrics, at="2026-10-05T10:00:00Z", agent=AGENT):
    wins = pd.DataFrame([{"window_start": s, "window_end": e, **dict(zip(lb.METRICS, m))}
                         for (s, e), m in zip(windows, metrics)])
    return lb.make_entry(name, lb.AGENTIC, wins, span=windows[0], sizing="pre_fee",
                         market_snapshot="snap", author="a", submitted_at=at, agent=agent)


def test_an_agentic_entry_never_moves_the_main_board_and_never_meets_another_agent():
    """A one-window agent beside 109-window means would read as the same kind of number,
    and two agents ranked against each other would each move the other's place."""
    a = _entry("a", [(0.02, 2.0, 0.01, 0.05)] * 3)
    g1 = _agentic("grok_free", [WINDOWS[1]], [(0.06, 10.0, 0.01, 0.03)])
    g2 = _agentic("grok_levered", [WINDOWS[1]], [(0.10, 12.0, 0.0, 0.0)])
    base = lb.standings([CASH, EW, a])
    with_agents = lb.standings([CASH, EW, a, g1, g2])
    assert with_agents["rows"] == base["rows"] and with_agents["excluded"] == base["excluded"]
    panel = {r["strategy"]: r for r in with_agents["agentic"]["rows"]}
    alone = lb.standings([CASH, EW, a, g1])["agentic"]["rows"][0]
    assert panel["grok_free"]["windows"] == alone["windows"]          # g2 changed nothing
    assert panel["grok_free"]["windows"][0]["of"] == 4                 # the field + itself


def test_an_agentic_place_is_its_rank_against_the_main_field_in_that_window():
    a = _entry("a", [(0.02, 2.0, 0.01, 0.05)] * 3)
    g = _agentic("grok", [WINDOWS[2]], [(0.03, 3.0, 0.0, 0.0)])
    w = lb.standings([CASH, EW, a, g])["agentic"]["rows"][0]["windows"][0]
    m = pd.DataFrame({e["strategy"]: e["windows"][2] for e in (CASH, EW, a)}).T[lb.METRICS]
    m.loc["grok"] = {k: g["windows"][0][k] for k in lb.METRICS}
    want = ranking.rank_window(m.astype(float))
    assert w["position"] == want.loc["grok", "position"]
    assert w["overall_score"] == pytest.approx(want.loc["grok", "overall_score"])


def test_an_agentic_entry_off_the_boards_windows_or_without_its_story_is_named_not_ranked():
    """A window the board lacks has no field to rank in; a place without the model, cost
    and how the window was chosen invites a reading the run cannot support."""
    off = _agentic("off", [("2026-01-03", "2026-01-24")], [(0.01, 1.0, 0.0, 0.0)])
    bare = _agentic("bare", [WINDOWS[0]], [(0.01, 1.0, 0.0, 0.0)])
    bare["agent"] = {k: v for k, v in AGENT.items() if k != "window_choice"}
    clash = _agentic("ew_hold", [WINDOWS[0]], [(0.01, 1.0, 0.0, 0.0)])
    p = lb.standings([CASH, EW, off, bare, clash])["agentic"]
    assert p["rows"] == []
    why = {e["strategy"]: e["reason"] for e in p["excluded"]}
    assert "not on the board" in why["off"] and "window_choice" in why["bare"]
    assert "main-board" in why["ew_hold"]
    with pytest.raises(lb.EntryError):
        _agentic("x", [WINDOWS[0]], [(0, 0, 0, 0)], agent={"model": "m"})
    with pytest.raises(lb.EntryError):
        _entry("y", [(0, 0, 0, 0)] * 3, agent=AGENT)

# ----------------------------------------------------------------------------------- suites

O4 = [tuple(w) for w in suites.SUITES["official4"].windows]
O4_SPAN = suites.SUITES["official4"].span


def _o4(name, per_window, kind=lb.SUBMITTED, windows=O4, **kw):
    wins = pd.DataFrame([{"window_start": s, "window_end": e, **dict(zip(lb.METRICS, m))}
                         for (s, e), m in zip(windows, per_window)])
    args = dict(span=O4_SPAN, sizing="pre_fee", market_snapshot="snap", author="a",
                submitted_at="" if kind == lb.REFERENCE else "2026-10-06T10:00:00Z",
                suite="official4")
    args.update(kw)
    return lb.make_entry(name, kind, wins, **args)


O4_CASH = _o4("cash", [(0, 0, 0, 0)] * 4, kind=lb.REFERENCE)
O4_EW = _o4("ew_hold", [(0.02, 2.0, 0.03, 0.01)] * 4, kind=lb.REFERENCE)


def test_an_entry_without_a_suite_ranks_on_the_holdout_exactly_as_before():
    """Every entry on the board predates suites. Read as belonging to none, they would
    all drop off the holdout board the day suites shipped."""
    a = _entry("a", [(0.02, 2.0, 0.01, 0.05), (-0.01, -1.0, 0.03, 0.05), (0.03, 3.0, 0.01, 0.05)])
    legacy = [{k: v for k, v in e.items() if k != "suite"} for e in (CASH, EW, a)]
    assert lb.standings(legacy)["rows"] == lb.standings([CASH, EW, a])["rows"]
    assert lb.boards(legacy)["rows"] == lb.standings([CASH, EW, a])["rows"]


def test_each_suite_ranks_only_its_own_entries_against_its_own_references():
    """Ranked together, an official4 entry and a holdout entry share one window at most:
    each would be excluded from the other's board for "other windows", or worse, push the
    other's places around in the window they share."""
    a = _entry("a", [(0.02, 2.0, 0.01, 0.05)] * 3)
    x = _o4("x", [(0.05, 5.0, 0.01, 0.02), (0.0, 0.0, 0.01, 0.02),
                  (0.01, 1.0, 0.0, 0.02), (-0.01, -1.0, 0.02, 0.02)])
    b = lb.boards([CASH, EW, a, O4_CASH, O4_EW, x])
    o4 = b["suites"]["official4"]

    assert {r["strategy"] for r in b["rows"]} == {"cash", "ew_hold", "a"}
    assert {r["strategy"] for r in o4["rows"]} == {"cash", "ew_hold", "x"}
    assert b["rows"] == lb.standings([CASH, EW, a])["rows"]
    assert b["excluded"] == [] and o4["excluded"] == []
    assert o4["suite"]["name"] == "official4" and o4["windows"] == 4
    for i, w in enumerate(o4["by_window"]):
        m = pd.DataFrame({e["strategy"]: e["windows"][i] for e in (O4_CASH, O4_EW, x)}).T
        want = ranking.rank_window(m[lb.METRICS].astype(float))
        assert {r["strategy"]: r["position"] for r in w["rows"]} == want["position"].to_dict()


def test_a_suite_entry_covering_other_windows_is_named_not_ranked():
    """An official4 entry run a day late in one window, or on three of the four, faces a
    field that played other markets. Ranked, its mean would read as the suite's."""
    late = _o4("late", [(0.01, 1.0, 0.0, 0.0)] * 4,
               windows=[O4[0], ("2025-10-14", "2025-11-03"), *O4[2:]])
    three = _o4("three", [(0.01, 1.0, 0.0, 0.0)] * 3, windows=O4[:3])
    o4 = lb.boards([CASH, EW, O4_CASH, O4_EW, late, three])["suites"]["official4"]
    assert {r["strategy"] for r in o4["rows"]} == {"cash", "ew_hold"}
    why = {e["strategy"]: e["reason"] for e in o4["excluded"]}
    assert set(why) == {"late", "three"} and "2025-10-14" in why["late"]


def test_references_built_for_another_definition_of_a_fixed_suite_refuse_to_rank():
    """If the references were scored on windows the suite no longer names, every correct
    entry would be excluded and the references would rank alone, looking like a board."""
    stale = [_o4(e["strategy"], [(0, 0, 0, 0)] * 4, kind=lb.REFERENCE,
                 windows=[("2025-04-14", "2025-05-05"), *O4[1:]]) for e in (O4_CASH, O4_EW)]
    with pytest.raises(lb.EntryError, match="rebuild the board"):
        lb.standings(stale, "official4")


def test_an_entry_naming_a_suite_the_board_lacks_is_named_not_dropped():
    """A submission from a newer scorer, or one whose suite has no references here, would
    otherwise vanish: on no board, in no list, with its author waiting for it to show."""
    new = {**_entry("future", [(0.01, 1.0, 0.0, 0.0)] * 3), "suite": "official9"}
    orphan = _o4("orphan", [(0.01, 1.0, 0.0, 0.0)] * 4)
    b = lb.boards([CASH, EW, new, orphan])
    why = {e["strategy"]: e["reason"] for e in b["excluded"]}
    assert "official9" in why["future"] and "no references" in why["orphan"]
    assert b["suites"] == {}
    with pytest.raises(lb.EntryError, match="official9"):
        _entry("y", [(0, 0, 0, 0)] * 3, suite="official9")
