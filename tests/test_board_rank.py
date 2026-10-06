import json
import urllib.error

import pandas as pd
import pytest

from icaif import boardrank as br
from icaif import leaderboard as lb

WINDOWS = [("2026-01-02", "2026-01-23"), ("2026-01-05", "2026-01-26"), ("2026-01-26", "2026-02-13")]
SPAN = ("2026-01-02", "2026-06-30")
HOLD = [(0.015, 3.1, 0.014, 0.007), (-0.02, -2.0, 0.03, 0.007), (0.01, 1.5, 0.01, 0.007)]
RULE = [(0.026, 5.1, 0.013, 0.008), (-0.03, -3.0, 0.035, 0.008), (0.02, 2.5, 0.02, 0.008)]


def _entry(name, per_window, kind=lb.SUBMITTED, **kw):
    wins = pd.DataFrame([{"window_start": s, "window_end": e, **dict(zip(lb.METRICS, m))}
                         for (s, e), m in zip(WINDOWS, per_window)])
    args = dict(span=SPAN, sizing="pre_fee", market_snapshot="snap", author="a",
                submitted_at="2026-09-30T10:00:00Z")
    args.update(kw)
    return lb.make_entry(name, kind, wins, **args)


def _board(extra=()):
    entries = [_entry("cash", [(0, 0, 0, 0)] * 3, kind=lb.REFERENCE, submitted_at=""),
               _entry("ew_hold", [(0.01, 1.0, 0.02, 0.01)] * 3, kind=lb.REFERENCE, submitted_at=""),
               _entry(br.ANCHOR, HOLD, kind=lb.REFERENCE, submitted_at=""),
               _entry("q_riskparity_entry_regime", RULE),
               _entry("b", [(0.03, 4.0, 0.02, 0.2), (0.0, 0.1, 0.01, 0.2), (0.005, 0.5, 0.02, 0.2)]),
               *extra]
    return lb.standings(entries)


def _replay(window, desk, hold=None, rule=None):
    i = [s for s, _ in WINDOWS].index(window)
    rows = {br.ANCHOR: hold or HOLD[i], "q_riskparity_entry_regime": rule or RULE[i], "v2_claude": desk}
    return pd.DataFrame([{"window": window, "strategy": n, **dict(zip(lb.METRICS, m))}
                         for n, m in rows.items()])


def test_a_replay_priced_differently_from_the_board_is_refused_before_any_place_is_read():
    """The replay and the board score on separate market snapshots. If they disagree on a
    window, every place in it is read against a field priced on other data, and the table
    still looks like a ranking. The hold is in both, so it has to match first."""
    off = (HOLD[0][0] + 1e-5, *HOLD[0][1:])
    with pytest.raises(br.AnchorMismatch, match="priced this window differently"):
        br.rank_replay(_replay("2026-01-02", (0.05, 9.0, 0.01, 0.03), hold=off), _board())

    no_anchor = _replay("2026-01-02", (0.05, 9.0, 0.01, 0.03))
    no_anchor = no_anchor[no_anchor["strategy"] != br.ANCHOR]
    with pytest.raises(br.AnchorMismatch):
        br.rank_replay(no_anchor, _board())


def test_a_replay_place_is_the_place_the_agentic_panel_would_give_it():
    """The report and the panel must not disagree on one run: a place computed a second
    way would drift from the board's tie handling, and the gate would read one number
    while the panel showed another."""
    desk = (0.02, 3.5, 0.012, 0.05)
    agent = lb.make_entry("v2_claude", lb.AGENTIC, pd.DataFrame([{
        "window_start": WINDOWS[1][0], "window_end": WINDOWS[1][1], **dict(zip(lb.METRICS, desk))}]),
        span=WINDOWS[1], sizing="pre_fee", market_snapshot="snap",
        agent={k: "x" for k in lb.AGENT_FIELDS})
    panel = _board([agent])["agentic"]["rows"][0]["windows"][0]

    got, off = br.rank_replay(_replay(WINDOWS[1][0], desk), _board())
    row = got.set_index("strategy").loc["v2_claude"]

    assert off == []
    assert (row["position"], row["of"], row["overall_score"]) == (
        panel["position"], panel["of"], panel["overall_score"])
    assert row["source"] == "placed"


def test_a_strategy_already_on_the_board_takes_its_board_place_and_never_ties_with_its_copy():
    """The rule was submitted to the board. Inserted beside its own entry it would tie with
    itself and push every entrant behind it down one, so a match is read off the board;
    a mismatch is placed under another name and said to differ, since it is another run."""
    b = _board()
    got, _ = br.rank_replay(_replay("2026-01-02", (0.05, 9.0, 0.01, 0.03)), b)
    by = got.set_index("strategy")
    board_row = next(r for r in b["by_window"][0]["rows"] if r["strategy"] == "q_riskparity_entry_regime")
    assert by.loc["q_riskparity_entry_regime", "source"] == "board"
    assert by.loc["q_riskparity_entry_regime", "position"] == board_row["position"]
    assert by.loc[br.ANCHOR, "source"] == "board"

    changed = (RULE[0][0] + 0.01, *RULE[0][1:])
    got, _ = br.rank_replay(_replay("2026-01-02", (0.05, 9.0, 0.01, 0.03), rule=changed), b)
    assert got.set_index("strategy").loc["q_riskparity_entry_regime", "source"].startswith("placed; differs")


def test_a_window_outside_the_board_is_listed_never_ranked_against_an_empty_field():
    """Jul-Sep windows have no board field. Ranked anyway, a desk alone in a window places
    first, and the evaluation would count it as beating the board."""
    r = _replay("2026-01-02", (0.05, 9.0, 0.01, 0.03)).assign(window="2026-07-01")
    got, off = br.rank_replay(r, _board())
    assert got.empty and off == ["2026-07-01"]


def test_the_gate_se_counts_windows_that_share_days_once():
    """Two windows 14 days apart share most of their days. Counted as two, the SE would
    claim more precision than the data holds, and a desk could pass the 2-SE gate on
    one window's luck seen twice."""
    b = _board()
    got = pd.concat([br.rank_replay(_replay(w, (0.05, 9.0, 0.01, 0.03)), b)[0]
                     for w, _ in WINDOWS], ignore_index=True)
    c = br.compare(got, br.ANCHOR).set_index("strategy")
    assert c.loc["v2_claude", "windows"] == 3
    assert c.loc["v2_claude", "independent"] == 2
    d = (got[got.strategy == "v2_claude"]["overall_score"].to_numpy()
         - got[got.strategy == br.ANCHOR]["overall_score"].to_numpy())
    assert c.loc["v2_claude", "se"] == pytest.approx(pd.Series(d).std() / 2 ** 0.5)


def test_the_board_is_read_as_its_page_reads_it_references_then_every_listed_entry():
    """The report must rank the field the page ranks. A missing index is an empty board,
    as on the page; any other failure stops the report rather than rank a partial field."""
    files = {"manifest.json": {"references": ["references/a.json"]},
             "references/a.json": {"strategy": "a"},
             "entries/index.json": {"entries": ["entries/b/1.json"]},
             "entries/b/1.json": {"strategy": "b"}}
    assert [e["strategy"] for e in br.fetch_entries(get=lambda p: files[p])] == ["a", "b"]

    def no_index(p):
        if p == "entries/index.json":
            raise urllib.error.HTTPError(p, 404, "not found", {}, None)
        return files[p]
    assert [e["strategy"] for e in br.fetch_entries(get=no_index)] == ["a"]

    def broken(p):
        if p == "entries/index.json":
            raise urllib.error.HTTPError(p, 503, "unavailable", {}, None)
        return files[p]
    with pytest.raises(urllib.error.HTTPError):
        br.fetch_entries(get=broken)
    json.dumps(files)   # the fixtures are what the page serves: plain JSON
