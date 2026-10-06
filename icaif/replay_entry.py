"""A leaderboard entry from paid agent replays: metrics and the run's description only.

`agent_replay` writes, per run, `output/agent/<tag>/windows.csv` (every strategy's four
metrics in each window it ran) and a console log `<tag>.out` ending in one spend line per
model it called. An entry is built from those two and nothing else: never the decisions
(`log.jsonl`) or the model's reasoning, which are ours and are not what a board ranks.

A suite's entry may need several runs, one per window, as the Earnings season's four
were run. Combining them is where a wrong-but-plausible entry comes from, so each of
these refuses rather than submits:

- **A window missing or run twice.** A fixed suite ranks only an entry covering exactly
  its windows; three of four would be named "not ranked" on the board after the upload,
  and a duplicated window means two runs of one window, either of which is a guess.
- **Runs of different desks or models.** Four windows from two desks, or from Pro in one
  and Flash in another, rank as one method that never existed.
- **No spend line.** A run without one did not finish, and a cost typed by hand is the
  one number on the board nobody could check against the run.
- **A window priced differently from the board.** The replay simulates on our market,
  the board's references on the scorer's snapshot; `inv_vol_hold_75` is in both, and
  must match to `boardrank.ANCHOR_TOL` (`boardrank.check_anchor`), or every place in that
  window is read against a field priced on other data.

Window ends come from the board itself (the suite's references), never computed here:
an end one session off is a different window, which the board would exclude.
"""

import re
from dataclasses import dataclass, field

import pandas as pd

from icaif import boardrank, leaderboard, ranking, suites

METRICS = list(ranking.METRICS)
# The desk row in a replay's windows.csv: agent_replay names it desk_*, free_* or v2_*;
# the other rows are its baselines (the hold, the rule), already board strategies.
DESK_PREFIXES = ("desk_", "free_", "v2_")
_SPEND = re.compile(r"^(\S+): (\d+) calls, \$([\d.]+); cache hits (\d+), misses (\d+)", re.M)
_FALLBACKS = re.compile(r"^fallbacks: (.*)$", re.M)


class ReplayError(ValueError):
    """The runs cannot make one honest entry; the message names what is wrong."""


@dataclass
class Run:
    tag: str
    desk: str                     # the desk's strategy name in windows.csv
    rows: pd.DataFrame            # every strategy's rows: window, strategy, metrics
    spend: list = field(default_factory=list)   # [{"model", "calls", "cost_usd", "cache_hits"}]
    fallbacks: int = 0

    @property
    def models(self) -> tuple:
        return tuple(sorted(s["model"] for s in self.spend))


def spend(tag: str, out_text: str) -> list[dict]:
    """Every model's spend line from a replay's console log. A v2 desk calls two models
    (quick and deep roles) and prints a line for each; reading only the first would
    report about half the cost."""
    lines = [{"model": m[0], "calls": int(m[1]), "cost_usd": float(m[2]), "cache_hits": int(m[3])}
             for m in _SPEND.findall(out_text)]
    if not lines:
        raise ReplayError(f"{tag}: no spend line in its log; was the replay paid, and did it finish?")
    return lines


def fallbacks(out_text: str, log: pd.DataFrame | None) -> int:
    """Answers replaced by a fallback. v2 prints a per-role tally; v1 marks each one in its
    decision log as source "fallback" and prints none."""
    m = _FALLBACKS.findall(out_text)
    if m:
        return sum(int(n) for n in re.findall(r" (\d+)(?:,|$)", m[-1]))
    if log is None or log.empty or "source" not in log:
        return 0
    return int((log["source"] == "fallback").sum())


def read_run(tag: str, windows: pd.DataFrame, out_text: str, log: pd.DataFrame | None) -> Run:
    windows = windows.assign(window=windows["window"].astype(str))
    desks = sorted(s for s in windows["strategy"].unique() if s.startswith(DESK_PREFIXES))
    if len(desks) != 1:
        raise ReplayError(f"{tag}: expected one agent desk in windows.csv, found {desks}")
    return Run(tag, desks[0], windows, spend(tag, out_text), fallbacks(out_text, log))


def suite_board(standing: dict, suite: str) -> dict:
    """The suite's standings from `leaderboard.boards`: what the entry will be ranked on."""
    if suite == suites.DEFAULT:
        return standing
    if suite not in standing.get("suites", {}):
        raise ReplayError(f"the board has no {suite} references; deploy the board first")
    return standing["suites"][suite]


def build_entry(runs: list[Run], suite: str, standing: dict, *, name: str, desk: str,
                window_choice: str, snapshot: str, author: str, note: str = "",
                code: dict | None = None, submitted_at: str | None = None) -> dict:
    """One agentic entry for `suite` from `runs`, after every check in the module docstring.

    `standing` is `leaderboard.boards` of the board's live entries.
    """
    definition = suites.get(suite)
    board = suite_board(standing, definition.name)
    by_start = {w["window_start"]: w for w in board["by_window"]}

    if len({r.desk for r in runs}) != 1:
        raise ReplayError("the runs are of different desks: "
                          + ", ".join(f"{r.tag} {r.desk}" for r in runs))
    if len({r.models for r in runs}) != 1:
        raise ReplayError("the runs called different models: "
                          + ", ".join(f"{r.tag} {' + '.join(r.models)}" for r in runs))

    rows, seen = [], {}
    for r in runs:
        mine = r.rows[r.rows["strategy"] == r.desk]
        for w, g in mine.groupby("window", sort=True):
            if w in seen:
                raise ReplayError(f"window {w} is in both {seen[w]} and {r.tag}")
            seen[w] = r.tag
            if w not in by_start:
                raise ReplayError(f"{r.tag}: window {w} is not one of the {definition.name} "
                                  "board's windows")
            boardrank.check_anchor(r.rows[r.rows["window"] == w], by_start[w])
            # Invalid rounds held, as the backend would hold them; the board shows the count
            # beside the entry, so dropping it would pass a desk that broke the rules as clean.
            rows.append({"window_start": w, "window_end": by_start[w]["window_end"],
                         **{k: float(g.iloc[0][k]) for k in METRICS},
                         "invalid_rounds": int(g.iloc[0].get("invalid_rounds", 0) or 0)})
    if definition.fixed:
        missing = [s for s, _ in definition.windows if s not in seen]
        if missing:
            raise ReplayError(f"suite {definition.name} needs all {len(definition.windows)} "
                              f"windows; no run covers {', '.join(missing)}")

    lines = [s for r in runs for s in r.spend]
    models = runs[0].models
    agent = {
        "model": " + ".join(models), "desk": desk,
        "calls": sum(s["calls"] for s in lines),
        "cost_usd": round(sum(s["cost_usd"] for s in lines), 4),
        # A cached answer is not paid again, so the cost covers only the misses.
        "cache_hits": sum(s["cache_hits"] for s in lines),
        "fallbacks": sum(r.fallbacks for r in runs),
        "window_choice": window_choice, "replay": [r.tag for r in runs],
        **({"code": code} if code else {}),
    }
    wins = pd.DataFrame(sorted(rows, key=lambda x: x["window_start"]))
    return leaderboard.make_entry(
        name, leaderboard.AGENTIC, wins,
        span=tuple(definition.span) if definition.fixed
        else (wins["window_start"].min(), wins["window_end"].max()),
        sizing=leaderboard.BOARD_SIZING, market_snapshot=snapshot, author=author, note=note,
        agent=agent, suite=definition.name, submitted_at=submitted_at)
