"""The holdout leaderboard: prospective strategies ranked the way the contest ranks them.

The contest scores one 15-day window by ranking every entrant on the four metrics.
So the board ranks all its entries against each other in every rolling 15-day window
of the holdout (`ranking.rank_window`, the rules' own tie handling), then orders them
by mean Overall Rank Score across windows. Lower is better. Two consequences are
deliberate:

- A rank is relative to the board, as in the real field. Adding an entry can reorder
  the others. A strategy that beats cash and equal weight is not thereby good, but one
  that loses to them is not worth entering.
- Continuous-run metrics are shown beside the rank but never decide it. The contest
  never scores an eight-month curve.

An entry is a result, not a decisions file: its continuous metrics and its per-window
metrics, from `holdout`. Only the newest version of each strategy name ranks. Older
versions are kept and counted, so the number of looks at the holdout stays visible.
Every look is a chance to tune against it.
"""

import re
from datetime import datetime, timezone

import pandas as pd

from icaif import ranking

SCHEMA = 1
METRICS = list(ranking.METRICS)
BOARD_SIZING = "pre_fee"
REFERENCE = "reference"
SUBMITTED = "submitted"


def independent_windows(windows_df: pd.DataFrame) -> int:
    """How many of the windows could be picked without any two sharing a day."""
    n, last_end = 0, ""
    for s, e in zip(windows_df["window_start"], windows_df["window_end"]):
        if s > last_end:
            n, last_end = n + 1, e
    return n


HIST_BINS = 16


def histogram_edges(field: dict) -> dict:
    """Per metric, one set of bin edges shared by every ranked entry.

    Shared, not per row: the histograms are read against each other down a column, and
    per-row edges would draw a tight and a wide distribution as the same shape.
    """
    edges = {}
    for k in METRICS:
        vals = [w[k] for e in field.values() for w in e["windows"]]
        lo, hi = min(vals), max(vals)
        if hi == lo:   # cash: every window 0. One bin centred on the value, not a crash.
            lo, hi = lo - 0.5e-3, hi + 0.5e-3
        step = (hi - lo) / HIST_BINS
        edges[k] = [lo + i * step for i in range(HIST_BINS)] + [hi]
    return edges


def histogram(entry: dict, metric: str, edges: list[float]) -> list[int]:
    counts = [0] * (len(edges) - 1)
    for w in entry["windows"]:
        i = sum(w[metric] >= e for e in edges[1:-1])   # right-open bins, last one closed
        counts[i] += 1
    return counts


class EntryError(ValueError):
    pass


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not s:
        raise EntryError(f"strategy name {name!r} has no usable characters")
    return s


def make_entry(strategy: str, kind: str, summary: dict, windows_df: pd.DataFrame, *,
               span: tuple, sizing: str, market_snapshot: str, author: str = "",
               note: str = "", decisions_sha256: str | None = None,
               submitted_at: str | None = None) -> dict:
    return {
        "schema": SCHEMA, "kind": kind, "strategy": strategy, "author": author,
        "note": note,
        "submitted_at": submitted_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "decisions_sha256": decisions_sha256, "sizing": sizing,
        "span": [str(span[0]), str(span[1])], "market_snapshot": market_snapshot,
        "continuous": {k: float(summary[k]) for k in METRICS},
        "missing_rounds": int(summary.get("missing_rounds", 0)),
        "invalid_rounds": int(summary.get("invalid_rounds", 0)),
        "windows": [{"window_start": r["window_start"], "window_end": r["window_end"],
                     **{k: float(r[k]) for k in METRICS}}
                    for r in windows_df.to_dict(orient="records")],
    }


def standings(entries: list[dict]) -> dict:
    """Rank the newest version of every strategy, plus the references, window by window.

    An entry that cannot be compared is left off and listed with the reason, never
    ranked on a subset of windows. That covers a different span or sizing, or a window
    set that differs from the references'. In a subset of windows every entrant would
    face a different field, and the mean scores would stop being comparable.
    """
    refs = [e for e in entries if e.get("kind") == REFERENCE]
    if not refs:
        raise EntryError("no reference entries; the board has nothing to anchor its windows")
    board_span = refs[0]["span"]
    board_windows = [w["window_start"] for w in refs[0]["windows"]]
    n_independent = max(1, independent_windows(pd.DataFrame(refs[0]["windows"])))

    ref_names = {r["strategy"] for r in refs}
    latest, versions, excluded = {}, {}, []
    for e in entries:
        name = e["strategy"]
        if e.get("schema") != SCHEMA:
            excluded.append({"strategy": name, "reason": f"schema {e.get('schema')}"})
            continue
        if e["kind"] == SUBMITTED and name in ref_names:
            # Checked before `latest`: sharing the name key, a submission would otherwise
            # replace the anchor every other entry is read against.
            excluded.append({"strategy": name, "reason": "uses a reference strategy's name"})
            continue
        if e["kind"] == SUBMITTED:
            versions[name] = versions.get(name, 0) + 1
            if name in latest and latest[name]["submitted_at"] >= e["submitted_at"]:
                continue
        elif name in latest:
            raise EntryError(f"two reference entries named {name!r}")
        latest[name] = e

    field = {}
    for name, e in latest.items():
        why = None
        if e["span"] != board_span:
            why = f"span {e['span']} is not the board's {board_span}"
        elif e["sizing"] != BOARD_SIZING:
            why = f"sizing {e['sizing']} is not the board's {BOARD_SIZING}"
        elif [w["window_start"] for w in e["windows"]] != board_windows:
            why = "its windows differ from the board's"
        if why:
            excluded.append({"strategy": name, "reason": why})
        else:
            field[name] = e

    per_window = []
    for i, start in enumerate(board_windows):
        m = pd.DataFrame({n: e["windows"][i] for n, e in field.items()}).T[METRICS].astype(float)
        # Rounded as rankplay does: float dust must not split what the kit's Decimals tie.
        r = ranking.rank_window(m.round(12))
        per_window.append(r[["overall_score", "position"]].assign(window=start))
    ranks = pd.concat(per_window).rename_axis("strategy").reset_index()
    g = ranks.groupby("strategy")

    rows = []
    for name, e in field.items():
        scores = g.get_group(name)["overall_score"]
        rows.append({
            "strategy": name, "kind": e["kind"], "author": e["author"], "note": e["note"],
            "submitted_at": e["submitted_at"] if e["kind"] == SUBMITTED else "",
            "versions": versions.get(name, 0),
            "mean_overall_score": float(scores.mean()),
            # About 11 independent windows stand behind ~150 overlapping ones, so the SE
            # divides by the independent count. Dividing by the overlapping count would
            # claim ~3.7x more precision than the data holds.
            "se_overall_score": float(scores.std(ddof=1) / n_independent ** 0.5)
            if len(scores) > 1 else 0.0,
            "mean_position": float(g.get_group(name)["position"].mean()),
            "wins": int((g.get_group(name)["position"] == 1).sum()),
            "missing_rounds": e["missing_rounds"], "invalid_rounds": e["invalid_rounds"],
            **{f"continuous_{k}": e["continuous"][k] for k in METRICS},
            **{f"median_window_{k}": float(pd.Series([w[k] for w in e["windows"]]).median())
               for k in METRICS},
        })
    edges = histogram_edges(field)
    for r in rows:
        r["hist"] = {k: histogram(field[r["strategy"]], k, edges[k]) for k in METRICS}
    rows.sort(key=lambda r: (r["mean_overall_score"], -r["continuous_cumulative_return"]))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    history = sorted(
        ({"strategy": e["strategy"], "author": e["author"], "submitted_at": e["submitted_at"],
          "ranked": latest.get(e["strategy"]) is e and e["strategy"] in field,
          **{f"continuous_{k}": e["continuous"][k] for k in METRICS}}
         for e in entries if e.get("kind") == SUBMITTED),
        key=lambda h: h["submitted_at"], reverse=True)
    return {"span": board_span, "sizing": BOARD_SIZING, "windows": len(board_windows),
            "bins": edges,
            "independent_windows": n_independent,
            "entrants": len(rows), "rows": rows, "excluded": excluded, "history": history}
