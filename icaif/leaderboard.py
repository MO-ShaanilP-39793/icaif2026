"""The holdout leaderboard: prospective strategies ranked the way the contest ranks them.

The contest scores one 15-day window by ranking every entrant on the four metrics.
So the board ranks all its entries against each other in every rolling 15-day window
of the holdout (`ranking.rank_window`, the rules' own tie handling), then orders them
by mean Overall Rank Score across windows. Lower is better. Two consequences are
deliberate:

- A rank is relative to the board, as in the real field. Adding an entry can reorder
  the others. A strategy that beats cash and equal weight is not thereby good, but one
  that loses to them is not worth entering.
- Every entrant, submitted or reference, starts every window from $1M in cash as
  itself. There is no six-month run: the contest never scores one, and showing one
  beside the rank made a buy-and-hold look like it held for six months.

An entry is a result, not a decisions file: its per-window metrics, from `holdout`. Only the newest version of each strategy name ranks. Older
versions are kept and counted, so the number of looks at the holdout stays visible.
Every look is a chance to tune against it.
"""

import re
from datetime import datetime, timezone

import pandas as pd

from icaif import ranking

# 2: every window run by the entrant from cash. Schema-1 entries replayed one six-month
# run into every window, which scored a different strategy for anything that decides
# from its own book; they are listed, never ranked beside schema 2.
SCHEMA = 2
METRICS = list(ranking.METRICS)
BOARD_SIZING = "pre_fee"
REFERENCE = "reference"
SUBMITTED = "submitted"


def disjoint_windows(windows: list[dict]) -> list[int]:
    """Indices of the windows that tile the span from its first day, no two sharing one.

    Greedy from the start: each window that begins after the last pick ended. These are
    the windows the board shows one by one, because neighbouring rolling windows share
    14 of 15 days and a table of all of them reads as ~100 results when it is ~8.
    """
    picked, last_end = [], ""
    for i, w in enumerate(windows):
        if w["window_start"] > last_end:
            picked.append(i)
            last_end = w["window_end"]
    return picked


def independent_windows(windows_df: pd.DataFrame) -> int:
    """How many of the windows could be picked without any two sharing a day."""
    return len(disjoint_windows(windows_df[["window_start", "window_end"]].to_dict("records")))


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


def _mean_window(entry: dict, metric: str) -> float:
    return float(pd.Series([w[metric] for w in entry["windows"]]).mean())


def _old_schema(entry: dict) -> str:
    if entry.get("schema") == 1:
        return "scored under the old format, one six-month run replayed into every window; resubmit"
    return f"schema {entry.get('schema')}"


def _status(entry: dict, latest: dict, field: dict) -> str:
    if entry.get("schema") != SCHEMA:
        return "old format"
    if latest.get(entry["strategy"]) is entry:
        return "ranked" if entry["strategy"] in field else "not ranked"
    return "superseded"


class EntryError(ValueError):
    pass


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not s:
        raise EntryError(f"strategy name {name!r} has no usable characters")
    return s


def make_entry(strategy: str, kind: str, windows_df: pd.DataFrame, *,
               span: tuple, sizing: str, market_snapshot: str, author: str = "",
               note: str = "", decisions_sha256: str | None = None,
               submitted_at: str | None = None) -> dict:
    held = {c: int(windows_df[c].sum()) if c in windows_df else 0
            for c in ("missing_rounds", "invalid_rounds")}
    return {
        "schema": SCHEMA, "kind": kind, "strategy": strategy, "author": author,
        "note": note,
        "submitted_at": submitted_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "decisions_sha256": decisions_sha256, "sizing": sizing,
        "span": [str(span[0]), str(span[1])], "market_snapshot": market_snapshot,
        **held,
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
    disjoint = disjoint_windows(refs[0]["windows"])
    n_independent = max(1, len(disjoint))

    ref_names = {r["strategy"] for r in refs}
    latest, versions, excluded, old_format = {}, {}, [], {}
    for e in entries:
        name = e["strategy"]
        if e["kind"] == SUBMITTED and name in ref_names:
            # Checked before `latest`: sharing the name key, a submission would otherwise
            # replace the anchor every other entry is read against.
            excluded.append({"strategy": name, "reason": "uses a reference strategy's name"})
            continue
        if e["kind"] == SUBMITTED:
            # Old-format versions count too: each was a look at the holdout.
            versions[name] = versions.get(name, 0) + 1
        if e.get("schema") != SCHEMA:
            old_format.setdefault(name, _old_schema(e))
            continue
        if e["kind"] == SUBMITTED:
            if name in latest and latest[name]["submitted_at"] >= e["submitted_at"]:
                continue
        elif name in latest:
            raise EntryError(f"two reference entries named {name!r}")
        latest[name] = e

    # Named once per strategy, and only while no current-format version stands in for it.
    excluded += [{"strategy": n, "reason": why} for n, why in old_format.items()
                 if n not in latest]
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
    disjoint_ranks = [per_window[i] for i in disjoint]
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
            **{f"mean_window_{k}": _mean_window(e, k) for k in METRICS},
            **{f"median_window_{k}": float(pd.Series([w[k] for w in e["windows"]]).median())
               for k in METRICS},
        })
        # The same ranks as the score above, against the whole field, in only the windows
        # that share no day. Ranked again on that subset alone, each window's field would
        # be the same but the reader would take it for a second, independent score.
        rows[-1]["disjoint"] = [
            {"window_start": e["windows"][i]["window_start"],
             "window_end": e["windows"][i]["window_end"],
             "position": int(r.loc[name, "position"]),
             "overall_score": float(r.loc[name, "overall_score"]),
             **{k: e["windows"][i][k] for k in METRICS}}
            for i, r in zip(disjoint, disjoint_ranks)]
        rows[-1]["mean_disjoint_overall_score"] = float(
            pd.Series([w["overall_score"] for w in rows[-1]["disjoint"]]).mean())
    edges = histogram_edges(field)
    for r in rows:
        r["hist"] = {k: histogram(field[r["strategy"]], k, edges[k]) for k in METRICS}
    rows.sort(key=lambda r: (r["mean_overall_score"], -r["mean_window_cumulative_return"]))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    history = sorted(
        ({"strategy": e["strategy"], "author": e["author"], "submitted_at": e["submitted_at"],
          "status": _status(e, latest, field),
          **{f"mean_window_{k}": _mean_window(e, k) for k in METRICS}}
         for e in entries if e.get("kind") == SUBMITTED),
        key=lambda h: h["submitted_at"], reverse=True)
    return {"span": board_span, "sizing": BOARD_SIZING, "windows": len(board_windows),
            "bins": edges,
            "independent_windows": n_independent,
            "entrants": len(rows), "rows": rows, "excluded": excluded, "history": history}
