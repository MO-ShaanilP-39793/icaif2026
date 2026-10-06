"""Where a replay's strategies would place on the holdout leaderboard, window by window.

`agent_replay` ranks a desk against our own simulated field (`baselines.FIELD`), a
dozen reference strategies. The v2 gate is judged against the holdout board's ~38
entrants, and a place among a dozen baselines reads like a place on the board while
meaning much less. So this reads the board as its page does (manifest, references,
then every entry in `entries/index.json`), ranks it with the board's own code
(`leaderboard.standings`), and places each replay strategy in each window the way the
agentic panel does: against the board's whole field plus itself, never against the
other strategies of the same replay.

**The anchor is checked before any place is read.** The replay simulates its windows on
our market; the board scored its entries with the holdout scorer on its own snapshot.
If the two disagree on a window (another price snapshot, another window end, a change
to fills), every place in it is read against a field priced on different data, and it
would still look like a ranking. `inv_vol_hold_75` is in every replay and is a board
reference, so its four metrics must match the board's to `ANCHOR_TOL`, or the window
is refused with the differences.
"""

import json
import urllib.error
import urllib.request

import pandas as pd

from icaif import leaderboard, net, ranking

BOARD_URL = "https://mo-ai-inv-icaif2026-leaderboard.static.hf.space/"
ANCHOR = "inv_vol_hold_75"
# The board ranks on values rounded to 12 places; the replay's metrics are the same
# floats when the prices agree (measured: equal to the last digit on three windows).
# A real disagreement is a fill price, which moves a metric by 1e-5 or more.
ANCHOR_TOL = 1e-9
METRICS = list(ranking.METRICS)


class AnchorMismatch(ValueError):
    pass


def fetch_entries(base: str = BOARD_URL, get=None) -> list[dict]:
    """Every entry the board page ranks: its references, then each submitted file.

    Read through the OS trust store (`net.ssl_context`): the office proxy re-signs the
    host, and turning verification off would accept any other interceptor too. A 404
    on the index means nothing has been submitted, as the page reads it.
    """
    def fetch(path):
        with urllib.request.urlopen(base + path, context=net.ssl_context(), timeout=60) as r:
            return json.load(r)

    get = get or fetch
    manifest = get("manifest.json")
    entries = [get(p) for p in manifest["references"]]
    try:
        index = get("entries/index.json")["entries"]
    except urllib.error.HTTPError as err:
        if err.code != 404:
            raise
        index = []
    return entries + [get(p) for p in index]


def board_windows(standing: dict) -> dict:
    """window_start -> that window's board record (`by_window` of `standings`)."""
    return {w["window_start"]: w for w in standing["by_window"]}


def check_anchor(replay: pd.DataFrame, window: dict) -> float:
    """The largest metric difference between the replay's anchor and the board's.

    Raises `AnchorMismatch` above `ANCHOR_TOL`, or when either side lacks the anchor:
    without it nothing says the replay and the board priced the window alike.
    """
    ours = replay[replay["strategy"] == ANCHOR]
    if len(ours) != 1:
        raise AnchorMismatch(f"window {window['window_start']}: the replay has {len(ours)} "
                             f"{ANCHOR} rows; it is the check that the replay and board agree")
    theirs = [r for r in window["rows"] if r["strategy"] == ANCHOR]
    if not theirs:
        raise AnchorMismatch(f"window {window['window_start']}: no {ANCHOR} on the board")
    diffs = {k: abs(float(ours.iloc[0][k]) - float(theirs[0][k])) for k in METRICS}
    worst = max(diffs.values())
    if not worst <= ANCHOR_TOL:   # NaN fails too
        raise AnchorMismatch(
            f"window {window['window_start']}: the replay's {ANCHOR} differs from the board's "
            "(" + ", ".join(f"{k} {float(ours.iloc[0][k]):.8g} vs {theirs[0][k]:.8g}"
                            for k in METRICS if not diffs[k] <= ANCHOR_TOL)
            + "); the replay and the board priced this window differently, so no place in "
              "it is comparable")
    return worst


def place(window: dict, name: str, metrics: dict) -> dict:
    """`name`'s place in `window` with the board's whole field plus it alone.

    The same ranking the agentic panel runs (`leaderboard.agentic_standings`): the
    field's metrics, the candidate added, rounded to 12 places so float dust cannot split
    what the contest's Decimals tie. Also the field entrants just ahead and behind.
    """
    field = {r["strategy"]: {k: r[k] for k in METRICS} for r in window["rows"]}
    key = name if name not in field else f"replay:{name}"
    m = pd.DataFrame(field).T[METRICS]
    m.loc[key] = {k: float(metrics[k]) for k in METRICS}
    r = ranking.rank_window(m.astype(float).round(12))
    order = r.sort_values(["position", "overall_score"]).index.tolist()
    i = order.index(key)
    return {"position": int(r.loc[key, "position"]), "of": len(m),
            "overall_score": float(r.loc[key, "overall_score"]),
            **{f"rank_{k}": float(r.loc[key, f"rank_{k}"]) for k in METRICS},
            "ahead": order[i - 1] if i > 0 else "", "behind": order[i + 1] if i + 1 < len(order) else ""}


def _board_row(window: dict, name: str):
    return next((r for r in window["rows"] if r["strategy"] == name), None)


def _matches(metrics: dict, row: dict) -> bool:
    return all(abs(metrics[k] - float(row[k])) <= ANCHOR_TOL for k in METRICS)


def rank_replay(replay: pd.DataFrame, standing: dict) -> tuple[pd.DataFrame, list[str]]:
    """Each replay strategy's place in each of its windows on the board.

    `replay`: an `agent_replay` windows.csv (window, strategy, the four metrics). Returns
    one row per (window, strategy) and the windows the board does not cover (outside its
    span), which have no field to rank in. Raises `AnchorMismatch` for a covered window
    whose anchor disagrees.

    A strategy the board already ranks under the same name (the anchor, and the rule,
    `q_riskparity_entry_regime`, which was submitted) takes its board place when its
    metrics match: inserted beside its own copy, it would tie with itself and push the
    field down one. If they differ it is placed as `replay:<name>` and `source` says so,
    since a same-named entry run on another version is not the strategy replayed here.
    """
    wins = board_windows(standing)
    rows, off = [], []
    for w, g in replay.groupby(replay["window"].astype(str), sort=True):
        if w not in wins:
            off.append(w)
            continue
        bw = wins[w]
        tol = check_anchor(g, bw)
        for s in g.itertuples():
            metrics = {k: float(getattr(s, k)) for k in METRICS}
            on_board = _board_row(bw, s.strategy)
            if on_board is not None and _matches(metrics, on_board):
                p = {"position": on_board["position"], "of": len(bw["rows"]),
                     "overall_score": on_board["overall_score"],
                     **{f"rank_{k}": on_board[f"rank_{k}"] for k in METRICS},
                     "ahead": "", "behind": "", "source": "board"}
            else:
                p = {**place(bw, s.strategy, metrics),
                     "source": "placed" if on_board is None else "placed; differs from the board's entry"}
            rows.append({"window_start": w, "window_end": bw["window_end"], "strategy": s.strategy,
                         **p, **metrics, "anchor_max_diff": tol})
    return pd.DataFrame(rows), off


def compare(placed: pd.DataFrame, against: str) -> pd.DataFrame:
    """Per strategy, its board score minus `against`'s, across the windows both cover.

    Each score is the strategy's own with the board's field (the anchor's is its board
    score), so the difference is the gate's: lower is better. The SE divides by the
    windows that share no day, as the board's does: overlapping windows repeat most of
    their days, and counting each would claim precision the data does not hold.
    """
    piv = placed.pivot(index="window_start", columns="strategy", values="overall_score")
    if against not in piv:
        return pd.DataFrame()
    ends = placed.drop_duplicates("window_start").set_index("window_start")["window_end"]
    out = []
    for s in piv.columns.drop(against):
        d = (piv[s] - piv[against]).dropna()
        if d.empty:
            continue
        spans = [{"window_start": w, "window_end": ends[w]} for w in d.index]
        n_ind = max(1, len(leaderboard.disjoint_windows(spans)))
        out.append({"strategy": s, "against": against, "windows": len(d), "independent": n_ind,
                    "mean_diff": float(d.mean()),
                    "se": float(d.std(ddof=1) / n_ind ** 0.5) if len(d) > 1 else float("nan"),
                    "better_in": int((d < 0).sum()), "worse_in": int((d > 0).sum())})
    return pd.DataFrame(out)
