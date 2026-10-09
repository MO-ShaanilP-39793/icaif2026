"""Read stage 1's arms and apply the design's decision rules (`stage1_doe.md`).

    .venv/bin/python tools/doe_report.py arch TAG TAG TAG          # round 1: single PM or analysts
    .venv/bin/python tools/doe_report.py streams TAG x16           # round 2: each stream's effect
    .venv/bin/python tools/doe_report.py cash BASE [--real TAG ..] # round 3: the cash floor
    .venv/bin/python tools/doe_report.py noise TAG TAG [TAG]       # round 4: repeats of one arm
    .venv/bin/python tools/doe_report.py table TAG ...             # every window's responses, as CSV

A TAG is a directory under output/entry/ written by `tools/entry_replay.py arm`. Scores
are the mean of the four ranks against the modelled field, lower better, on the
no-clone field (primary) with the default field printed beside it. Every comparison is
paired by window, so a window's own market swing cancels; SEs are over windows.

The rules are written here, once, so a round's verdict is the rule's and not a reading
of the table after the fact:

- **Architecture**: the lowest mean score, unless a cheaper architecture is within one
  SE of it (paired), in which case the cheapest such. Cost: none < reports_only <
  reports_raw.
- **Streams**: a stream is kept only if its main effect helps by more than one SE. Every
  other stream is dropped: one that can't show its worth in 16 x 13 entries isn't
  earning its prompt length, latency and money.
- **Cash floor**: the lowest mean score, unless "no floor" is within one SE of it.
- **Noise**: reported, and used to read the others: a gap smaller than the spread
  between repeats of one arm is not a finding.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icaif import data  # noqa: E402
from icaif.agents.v3 import CASH_FLOORS, STREAMS, STREAMS16, capped_book, floor_gross  # noqa: E402

OUT = data.ROOT / "output" / "entry"
FIELDS = ("no_clone", "default")
METRICS = ("cumulative_return", "sharpe_ratio", "maximum_drawdown", "turnover")
RESPONSES = ("overall_score",) + METRICS + tuple(f"rank_{m}" for m in METRICS)
COST = {"none": 0, "reports_only": 1, "reports_raw": 2}
BASELINES = ("inv_vol_hold_75", "q_riskparity_entry_regime", "cash")


class Arm:
    """One `entry_replay.py arm` run: its config, its windows table and its entries."""

    def __init__(self, tag: str, root: Path = OUT):
        self.tag, self.dir = tag, Path(root) / tag
        if not (self.dir / "windows.csv").exists():
            raise SystemExit(f"{self.dir} has no windows.csv: not an arm run")
        self.config = json.loads((self.dir / "config.json").read_text())
        self.name = self.config["arm"]
        self.table = pd.read_csv(self.dir / "windows.csv")
        path = self.dir / "entries.jsonl"
        self.entries = ([json.loads(x) for x in path.read_text().splitlines()] if path.exists() else [])

    def rows(self, strategy: str = None, field: str = "no_clone") -> pd.DataFrame:
        t = self.table[(self.table["strategy"] == (strategy or self.name)) & (self.table["field"] == field)]
        return t.set_index("window")

    def books(self) -> dict:
        """{window: {ticker: weight}} as bought (real names: the codes are the tickers)."""
        return {e["window"]: (e["entry"] or {}).get("weights", {}) for e in self.entries}


def paired(a: pd.Series, b: pd.Series) -> dict:
    """a - b by window, on the windows both have."""
    d = (a - b).dropna()
    n = len(d)
    return {"diff": d.mean() if n else np.nan, "se": d.std() / np.sqrt(n) if n > 1 else np.nan,
            "better": int((d < 0).sum()), "worse": int((d > 0).sum()), "n": n}


def same_windows(arms: list) -> list:
    sets = [set(a.rows().index) for a in arms]
    if any(s != sets[0] for s in sets):
        raise SystemExit("the arms ran on different windows: "
                         + "; ".join(f"{a.tag}: {len(s)}" for a, s in zip(arms, sets)))
    return sorted(sets[0])


def summary(arms: list, field: str, baselines=True) -> pd.DataFrame:
    rows = {}
    for a in arms:
        r = a.rows(field=field)
        rows[a.tag] = {"mean_score": r["overall_score"].mean(),
                       "se": r["overall_score"].std() / np.sqrt(len(r)),
                       **{f"mean_{m}": r[m].mean() for m in METRICS}}
    if baselines:
        for b in BASELINES:
            r = arms[0].rows(b, field)
            if len(r):
                rows[f"({b})"] = {"mean_score": r["overall_score"].mean(),
                                  "se": r["overall_score"].std() / np.sqrt(len(r)),
                                  **{f"mean_{m}": r[m].mean() for m in METRICS}}
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------- rules

def choose_architecture(scores: dict, cost: dict) -> tuple:
    """(the chosen key, why). `scores`: {key: per-window score Series}; `cost`: {key: rank}."""
    means = {k: v.mean() for k, v in scores.items()}
    best = min(means, key=means.get)
    # "Within one SE" is anything not shown to be worse by more than one: with too few
    # windows for an SE, the cheaper arm stands rather than whichever won by chance.
    near = [k for k in scores if k != best and cost[k] < cost[best]
            and not paired(scores[k], scores[best])["diff"] > paired(scores[k], scores[best])["se"]]
    if near:
        pick = min(near, key=cost.get)
        p = paired(scores[pick], scores[best])
        return pick, (f"{best} scored best, but {pick} is cheaper and within one SE of it "
                      f"({p['diff']:+.3f}, SE {p['se']:.3f})")
    return best, f"{best} scored best, and no cheaper arm is within one SE"


def stream_effects(scores: dict, included: dict) -> pd.DataFrame:
    """Each stream's main effect, by window then averaged.

    `scores`: {run: per-window Series of one response}; `included`: {run: set of streams}.
    effect(s, w) = mean over the runs with s, minus the mean over the runs without, in
    window w. Negative helps a score (lower is better).
    """
    frame = pd.DataFrame(scores)
    out = {}
    for s in STREAMS:
        with_ = [r for r in frame.columns if s in included[r]]
        without = [r for r in frame.columns if s not in included[r]]
        if not with_ or not without:
            continue
        e = frame[with_].mean(axis=1) - frame[without].mean(axis=1)
        out[s] = {"effect": e.mean(), "se": e.std() / np.sqrt(len(e)), "helps_in": int((e < 0).sum()),
                  "hurts_in": int((e > 0).sum()), "n": int(len(e))}
    return pd.DataFrame(out).T


def keep_streams(effects: pd.DataFrame) -> list:
    """Kept only if the stream helps by more than one SE; a tie drops it."""
    return [s for s, r in effects.iterrows() if r["effect"] < -r["se"]]


def choose_floor(scores: dict) -> tuple:
    """`scores`: {floor: per-window Series}, floor 0.0 being no floor."""
    means = {f: v.mean() for f, v in scores.items()}
    best = min(means, key=means.get)
    if best == 0.0:
        return 0.0, "no floor scored best"
    p = paired(scores[0.0], scores[best])
    if not p["diff"] > p["se"]:     # a NaN SE (too few windows) is a tie, not a win
        return 0.0, (f"a {best:.0%} floor scored best, but no floor is within one SE of it "
                     f"({p['diff']:+.3f}, SE {p['se']:.3f})")
    return best, f"a {best:.0%} floor scored best, by more than one SE over no floor"


def book_agreement(a: dict, b: dict) -> dict:
    """How alike two books are: shared names (Jaccard) and the correlation of their shares."""
    na, nb = {t for t, v in a.items() if v > 0}, {t for t, v in b.items() if v > 0}
    names = sorted(na | nb)
    if not names:
        return {"jaccard": 1.0, "share_corr": 1.0}
    sa = np.array([a.get(t, 0.0) for t in names]) / max(sum(a.values()), 1e-12)
    sb = np.array([b.get(t, 0.0) for t in names]) / max(sum(b.values()), 1e-12)
    corr = float(np.corrcoef(sa, sb)[0, 1]) if len(names) > 1 and sa.std() > 0 and sb.std() > 0 else np.nan
    return {"jaccard": len(na & nb) / len(names), "share_corr": corr}


# ----------------------------------------------------------------------------- commands

def cmd_arch(tags: list) -> None:
    arms = [Arm(t) for t in tags]
    same_windows(arms)
    kinds = [a.config["config"]["analysts"] for a in arms]
    if len(set(kinds)) != len(kinds):
        raise SystemExit(f"each architecture once, please: got {kinds}")
    for f in FIELDS:
        print(f"\n{f} field" + ("  (primary)" if f == "no_clone" else ""))
        print(summary(arms, f).round(4).to_string())
    scores = {k: a.rows()["overall_score"] for k, a in zip(kinds, arms)}
    pick, why = choose_architecture(scores, COST)
    print(f"\nround 1 verdict: {pick}. {why}.")


def cmd_streams(tags: list) -> None:
    arms = [Arm(t) for t in tags]
    same_windows(arms)
    runs = {a.config.get("design_run"): a for a in arms}
    if a_bad := [a.tag for a in arms if a.config.get("design") != "streams16"]:
        raise SystemExit(f"not streams16 runs: {a_bad}")
    if sorted(runs) != list(range(1, 17)):
        raise SystemExit(f"need runs 1-16 once each; got {sorted(runs)}")
    for r, a in runs.items():
        if set(a.config["config"]["streams"]) != set(STREAMS16[r - 1]):
            raise SystemExit(f"{a.tag} says run {r} but carries streams {a.config['config']['streams']}")
    arch = {a.config["config"]["analysts"] for a in arms}
    floors = {a.config.get("cash_floor", 0.0) for a in arms}
    if len(arch) != 1 or len(floors) != 1:
        raise SystemExit(f"the 16 runs must share architecture and cash floor: {arch}, {floors}")
    included = {r: set(STREAMS16[r - 1]) for r in runs}
    for f in FIELDS:
        print(f"\n{f} field" + ("  (primary)" if f == "no_clone" else "")
              + ": each stream's main effect on the overall score (negative helps)")
        eff = stream_effects({r: a.rows(field=f)["overall_score"] for r, a in runs.items()}, included)
        print(eff.round(4).to_string())
        if f == "no_clone":
            primary = eff
    print("\nhow each stream moves the raw metrics and their ranks (primary field; return and "
          "Sharpe up is better, drawdown and turnover down):")
    cols = {}
    for resp in METRICS + tuple(f"rank_{m}" for m in METRICS):
        e = stream_effects({r: a.rows()[resp] for r, a in runs.items()}, included)
        cols[resp] = e["effect"].round(5)
    print(pd.DataFrame(cols).to_string())
    keep = keep_streams(primary)
    print(f"\nround 2 verdict: keep {keep or 'none'}; drop {[s for s in STREAMS if s not in keep]}.")
    print("The kept set is generally not one of the 16 rows: rounds 3 and 4 run it as its own arm "
          f"(entry_replay.py arm --analysts {arch.pop()} --drop {' '.join(s for s in STREAMS if s not in keep)}).")


def cmd_cash(base: str, real: list) -> None:
    b = Arm(base)
    if b.config.get("cash_floor", 0.0) != 0.0:
        raise SystemExit(f"{base} ran with a cash floor; the base is the no-floor arm")
    for f in FIELDS:
        rows = {0.0: b.rows(field=f)["overall_score"]}
        for fl in CASH_FLOORS[1:]:
            r = b.rows(f"arm_floor_{int(round(fl * 100))}", f)
            if len(r):
                rows[fl] = r["overall_score"]
        print(f"\n{f} field" + ("  (primary)" if f == "no_clone" else "")
              + ": each floor, scored by capping the no-floor run's books (no calls)")
        tab = pd.DataFrame({f"{int(fl * 100)}%": {"mean_score": s.mean(), "se": s.std() / np.sqrt(len(s)),
                                                   **{k: v for k, v in paired(s, rows[0.0]).items()
                                                      if k in ("diff", "better", "worse")}}
                            for fl, s in rows.items()}).T
        print(tab.round(4).to_string())
        if f == "no_clone":
            primary = rows
    if real:
        print("\nthe free scores' assumption, checked: a real floor's books against the no-floor "
              "books capped at that floor")
        books = b.books()
        for t in real:
            r = Arm(t)
            fl = r.config.get("cash_floor", 0.0)
            if not fl:
                raise SystemExit(f"{t} ran with no cash floor")
            agree = pd.DataFrame({w: book_agreement(rb, capped_book(books.get(w, {}), floor_gross(fl)))
                                  for w, rb in r.books().items()}).T
            p = paired(r.rows()["overall_score"], b.rows(f"arm_floor_{int(round(fl * 100))}")["overall_score"])
            print(f"  {fl:.0%} floor ({t}): shared names {agree['jaccard'].mean():.2f}, share correlation "
                  f"{agree['share_corr'].mean():.2f}; real minus capped score {p['diff']:+.3f} "
                  f"(SE {p['se']:.3f}, n {p['n']}). Compare with round 4's noise before trusting the free scores.")
    pick, why = choose_floor(primary)
    print(f"\nround 3 verdict (free scores): {'no floor' if pick == 0.0 else f'at least {pick:.0%} cash'}. {why}.")


def cmd_noise(tags: list) -> None:
    arms = [Arm(t) for t in tags]
    same_windows(arms)
    cfgs = [{k: v for k, v in a.config["config"].items()} for a in arms]
    if any(c != cfgs[0] for c in cfgs):
        raise SystemExit("noise compares repeats of one arm: the configs differ")
    reps = sorted(a.config.get("repeat", 0) for a in arms)
    if len(set(reps)) != len(reps):
        raise SystemExit(f"each repeat once: got {reps}")
    frame = pd.DataFrame({a.config.get("repeat", 0): a.rows()["overall_score"] for a in arms})
    means = frame.mean()
    pairs = [(i, j) for i in frame.columns for j in frame.columns if i < j]
    gaps = [abs(frame[i] - frame[j]).mean() for i, j in pairs]
    print(f"repeats {reps}: mean scores " + ", ".join(f"{k}: {v:.3f}" for k, v in means.items()))
    print(f"spread of the arm's mean across repeats (SD): {means.std():.3f}")
    print(f"mean |difference| in one window between two repeats: {np.mean(gaps):.3f}")
    books = [a.books() for a in arms]
    agree = [book_agreement(books[i][w], books[j][w]) for i, j in
             [(i, j) for i in range(len(arms)) for j in range(i + 1, len(arms))] for w in books[0]]
    print(f"books alike across repeats: shared names {np.mean([x['jaccard'] for x in agree]):.2f}, "
          f"share correlation {np.nanmean([x['share_corr'] for x in agree]):.2f}")
    print("\nA gap between arms smaller than the SD of the arm's mean is not a finding.")


def cmd_table(tags: list) -> None:
    rows = []
    for t in tags:
        a = Arm(t)
        for f in FIELDS:
            r = a.rows(field=f).reset_index()
            rows.append(r.assign(tag=t, arm=a.name, field=f))
    out = pd.concat(rows, ignore_index=True)[["tag", "arm", "field", "window", *RESPONSES]]
    path = OUT / "doe_table.csv"
    out.to_csv(path, index=False)
    print(out[out["field"] == "no_clone"].drop(columns="field").round(4).to_string(index=False))
    print(f"\n-> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["arch", "streams", "cash", "noise", "table"])
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--real", nargs="*", default=[], help="cash: real-floor arms to check the free scores")
    args = ap.parse_args()
    if args.command == "arch":
        cmd_arch(args.tags)
    elif args.command == "streams":
        cmd_streams(args.tags)
    elif args.command == "cash":
        if len(args.tags) != 1:
            raise SystemExit("cash takes one base tag (the no-floor arm); real floors go in --real")
        cmd_cash(args.tags[0], args.real)
    elif args.command == "noise":
        cmd_noise(args.tags)
    else:
        cmd_table(args.tags)


if __name__ == "__main__":
    main()
