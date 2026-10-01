"""Does the Black-Litterman tilt pay as a rule, before an LLM is asked to choose it?

    .venv/bin/python tools/views_report.py        # ~1 min
    .venv/bin/python tools/views_report.py 3      # the first 3 windows, to time it

Two rule desks that always take the scores as views at entry ("light", "strong"), and
otherwise do what the rule does: risk parity at the regime-blended exposure, bought
once, then hold. Each is scored against `q_riskparity_entry_regime` (the same book with
no views) on two fields: the default one, and the default one without `inv_vol_hold`,
which is a near-clone of a 75% hold and hands any book that is not a copy ~0.1 of
score (TODO, 2026-09-29).

Only windows whose entry day has daily-model scores (the walk-forward predictions,
2023 on): before that the views cannot be taken, every desk is the reference, and the
difference would be diluted by zeros. Reviews and events are off: the rule holds
through both, so the ledgers are the same and the run is shorter. An entry whose views
book cannot be built falls back to the plain book, and the count is printed.

Writes reports/views_windows.csv.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import baselines, compiler, data, markets, quant_strategies as qs, ranking, windows  # noqa: E402
from icaif.agents.brains import RuleBrain  # noqa: E402
from icaif.agents.desk import Desk, DeskConfig  # noqa: E402

REF = "q_riskparity_entry_regime"


class ViewsRule(RuleBrain):
    """The rule's answer to every question, except the entry takes views at `level`."""

    def __init__(self, level: str):
        self.level, self.name = level, f"rule+views_{level}"

    def decide(self, role, system, payload, schema, timeout):
        answer = dict(payload["rule_proposal"])
        if role == "entry":
            answer["views"] = self.level
        return schema.model_validate(answer)


def main() -> None:
    t0 = time.time()
    market = markets.research_market()
    scores = compiler.load_daily_scores()
    first = scores.frame.index.min().date()
    starts = [s for s in windows.window_starts(market) if s >= first]
    if len(sys.argv) > 1:  # a timing run: the first N windows, report not written
        starts = starts[: int(sys.argv[1])]
    cfg = DeskConfig(review=False, events=False)
    desks: dict[str, list] = {"light": [], "strong": []}

    def views_desk(level):
        def make():
            d = Desk(ViewsRule(level), cfg, scores=scores)
            desks[level].append(d)
            return d
        return make

    cands = {REF: qs.CANDIDATES[REF], "views_light": views_desk("light"),
             "views_strong": views_desk("strong")}
    runs = {n: windows.run_field({n: f}, market, starts) for n, f in cands.items()}
    no_clone = {k: v for k, v in baselines.FIELD.items() if k != "inv_vol_hold"}
    rows = []
    for fname, fdef in (("default", baselines.FIELD), ("default_no_clone", no_clone)):
        field = windows.run_field(fdef, market, starts)
        rows += [windows.rank_against_field(r, field).assign(field=fname) for r in runs.values()]
    res = pd.concat(rows, ignore_index=True)
    if len(sys.argv) == 1:
        res.to_csv(data.ROOT / "reports" / "views_windows.csv", index=False)

    for level, ds in desks.items():
        fell = sum(e["source"] == "fallback" for d in ds for e in d.log)
        print(f"views_{level}: {len(ds)} entries, {fell} fell back to the plain book")
    print(f"{len(starts)} windows with scores, {starts[0]} .. {starts[-1]}\n")
    metrics = ["overall_score"] + [f"rank_{m}" for m in ranking.METRICS]
    for fname, g in res.groupby("field"):
        print(f"field {fname}: difference from {REF} (negative is better)")
        out = {}
        for name in ("views_light", "views_strong"):
            row = {}
            for m in metrics:
                piv = g.pivot(index="window", columns="strategy", values=m)
                d = piv[name] - piv[REF]
                row[f"{m}"] = d.mean()
                if m == "overall_score":
                    row["se"] = d.std() / len(d) ** 0.5
                    row["better"] = (d < 0).mean()
                    row["worse"] = (d > 0).mean()
            out[name] = row
        print(pd.DataFrame(out).T.round(3).to_string(), "\n")
        by_year = g.assign(year=pd.to_datetime(g["window"]).dt.year).pivot_table(
            index="year", columns="strategy", values="overall_score", aggfunc="mean")
        print((by_year[["views_light", "views_strong"]].sub(by_year[REF], axis=0)).round(3)
              .assign(windows=g[g.strategy == REF].assign(
                  year=pd.to_datetime(g["window"]).dt.year).groupby("year").size()).to_string(), "\n")
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
