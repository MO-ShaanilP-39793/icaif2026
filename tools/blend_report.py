"""Blend the risk-parity and inverse-vol shapes, and score every mix on both fields.

    .venv/bin/python tools/blend_report.py

Risk parity beats the inverse-vol hold against the near-hold field (`baselines.FIELD`)
and loses to it against the active one (`baselines.ACTIVE_FIELD`), and the real field
can't be seen until the competition ends. A shape worth submitting should hold up on
every field, including the default one without the reference's own near-clone. Each candidate is bought once and held, at 75% or at the regime-blended entry
exposure. The 167 windows split into eras as in quant_report. Writes
reports/blend_windows.csv.
"""

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import baselines, data, markets, quant_strategies as qs, windows  # noqa: E402

CONFIRM_START = date(2023, 1, 1)
REF = "inv_vol_hold_75"


def main() -> None:
    t0 = time.time()
    market = markets.research_market()
    starts = [s for s in windows.window_starts(market) if s >= market.days[60]]
    cands = {REF: baselines.scaled(baselines.InverseVolHold, 0.75)}
    for shape in ("inverse_vol", "blend25", "blend50", "blend75", "risk_parity"):
        cands[f"{shape}_75"] = qs.book(shape, qs.Fixed, qs.ENTRY_ONLY)
        cands[f"{shape}_regime"] = qs.book(shape, qs.Regime, qs.ENTRY_ONLY)
    runs = {n: windows.run_field({n: f}, market, starts) for n, f in cands.items()}
    rows = []
    # The default field holds inv_vol_hold at 100%, a near-clone of the reference at
    # 75%: holding cash shaves the clone's Sharpe by ~0.4%, so it loses that near-tie
    # in ~89% of windows and ~0.1 of score, which any book that isn't a copy collects
    # for free. Scored without the clone too, so a shape's edge is not that artefact.
    no_clone = {k: v for k, v in baselines.FIELD.items() if k != "inv_vol_hold"}
    for fname, fdef in (("default", baselines.FIELD), ("default_no_clone", no_clone),
                        ("active", baselines.ACTIVE_FIELD)):
        field = windows.run_field(fdef, market, starts)
        rows += [windows.rank_against_field(r, field).assign(field=fname) for r in runs.values()]
    res = pd.concat(rows, ignore_index=True)
    res["era"] = ["choose" if w < CONFIRM_START else "confirm" for w in res.window]
    res.to_csv(data.ROOT / "reports" / "blend_windows.csv", index=False)

    out = {}
    for (fname, era), g in res.groupby(["field", "era"]):
        piv = g.pivot(index="window", columns="strategy", values="overall_score")
        d = piv.sub(piv[REF], axis=0)
        out[(fname, era, "diff")] = d.mean()
        out[(fname, era, "se")] = d.std() / len(d) ** 0.5
    table = pd.DataFrame(out)
    table[("both", "all", "worst_diff")] = table.xs("diff", axis=1, level=2).max(axis=1)
    table[("both", "all", "mean_diff")] = table.xs("diff", axis=1, level=2).mean(axis=1)
    print(f"score difference from {REF} (negative is better), by field and era:\n")
    print(table.sort_values(("both", "all", "worst_diff")).round(3).to_string())
    print(f"\n{len(starts)} windows, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
