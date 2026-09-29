"""Race the rank-playing controller against the holds, planned and scored on two fields.

    .venv/bin/python tools/rankplay_report.py

The planner needs a model of the field, and the real field is unobservable until the
competition ends. So each planner variant is scored against both the field it planned
against and the one it didn't: a controller that only wins against its own model of
the rivals has learned the model, not the game. Eras split as in quant_report
(2016-22 to look at, 2023-26 to confirm). Writes reports/rankplay_windows.csv.
"""

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import baselines, data, markets, quant_strategies as qs, rankplay, windows  # noqa: E402

CONFIRM_START = date(2023, 1, 1)
REFS = ("inv_vol_hold_75", "rp_entry_regime")


def main() -> None:
    t0 = time.time()
    market = markets.research_market()
    # The planner bootstraps from the 250 days before each morning; earlier windows
    # would have it plan on too little history (it raises there rather than sit in cash).
    starts = [s for s in windows.window_starts(market) if s >= market.days[260]]
    players = {}

    def tracked(name, factory):
        def make():
            p = factory()
            players.setdefault(name, []).append(p)
            return p
        return make

    cands = {
        "inv_vol_hold_75": baselines.scaled(baselines.InverseVolHold, 0.75),
        "rp_entry_regime": qs.CANDIDATES["q_riskparity_entry_regime"],
        "plan_default": tracked("plan_default", rankplay.rank_player()),
        "plan_default_entry": tracked("plan_default_entry", rankplay.rank_player(entry_only=True)),
        "plan_active": tracked("plan_active", rankplay.rank_player(plan_field=baselines.ACTIVE_FIELD)),
    }
    runs = {n: windows.run_field({n: f}, market, starts) for n, f in cands.items()}
    print(f"candidates simulated ({time.time() - t0:.0f}s)")
    rows = []
    for fname, fdef in (("default", baselines.FIELD), ("active", baselines.ACTIVE_FIELD)):
        field = windows.run_field(fdef, market, starts)
        for n, r in runs.items():
            rows.append(windows.rank_against_field(r, field).assign(scored_on=fname))
    res = pd.concat(rows, ignore_index=True)
    res["era"] = ["choose" if w < CONFIRM_START else "confirm" for w in res.window]
    res.to_csv(data.ROOT / "reports" / "rankplay_windows.csv", index=False)

    ranks = ["rank_cumulative_return", "rank_sharpe_ratio", "rank_maximum_drawdown", "rank_turnover"]
    for (scored, era), g in res.groupby(["scored_on", "era"]):
        piv = g.pivot(index="window", columns="strategy", values="overall_score")
        s = g.groupby("strategy").agg(score=("overall_score", "mean"),
                                      **{r.replace("rank_", "r_"): (r, "mean") for r in ranks},
                                      ret=("cumulative_return", "mean"),
                                      sharpe=("sharpe_ratio", "mean"),
                                      mdd=("maximum_drawdown", "mean"),
                                      turnover=("turnover", "mean"))
        for ref in REFS:
            d = piv.sub(piv[ref], axis=0)
            s[f"vs_{ref}"] = d.mean()
            s[f"se_{ref}"] = d.std() / len(d) ** 0.5
        print(f"\n=== scored on {scored} field, {era} ({len(piv)} windows; lower is better) ===")
        print(s.sort_values("score").round(4).to_string())

    for name, ps in players.items():
        log = pd.DataFrame([dict(e, window=p.start) for p in ps for e in p.log])
        entry = log.groupby("window").first()["choice"].value_counts().sort_index()
        moves = log[(log.day_no > 1) & (log.choice != "hold")]
        print(f"\n{name}: entry exposure {entry.to_dict()}; "
              f"moves after entry in {moves.window.nunique()} of {log.window.nunique()} windows "
              f"({len(moves)} moves)")
    print(f"\n{len(starts)} windows, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
