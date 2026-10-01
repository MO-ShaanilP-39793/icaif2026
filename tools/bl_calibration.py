"""How far do the Black-Litterman view levels move the risk-parity book? (Roadmap step 2)

    .venv/bin/python tools/bl_calibration.py

For every window entry day with daily-model scores (the walk-forward predictions,
2023 on), the entry's risk-parity book on its 60-session tail, and the book the scores
make of it as views at a grid of Idzorek confidences. Reports the active share (half
the summed |weight change|, the fraction of the book moved) and the largest weight,
per confidence, across those days. `signals.VIEW_LEVELS` is chosen from this table:
a level is a fraction of the book the Strategist can see and name, not a number whose
effect it has to guess. No trading, no ranking: about 30 s, most of it loading bars.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from icaif import calendar, compiler, markets, quant, quant_strategies as qs, sim, windows  # noqa: E402
from icaif import weights as W  # noqa: E402
from icaif.agents import signals  # noqa: E402

CONFIDENCES = (0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.5, 1.0)


def main() -> None:
    market = markets.research_market()
    scores = compiler.load_daily_scores()
    first = scores.frame.index.min().date()
    starts = [s for s in windows.window_starts(market) if s >= first]
    rows = []
    for day in starts:
        deadline = calendar.at(day, calendar.ROUNDS[1][0])
        ctx = sim.RoundContext(day, 1, deadline, calendar.at(day, calendar.ROUNDS[1][1]),
                               {}, sim.INITIAL_NAV, market)
        closes = qs.daily_closes(ctx, qs.SHAPE_DAYS + 1)
        tail = np.log(closes).diff().iloc[1:]
        prior = qs.shape_risk_parity(tail)
        s = scores.for_day(day, deadline)
        if prior is None or s.notna().sum() < 2:
            continue
        cov = quant.shrunk_cov(tail)
        alpha = signals.score_alpha(s, cov)
        for c in CONFIDENCES:
            w = quant.black_litterman(prior, cov, alpha, c, cap=W.CAP)
            rows.append({"day": day, "confidence": c,
                         "active_share": 0.5 * float((w - prior).abs().sum()),
                         "max_weight": float(w.max()), "names_held": int((w > W.GRID).sum())})
    res = pd.DataFrame(rows)
    print(f"{res['day'].nunique()} entry days with scores, {res['day'].min()} .. {res['day'].max()}")
    table = res.groupby("confidence").agg(
        active_share_p25=("active_share", lambda x: x.quantile(0.25)),
        active_share_median=("active_share", "median"),
        active_share_p75=("active_share", lambda x: x.quantile(0.75)),
        max_weight_median=("max_weight", "median"),
        names_held_min=("names_held", "min"))
    print(table.round(3).to_string())
    print("\nVIEW_LEVELS:", signals.VIEW_LEVELS)


if __name__ == "__main__":
    main()
