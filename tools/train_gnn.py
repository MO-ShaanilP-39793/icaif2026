"""Walk-forward training of the cross-sectional attention model, and its stage-1 gate.

    .venv/bin/python tools/train_gnn.py --smoke                  # plumbing, ~1 minute
    .venv/bin/python tools/train_gnn.py --years 2023 --seeds 1   # one fold, timed
    .venv/bin/python tools/train_gnn.py                          # four folds x 3 seeds

Folds are pass 1's test years. Within each, the year before the test year is the
validation year that picks the epoch (`gnn_data.split_days`, purged both ways), and
the model trains on everything whose label ended before it. `--refit` then retrains
from scratch on train + validation for the chosen number of epochs, so the model has
seen the same years as the AutoGluon ensemble it is compared with. Without it, the
GNN trains on one year less, which handicaps it in exactly the fold where recent
regimes matter.

Seeds are averaged as within-day ranks, so a seed with larger raw scores doesn't
dominate the mean.

**The gate** (design doc): on the 30 names, 2023-26 out of sample, does the GNN beat
or diversify the daily ensemble (`output/preds/daily_d5_pct.parquet`)? Reported: each
model's IC, their per-day rank correlation, and a 50/50 rank blend's IC with the
t-stat of its gain over the ensemble alone. t-stats are overlap-adjusted for the
5-day label, as in pass 1.

Writes:
  output/gnn/<target>/<year>/seed<k>.pt        weights and config (gitignored)
  output/preds/gnn_<target>.parquet            out-of-sample scores on the 30, all folds
  reports/walkforward_gnn.csv                  one row per (year, seed)
  reports/gnn_gate.csv                         the gate, per year and pooled
"""

import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from icaif import data, external, gnn, gnn_data, train  # noqa: E402

STEPS = 5  # label horizon in sessions, for the overlap-adjusted t-stat


def _t(daily: pd.Series) -> float:
    return float(daily.mean() / (daily.std() / np.sqrt(len(daily) / STEPS)))


def gate(g: pd.DataFrame, ens: pd.DataFrame) -> pd.DataFrame:
    """g, ens: index (date, ticker), columns pred, target (the 30 names only)."""
    both = g[["pred", "target"]].join(ens[["pred"]], rsuffix="_ens", how="inner").dropna()
    both["year"] = both.index.get_level_values("date").year
    rows = []
    for year, b in list(both.groupby("year")) + [("2023-26", both)]:
        day = b.groupby(level="date")
        rg, re_ = day["pred"].rank(pct=True), day["pred_ens"].rank(pct=True)
        ic_g = train.per_decision_ic(rg, b["target"], "date")
        ic_e = train.per_decision_ic(re_, b["target"], "date")
        ic_b = train.per_decision_ic((rg + re_) / 2, b["target"], "date")
        corr = train.per_decision_ic(rg, re_, "date")
        gain = (ic_b - ic_e).dropna()
        rows.append({"year": year, "days": len(ic_g), "ic_gnn": ic_g.mean(), "t_gnn": _t(ic_g),
                     "ic_ensemble": ic_e.mean(), "t_ensemble": _t(ic_e),
                     "rank_corr": corr.mean(), "ic_blend": ic_b.mean(),
                     "blend_gain": gain.mean(), "t_gain": _t(gain)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="d5_pct")
    ap.add_argument("--years", nargs="+", type=int, default=list(train.TEST_YEARS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="last 300 training days, 2 epochs, 1 seed, 2023 only: plumbing, not signal")
    args = ap.parse_args()

    cfg = gnn.Config()
    if args.smoke:
        cfg = dataclasses.replace(cfg, max_epochs=2, train_days=300)
        args.seeds, args.years = 1, [2023]
    if args.max_epochs:
        cfg = dataclasses.replace(cfg, max_epochs=args.max_epochs)
    tag = "_smoke" if args.smoke else ""

    t0 = time.time()
    ds = train.daily_dataset()
    t = gnn_data.build(ds.frame, external.load("yahoo_daily_universe"), target=args.target)
    dev = gnn.device()
    probe = gnn.CrossSectionalRanker(t.x.shape[2], t.ctx.shape[1], cfg)
    print(f"tensors: {len(t.dates)} sessions x {len(t.tickers)} names x {t.x.shape[2]} features, "
          f"{t.ctx.shape[1]} context; {t.x.nbytes / 1e6:.0f} MB; model {gnn.n_params(probe):,} params; "
          f"device {dev} ({time.time() - t0:.0f}s)", flush=True)

    out_dir = data.ROOT / "output" / "gnn" / args.target
    rows, preds = [], []
    for year in args.years:
        split = gnn_data.split_days(t, year)
        if not len(split.test):
            print(f"{year}: no test days, skipped")
            continue
        print(f"{year}: {len(split.train)} train, {len(split.val)} val, {len(split.test)} test days", flush=True)
        per_seed = []
        for seed in range(args.seeds):
            res = gnn.fit(t, split.train, split.val, cfg, seed=seed, dev=dev,
                          log=lambda r: print("   ", r, flush=True))
            fit_s, used = res.seconds, res
            if args.refit:
                days = np.concatenate([split.train, split.val])
                used = gnn.fit(t, days, None, cfg, seed=seed, dev=dev, epochs=res.best_epoch,
                               log=lambda r: print("    refit", r, flush=True))
                fit_s += used.seconds
            s = gnn.score_competition(used, split.test)
            path = out_dir / str(year) / f"seed{seed}{tag}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": used.model.state_dict(), "config": dataclasses.asdict(cfg),
                        "ctx_names": t.ctx_names, "features": t.feature_names}, path)
            truth = pd.Series(t.y[t.dates.get_indexer(s.index.get_level_values("date")),
                                  t.tickers.get_indexer(s.index.get_level_values("ticker"))],
                              index=s.index)
            ic = train.per_decision_ic(s, truth, "date")
            rows.append({"target": args.target, "year": year, "seed": seed, "refit": args.refit,
                         "train_days": len(split.train), "best_epoch": res.best_epoch,
                         "val_ic": round(res.best_val_ic, 4),
                         "val_ic_30": round(res.history[res.best_epoch - 1].get("val_ic_30", np.nan), 4),
                         "test_ic_30": round(float(ic.mean()), 4), "fit_seconds": round(fit_s)})
            print(rows[-1], flush=True)
            per_seed.append(s.groupby(level="date").rank(pct=True))
        mean = pd.concat(per_seed, axis=1).mean(axis=1).rename("pred")
        truth = pd.Series(t.y[t.dates.get_indexer(mean.index.get_level_values("date")),
                              t.tickers.get_indexer(mean.index.get_level_values("ticker"))],
                          index=mean.index)
        preds.append(mean.to_frame().assign(target=truth, year=year))

    if not preds:
        return
    g = pd.concat(preds)
    (data.ROOT / "output" / "preds").mkdir(parents=True, exist_ok=True)
    g.to_parquet(data.ROOT / "output" / "preds" / f"gnn_{args.target}{tag}.parquet")
    pd.DataFrame(rows).to_csv(data.ROOT / "reports" / f"walkforward_gnn{tag}.csv", index=False)

    ens_path = data.ROOT / "output" / "preds" / f"daily_{args.target}.parquet"
    if ens_path.exists():
        ens = pd.read_parquet(ens_path)
        ens = ens[ens.index.get_level_values("ticker").isin(set(data.load_universe()))]
        table = gate(g, ens)
        table.to_csv(data.ROOT / "reports" / f"gnn_gate{tag}.csv", index=False)
        print("\nStage-1 gate, the 30 names, out of sample:")
        print(table.round(4).to_string(index=False))
    print(f"(total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
