"""Walk-forward training, pass 1 (Mac): one AutoGluon ensemble per target and test year.

    .venv/bin/python tools/train_walkforward.py --model daily --targets d5_pct \\
        --years 2023 --time-limit 900
    .venv/bin/python tools/train_walkforward.py --model intraday --targets h35_pct --smoke

--smoke keeps only the last 2,000 decision times of training, fits only LightGBM and
a linear model, and caps the fit at 60 s. It checks the plumbing, not the signal. It
can't be much smaller: 8 blocks must each be far longer than a 5-day label, or the
within-block purge refuses to fit.

Writes:
  output/ag/<model>/<target>/<year>/       the fitted predictor (gitignored)
  output/preds/<model>_<target>.parquet     out-of-sample predictions, all folds so far
  reports/walkforward_<model>.csv           one row per (target, year): IC, t, fit time,
                                            rows, the best model's validation score
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from icaif import data, train  # noqa: E402

SMOKE_FAMILIES = {"GBM": {}, "LR": {}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["daily", "intraday"], required=True)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--years", nargs="+", type=int, default=list(train.TEST_YEARS))
    ap.add_argument("--time-limit", type=int, default=900, help="seconds per fold")
    ap.add_argument("--presets", default="medium_quality")
    ap.add_argument("--train-start", default=None, help="daily model: first training date")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    ds = train.daily_dataset(args.train_start) if args.model == "daily" else train.intraday_dataset()
    print(f"{ds.name} dataset: {len(ds.frame):,} rows, {len(ds.feature_cols)} features "
          f"({time.time() - t0:.0f}s to build)", flush=True)

    preds_dir = data.ROOT / "output" / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    summary_path = data.ROOT / "reports" / f"walkforward_{ds.name}{'_smoke' if args.smoke else ''}.csv"
    rows = []
    for target in args.targets:
        preds = []
        for year in args.years:
            tr, te = train.fold_split(ds, target, year)
            if te.empty:
                print(f"{target} {year}: no test rows, skipped")
                continue
            if args.smoke:
                keep = tr.index.get_level_values(ds.time_level).unique()
                tr = tr[tr.index.get_level_values(ds.time_level).isin(keep.sort_values()[-2000:])]
            path = train.OUTPUT / ds.name / target / str(year)
            t1 = time.time()
            predictor = train.fit_fold(tr, ds, target, path,
                                       time_limit=60 if args.smoke else args.time_limit,
                                       presets=args.presets,
                                       families=SMOKE_FAMILIES if args.smoke else None)
            fit_s = time.time() - t1
            p = pd.Series(predictor.predict(te[ds.feature_cols].reset_index(drop=True)).to_numpy(),
                          index=te.index, name="pred")
            res = train.evaluate(p, te, target, ds)
            board = predictor.leaderboard(silent=True)
            rows.append({"target": target, "year": year, "train_rows": len(tr), "test_rows": len(te),
                         "fit_seconds": round(fit_s), "best_model": predictor.model_best,
                         "best_val_spearman": round(float(board["score_val"].max()), 4),
                         "n_models": len(board), **{k: round(v, 4) if isinstance(v, float) else v
                                                     for k, v in res.items()}})
            print(rows[-1], flush=True)
            preds.append(p.to_frame().assign(target=te[target], year=year))
        if preds:
            pd.concat(preds).to_parquet(preds_dir / f"{ds.name}_{target}{'_smoke' if args.smoke else ''}.parquet")
    if rows:
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"-> {summary_path}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
