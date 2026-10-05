"""Benchmark ContestSolver against the repo's fixed XGBoost base learner.

Protocol (per dataset, per repeat):
  1. Stratified 80/20 outer split. The 20% holdout is never seen by anything.
  2. ``xgb_fixed``: the benchmark's existing base learner
     (``benchmarks.feature_gen.evaluator``), early-stopped on a 10% inner split.
  3. ``ContestSolver``: 5-fold LightGBM / XGBoost / CatBoost on the 80%,
     test predictions averaged over fold models, hill-climbed weights fitted on
     OOF only.
  4. Every arm is scored on the same holdout.

Datasets are PMLB files (``<name>.tsv.gz`` with a ``target`` column) read from
``--data-dir``. Results are appended to a CSV so runs can be resumed.

    python scripts/bench_contest.py --data-dir ~/data --out reports/contest.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabularaml.benchmarks.feature_gen.evaluator import (  # noqa: E402
    build_base_learner, split_early_stopping_validation,
)
from tabularaml.contest import ContestSolver, get_metric, infer_task  # noqa: E402
from tabularaml.contest.ensemble import blend  # noqa: E402

DEFAULT_DATASETS = [
    "adult", "churn", "phoneme", "spambase", "magic", "twonorm", "ring",
    "satimage", "letter", "texture", "car_evaluation", "waveform_40",
    "564_fried", "197_cpu_act", "503_wind", "344_mv", "1201_BNG_breastTumor",
    "529_pollen", "201_pol", "537_houses", "215_2dplanes", "1028_SWD",
]


def load(data_dir: Path, name: str, max_rows: int, seed: int = 0):
    df = pd.read_csv(data_dir / f"{name}.tsv.gz", sep="\t")
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    y = df.pop("target")
    return df, y


def xgb_fixed(X_tr, y_tr, X_te, task, n_classes, seed):
    t = "regression" if task == "regression" else "classification"
    X_fit, X_val, y_fit, y_val = split_early_stopping_validation(X_tr, y_tr, t, seed)
    model = build_base_learner(t, n_classes, seed, n_jobs=-1)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    if task == "regression":
        return model.predict(X_te)
    p = model.predict_proba(X_te)
    return p[:, 1] if task == "binary" else p


def run_one(name, X, y, seed, models, n_folds, metric_names):
    task = infer_task(y)
    if task != "regression":
        y = pd.Series(pd.factorize(y, sort=True)[0])
    n_classes = int(y.nunique()) if task != "regression" else 0
    strat = y if task != "regression" else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)
    X_tr, X_te = X_tr.reset_index(drop=True), X_te.reset_index(drop=True)
    y_tr, y_te = y_tr.reset_index(drop=True).to_numpy(), y_te.reset_index(drop=True).to_numpy()

    metrics = [get_metric(m) for m in metric_names[task]]
    primary = metrics[0]
    preds, secs = {}, {}

    t0 = time.time()
    preds["xgb_fixed"] = xgb_fixed(X_tr, y_tr, X_te, task, n_classes, seed)
    secs["xgb_fixed"] = time.time() - t0

    t0 = time.time()
    solver = ContestSolver(task=task, metric=primary, models=models, n_folds=n_folds,
                           seeds=(seed,), verbose=False).fit(X_tr, y_tr, X_te)
    secs["ensemble"] = time.time() - t0
    for k, v in solver.test_pred_.items():
        preds[k] = v
    preds["mean"] = blend(list(solver.test_pred_.values()),
                          np.full(len(solver.test_pred_), 1 / len(solver.test_pred_)))
    preds["ensemble"] = solver.test_ensemble_

    rows = []
    for arm, p in preds.items():
        row = dict(dataset=name, seed=seed, task=task, n_train=len(X_tr), n_features=X.shape[1],
                   arm=arm, seconds=secs.get(arm, np.nan),
                   oof_score=solver.oof_scores_.get(arm, solver.oof_ensemble_score_ if arm == "ensemble" else np.nan),
                   weights=json.dumps(solver.weights_) if arm == "ensemble" else "")
        scores = {m.name: m(y_te, p) for m in metrics}
        row["primary_metric"] = primary.name
        row["test_primary"] = scores[primary.name]
        row["test_metrics"] = json.dumps(scores)
        rows.append(row)
    return rows


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path.home() / "data")
    ap.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--models", nargs="*", default=["lgbm", "xgb", "catboost"])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--max-rows", type=int, default=20_000)
    ap.add_argument("--out", type=Path, default=Path("reports/contest_bench.csv"))
    args = ap.parse_args()

    metric_names = {
        "regression": ["rmse", "mae", "r2"],
        "binary": ["logloss", "auc", "accuracy"],
        "multiclass": ["logloss", "accuracy"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        prev = pd.read_csv(args.out)
        done = set(zip(prev.dataset, prev.seed))

    for name in args.datasets:
        X, y = load(args.data_dir, name, args.max_rows)
        for seed in args.seeds:
            if (name, seed) in done:
                continue
            t0 = time.time()
            rows = run_one(name, X, y, seed, args.models, args.n_folds, metric_names)
            df = pd.DataFrame(rows)
            df.to_csv(args.out, mode="a", header=not args.out.exists(), index=False)
            base = df.loc[df.arm == "xgb_fixed", "test_primary"].iloc[0]
            ens = df.loc[df.arm == "ensemble", "test_primary"].iloc[0]
            print(f"{name:<22} seed={seed} {df.primary_metric.iloc[0]}: xgb_fixed={base:.5f} "
                  f"ensemble={ens:.5f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
