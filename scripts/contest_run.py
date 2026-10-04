"""One-command contest pipeline: FeatureForge features + ContestSolver ensemble.

    python scripts/contest_run.py --train train.csv --test test.csv --target target \\
        --id id --metric auc --budget 600 --out submission.csv

Steps:
  1. FeatureForge searches features on the training rows (the test rows only
     contribute to unsupervised count/group statistics, never to targets).
  2. ContestSolver trains K-fold LightGBM / XGBoost / CatBoost on the
     engineered features and hill-climbs ensemble weights on OOF predictions.
  3. The submission holds probabilities (classification metrics that score
     probabilities), class labels (label metrics) or values (regression).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabularaml.contest import ContestSolver, get_metric, infer_task  # noqa: E402
from tabularaml.generate.forge import FeatureForge  # noqa: E402

PROBA_METRICS = {"logloss", "auc"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--id", default=None, help="id column copied to the submission and excluded from features")
    ap.add_argument("--metric", default=None)
    ap.add_argument("--task", default=None, choices=["regression", "binary", "multiclass"])
    ap.add_argument("--budget", type=float, default=600, help="FeatureForge time budget (s)")
    ap.add_argument("--no-fe", action="store_true")
    ap.add_argument("--models", nargs="*", default=["lgbm", "xgb", "catboost"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    train, test = pd.read_csv(args.train), pd.read_csv(args.test)
    y = train.pop(args.target)
    ids = test[args.id] if args.id else pd.Series(range(len(test)), name="id")
    drop = [args.id] if args.id else []
    X, X_test = train.drop(columns=drop), test.drop(columns=drop)[train.drop(columns=drop).columns]
    task = args.task or infer_task(y)
    metric = get_metric(args.metric) if args.metric else None

    if not args.no_fe:
        if task != "regression":
            y_fe = pd.Series(pd.factorize(y, sort=True)[0])
        else:
            y_fe = y
        forge = FeatureForge(task=task, log_target=(args.metric == "rmsle"), time_budget=args.budget,
                             n_jobs=args.threads)
        forge.fit(X, y_fe, X_unlabeled=X_test)
        X, X_test = forge.transform_train(X), forge.transform(X_test)
        print(f"FeatureForge added {len(forge.new_columns_)} features")

    solver = ContestSolver(task=task, metric=metric, models=args.models, n_folds=args.folds,
                           seeds=args.seeds, n_jobs=args.threads).fit(X, y, X_test)
    print(solver.leaderboard_.to_string(index=False))

    p = solver.test_ensemble_
    if task == "regression":
        pred = p
    elif solver.metric_.name in PROBA_METRICS:
        if p.ndim == 2:
            sub = pd.DataFrame(p, columns=[str(c) for c in solver.label_encoder_.classes_])
            sub.insert(0, ids.name, ids.to_numpy())
            sub.to_csv(args.out, index=False)
            print(f"wrote {args.out}")
            return
        pred = p
    else:
        pred = solver.predict(X_test)
    pd.DataFrame({ids.name: ids.to_numpy(), args.target: pred}).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
