"""IEEE-CIS Fraud Detection (Kaggle 2019, $20k): raw columns vs FeatureForge on a
time-ordered holdout (latest 20% of transactions), as the competition's test set was.

Data come from an open Hugging Face mirror of the competition files.

    python scripts/ieee_fraud.py --frac 0.4 --budget 900
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tabularaml.generate.forge import FeatureForge  # noqa: E402

warnings.filterwarnings("ignore")
MIRROR = "https://huggingface.co/datasets/aliceczr/ieee-fraud-detection/resolve/main/"


def load(cache: Path) -> pd.DataFrame:
    pq = cache / "ieee.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    cache.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(MIRROR + "train_transaction.csv")
    i = pd.read_csv(MIRROR + "train_identity.csv")
    df = t.merge(i, on="TransactionID", how="left")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("category")
        elif df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    df.to_parquet(pq)
    return df


def judge(Xtr, ytr, Xte, seed=0):
    """One LightGBM, early-stopped on a random 15% of training rows, refitted on all."""
    for c in Xtr.columns:
        if not pd.api.types.is_numeric_dtype(Xtr[c]):
            u = pd.Categorical(pd.concat([Xtr[c], Xte[c]]).astype(str)).categories
            Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=u)
            Xte[c] = pd.Categorical(Xte[c].astype(str), categories=u)
    P = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=100, feature_fraction=0.5,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_cat_to_onehot=8, cat_smooth=50,
             num_threads=4, verbose=-1, seed=seed)
    perm = np.random.default_rng(seed).permutation(len(Xtr))
    tr, va = np.sort(perm[: int(0.85 * len(perm))]), np.sort(perm[int(0.85 * len(perm)):])
    b = lgb.train(P, lgb.Dataset(Xtr.iloc[tr], ytr.iloc[tr]), 3000,
                  valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr.iloc[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    return lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.4, help="most recent fraction of the 590k rows to use")
    ap.add_argument("--budget", type=float, default=900)
    ap.add_argument("--forge-kw", default="{}")
    ap.add_argument("--shuffle", action="store_true", help="leakage control: permute the training labels")
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "ieee_fraud")
    a = ap.parse_args()
    df = load(a.cache)
    df = df.iloc[int(len(df) * (1 - a.frac)):].reset_index(drop=True)
    y = df.pop("isFraud")
    df = df.drop(columns=["TransactionID"])
    n = int(0.8 * len(df))
    Xtr, Xte = df.iloc[:n].reset_index(drop=True), df.iloc[n:].reset_index(drop=True)
    ytr, yte = y.iloc[:n].reset_index(drop=True), y.iloc[n:].reset_index(drop=True)
    if a.shuffle:
        ytr = pd.Series(np.random.default_rng(0).permutation(ytr.to_numpy()))
    p = judge(Xtr.copy(), ytr, Xte.copy())
    print(f"raw    AUC={roc_auc_score(yte, p):.4f} logloss={log_loss(yte, p):.5f}", flush=True)
    t0 = time.time()
    f = FeatureForge(task="binary", time_budget=a.budget, n_jobs=4, **json.loads(a.forge_kw)).fit(Xtr, ytr, X_unlabeled=Xte)
    A, B = f.transform_train(Xtr), f.transform(Xte)
    p = judge(A, ytr, B)
    print(f"forge  AUC={roc_auc_score(yte, p):.4f} logloss={log_loss(yte, p):.5f} "
          f"({len(f.new_columns_)} features, {time.time() - t0:.0f}s)")
    print("features:", ", ".join(f.new_columns_), flush=True)


if __name__ == "__main__":
    main()
