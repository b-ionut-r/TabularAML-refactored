"""IEEE-CIS reference entry: the winners' public hand features on Kaggle's own files.

Reads the contest's train/test transaction and identity CSVs, computes ``ieee_hand.hand_features``
over training and test rows together (label-free, as the winners did) and writes
``train_features.parquet`` (with isFraud) and ``test_features.parquet`` keyed by TransactionID,
the layout ``contest_features.py`` writes, so any judge can score them.

    python scripts/ieee_hand_kaggle.py --data kaggle/ieee-fraud-detection --out features/ieee_hand
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ieee_hand import hand_features  # noqa: E402


def read(d: Path, part: str) -> pd.DataFrame:
    t = pd.read_csv(d / f"{part}_transaction.csv")
    i = pd.read_csv(d / f"{part}_identity.csv")
    i.columns = [c.replace("-", "_") for c in i.columns]  # the test file writes id-01 for id_01
    df = t.merge(i, on="TransactionID", how="left")
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0, d = time.time(), Path(a.data)
    tr, te = read(d, "train"), read(d, "test")
    y = tr.pop("isFraud")
    both = pd.concat([tr, te[tr.columns]], ignore_index=True)
    ids = both.pop("TransactionID").to_numpy()
    X = hand_features(both)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str).astype("category")
    X.insert(0, "TransactionID", ids)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Xtr = X.iloc[:len(tr)].reset_index(drop=True)
    Xtr["isFraud"] = y.to_numpy()
    Xtr.to_parquet(out / "train_features.parquet")
    X.iloc[len(tr):].reset_index(drop=True).to_parquet(out / "test_features.parquet")
    print(f"{X.shape[1] - 1} columns for {len(tr)} train + {len(te)} test rows in {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
