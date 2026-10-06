"""Contest feature pipeline: related tables + child-row models + FeatureForge.

Features only; train any AutoML (AutoGluon, ...) on the outputs.

    python scripts/contest_features.py --train application_train.csv --test application_test.csv \\
        --target TARGET --id SK_ID_CURR \\
        --table bureau=bureau.csv --table prev=previous_application.csv \\
        --table inst=installments_payments.csv --child-models --out-dir features/

Each ``--table name=path[:key[:time]]`` is a child table. The key defaults to the
main table's id column (or the one column it shares with the main table); the
time column defaults to the first column named like a date or a day/month
count. Writes ``train_features.parquet`` and ``test_features.parquet``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabularaml.generate.forge import FeatureForge  # noqa: E402
from tabularaml.generate.relational import Child, RelatedTables, child_model_features  # noqa: E402

TIME_HINTS = ("days", "day", "month", "date", "time", "week", "year")


def read(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("category")
        elif df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def parse_table(spec: str, main: pd.DataFrame, id_col: str | None) -> Child:
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    df = read(parts[0])
    key = parts[1] if len(parts) > 1 and parts[1] else None
    if key is None:
        shared = [c for c in df.columns if c in main.columns]
        key = id_col if id_col in df.columns else (shared[0] if len(shared) == 1 else None)
    if key is None:
        raise SystemExit(f"--table {name}: give the key column explicitly (name=path:key)")
    tcol = parts[2] if len(parts) > 2 else next(
        (c for c in df.columns if c != key and pd.api.types.is_numeric_dtype(df[c])
         and any(h in c.lower() for h in TIME_HINTS)), None)
    # Other id-like columns (foreign keys to further tables) are not features.
    drop = [c for c in df.columns if c != key and (c.upper().startswith(("SK_ID", "ID_")) or c.lower().endswith("_id"))]
    print(f"table {name}: {df.shape}, key={key}, time={tcol}, ignored ids={drop}", flush=True)
    return Child(name, df, key=key, time=tcol, drop=[c for c in drop if c != key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--id", default=None)
    ap.add_argument("--table", action="append", default=[], help="name=path[:key[:time]]")
    ap.add_argument("--child-models", action="store_true", help="add out-of-fold child-row model features")
    ap.add_argument("--task", default=None, choices=["regression", "binary", "multiclass"])
    ap.add_argument("--budget", type=float, default=900, help="FeatureForge time budget (s)")
    ap.add_argument("--top-related", type=int, default=200,
                    help="related-table columns handed to FeatureForge's search (all are kept in the output)")
    ap.add_argument("--out-dir", default="features")
    a = ap.parse_args()

    t0 = time.time()
    tr, te = read(a.train), read(a.test)
    y = tr.pop(a.target)
    ids_tr = tr[a.id].to_numpy() if a.id else np.arange(len(tr))
    ids_te = te[a.id].to_numpy() if a.id else np.arange(len(tr), len(tr) + len(te))
    rel_tr = rel_te = None
    if a.table:
        children = [parse_table(s, tr, a.id) for s in a.table]
        F = RelatedTables(children).features()
        if a.child_models:
            task = a.task or ("binary" if y.nunique() == 2 else "regression")
            obj = "binary" if task == "binary" else "regression"
            yk = pd.Series(y.to_numpy(), index=ids_tr)
            for ch in children:
                t = time.time()
                F = F.join(child_model_features(ch, yk, ids_te, task=obj), how="outer")
                print(f"child model {ch.name}: {time.time() - t:.0f}s", flush=True)
        rel_tr, rel_te = F.reindex(ids_tr).reset_index(drop=True), F.reindex(ids_te).reset_index(drop=True)
        for c in [c for c in F.columns if c.endswith("__count")]:
            rel_tr[c], rel_te[c] = rel_tr[c].fillna(0), rel_te[c].fillna(0)
        print(f"related tables: {F.shape[1]} columns in {time.time() - t0:.0f}s", flush=True)

    Xtr = tr.drop(columns=[a.id]) if a.id else tr
    Xte = te.drop(columns=[a.id]) if a.id else te
    Xte = Xte[Xtr.columns]
    if rel_tr is not None:
        # The search sees the related columns a quick model uses most; all are written out.
        import lightgbm as lgb
        both = pd.concat([Xtr.reset_index(drop=True), rel_tr], axis=1)
        b = lgb.train(dict(objective="binary" if y.nunique() == 2 else "regression", learning_rate=0.1,
                           num_leaves=31, feature_fraction=0.5, verbose=-1), lgb.Dataset(both, y), 300)
        gain = pd.Series(b.feature_importance("gain"), index=both.columns)[rel_tr.columns]
        top = list(gain.sort_values(ascending=False).index[:a.top_related])
        Xtr = both[list(Xtr.columns) + top]
        Xte = pd.concat([Xte.reset_index(drop=True), rel_te[top]], axis=1)
    forge = FeatureForge(task=a.task, time_budget=a.budget).fit(Xtr, y, X_unlabeled=Xte)
    out_tr, out_te = forge.transform_train(Xtr), forge.transform(Xte)
    if rel_tr is not None:
        rest = [c for c in rel_tr.columns if c not in out_tr.columns]
        out_tr = pd.concat([out_tr, rel_tr[rest]], axis=1)
        out_te = pd.concat([out_te, rel_te[rest]], axis=1)
    out_tr[a.target] = y.to_numpy()
    if a.id:
        out_tr.insert(0, a.id, ids_tr)
        out_te.insert(0, a.id, ids_te)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_tr.to_parquet(out / "train_features.parquet")
    out_te.to_parquet(out / "test_features.parquet")
    print(f"done in {time.time() - t0:.0f}s: {out_tr.shape[1] - tr.shape[1] - 1} columns added -> {out}/")


if __name__ == "__main__":
    main()
