"""Contest feature pipeline: related tables + child-row models + FeatureForge.

Features only; train any AutoML (AutoGluon, ...) on the outputs.

    python scripts/contest_features.py --train application_train.csv --test application_test.csv \\
        --target TARGET --id SK_ID_CURR \\
        --table bureau=bureau.csv --table prev=previous_application.csv \\
        --table inst=installments_payments.csv --child-models --out-dir features/

Each ``--table name=path[:key[:time]]`` is a child table. The key defaults to the
main table's id column (or the one column it shares with the main table); the
time column defaults to the first column named like a date or a day/month
count. When the main table has several rows per key (assessments of a player,
orders of a user) and a time column comparable to the child's (the same name,
or ``--time``), the child is aggregated as of each main row: only its earlier
rows count (``asof_features``). Writes ``train_features.parquet`` and ``test_features.parquet``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabularaml.generate.forge import FeatureForge  # noqa: E402
from tabularaml.generate.history import history_features, repeats  # noqa: E402
from tabularaml.generate.relational import (Child, RelatedTables, asof_features, child_model_features,  # noqa: E402
                                             lookup_features, match_features)

TIME_HINTS = ("days", "day", "month", "date", "time", "week", "year")


def read(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False)
    for c in df.columns:
        if df[c].dtype == object:
            # Mixed 0 / "0" (pandas reads a CSV column chunk by chunk) become one level.
            df[c] = df[c].where(df[c].isna(), df[c].astype(str)).astype("category")
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
        # No shared column: a lookup table (components, products) whose first column's values
        # appear in main-table columns, possibly numbered slots (component_id_1..8).
        print(f"table {name}: {df.shape}, lookup on {df.columns[0]}", flush=True)
        return Child(name, df, key=df.columns[0], drop=["__lookup__"])
    tcol = parts[2] if len(parts) > 2 else next(
        (c for c in df.columns if c != key and pd.api.types.is_numeric_dtype(df[c])
         and any(h in c.lower() for h in TIME_HINTS)), None)
    # Other id-like columns (foreign keys to further tables) are not features.
    drop = [c for c in df.columns if c != key and (c.upper().startswith(("SK_ID", "ID_")) or c.lower().endswith("_id"))]
    print(f"table {name}: {df.shape}, key={key}, time={tcol}, ignored ids={drop}", flush=True)
    return Child(name, df, key=key, time=tcol, drop=[c for c in drop if c != key])


def date_col(df: pd.DataFrame) -> str | None:
    """The first text column holding calendar dates (2013-04-22, 2016-01-01 19:00:00)."""
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        v = df[c].dropna().astype(str).head(1000)
        if len(v) and v.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.95:
            return c
    return None


def match_cols(main: pd.DataFrame, ch: Child) -> list[str]:
    """Code columns a child shares with the main table besides the key (category, brand, company):
    integer or text codes with 20+ levels whose main-table values mostly occur in the child."""
    out = []
    for c in ch.df.columns:
        if c == ch.key or c == ch.time or c in ch.drop or c not in main.columns or any(h in c.lower() for h in TIME_HINTS):
            continue
        s = ch.df[c]
        if pd.api.types.is_float_dtype(s) and not np.all(np.mod(s.dropna().to_numpy()[:10000], 1) == 0):
            continue
        if s.nunique() < 20:
            continue
        vals = pd.unique(main[c].dropna().astype(str))
        if len(vals) and np.isin(vals, pd.unique(s.dropna().astype(str))).mean() >= 0.5:
            out.append(c)
    return out


def main_time(main: pd.DataFrame, ch: Child, given: str | None) -> str | None:
    """The main table's column that is comparable with the child's time column."""
    if given:
        return given
    if ch.time in main.columns:
        return ch.time
    cands = [c for c in main.columns if pd.api.types.is_numeric_dtype(main[c])
             and any(h in c.lower() for h in TIME_HINTS)]
    return cands[0] if len(cands) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--id", default=None)
    ap.add_argument("--table", action="append", default=[], help="name=path[:key[:time]]")
    ap.add_argument("--time", default=None, help="main-table time column for as-of aggregation (default: auto)")
    ap.add_argument("--child-models", action="store_true", help="add out-of-fold child-row model features")
    ap.add_argument("--task", default=None, choices=["regression", "binary", "multiclass"])
    ap.add_argument("--log-target", action="store_true", help="search on log1p(target) (RMSLE-scored contests)")
    ap.add_argument("--budget", type=float, default=900, help="FeatureForge time budget (s)")
    ap.add_argument("--top-related", type=int, default=200,
                    help="related-table columns handed to FeatureForge's search (all are kept in the output)")
    ap.add_argument("--out-dir", default="features")
    ap.add_argument("--forge-kw", default="{}", help="JSON of extra FeatureForge arguments")
    ap.add_argument("--forecast", default="auto", choices=["auto", "off"],
                    help="forecasting family (tabularaml/generate/forecast.py): on when the data has a date, "
                         "repeating entity keys and test rows after the training period")
    ap.add_argument("--forecast-kw", default="{}", help="JSON of extra ForecastFeatures arguments")
    ap.add_argument("--history", default="off", choices=["auto", "off"],
                    help="latest-state features of keyed child tables with a time column "
                         "(tabularaml/generate/history.py): last, last - mean, last - previous")
    a = ap.parse_args()

    t0 = time.time()
    tr, te = read(a.train), read(a.test)
    y = tr.pop(a.target)
    ids_tr = tr[a.id].to_numpy() if a.id else np.arange(len(tr))
    ids_te = te[a.id].to_numpy() if a.id else np.arange(len(tr), len(tr) + len(te))
    rel_tr = rel_te = None
    if a.table:
        children = [parse_table(s, tr, a.id) for s in a.table]
        lookups = [ch for ch in children if "__lookup__" in ch.drop]
        children = [ch for ch in children if ch not in lookups]
        # One row per key: attributes of the main row (a tube's dimensions, its bill of
        # materials), joined as columns before anything is aggregated or looked up.
        for ch in [ch for ch in children if ch.key in tr.columns and not ch.df[ch.key].duplicated().any()]:
            cols = [c for c in ch.df.columns if c == ch.key or c not in tr.columns]
            tr = tr.merge(ch.df[cols], on=ch.key, how="left")
            te = te.merge(ch.df[cols], on=ch.key, how="left")
            children.remove(ch)
            print(f"table {ch.name}: one row per {ch.key}, joined as columns", flush=True)
        both_main = pd.concat([tr, te[[c for c in tr.columns if c in te.columns]]], ignore_index=True)
        asof = {ch.name: main_time(tr, ch, a.time) for ch in children
                if ch.time is not None and ch.key in tr.columns and tr[ch.key].duplicated().any()}
        asof = {k: v for k, v in asof.items() if v is not None}
        keyed = [ch for ch in children if ch.name not in asof]
        F = pd.DataFrame()
        A = []
        for ch in children:
            if ch.name in asof:
                print(f"table {ch.name}: as of {asof[ch.name]} per main row", flush=True)
                A.append(asof_features(both_main, ch.key, asof[ch.name], ch))
        if a.child_models and keyed:
            task = a.task or ("binary" if y.nunique() == 2 else "regression")
            obj = "binary" if task == "binary" else "regression"
            for ch in keyed:  # as-of children would see later rows' outcomes
                if not tr[ch.key].is_unique:
                    continue  # labels per key are ambiguous when the key repeats
                t = time.time()
                yk = pd.Series(y.to_numpy(), index=tr[ch.key].to_numpy())
                M = child_model_features(ch, yk, te[ch.key].to_numpy(), task=obj)
                A.append(M.reindex(both_main[ch.key].to_numpy()).set_index(both_main.index))
                print(f"child model {ch.name}: {time.time() - t:.0f}s", flush=True)
        for ch in keyed:
            Fk = RelatedTables([ch]).features()
            Fk = Fk.reindex(both_main[ch.key].to_numpy()).set_index(both_main.index)
            for c in [c for c in Fk.columns if c.endswith("__count")]:
                Fk[c] = Fk[c].fillna(0)
            A.insert(0, Fk)
            if a.history == "auto" and repeats(ch):
                t = time.time()
                Hk = history_features(ch).reindex(both_main[ch.key].to_numpy()).set_index(both_main.index)
                print(f"table {ch.name}: latest-state history, {Hk.shape[1]} columns in {time.time() - t:.0f}s", flush=True)
                A.insert(1, Hk)
        rel_tr, rel_te = pd.DataFrame(index=range(len(tr))), pd.DataFrame(index=range(len(te)))
        # Child rows that also share the main row's code values (a customer's purchases of the
        # offer's brand), counted before the main row's date when both tables carry dates.
        mt = date_col(both_main)
        for ch in keyed + [ch for ch in children if ch.name in asof]:
            cols = match_cols(both_main, ch)
            if not cols:
                continue
            ct = ch.time if ch.time is not None else date_col(ch.df.drop(columns=[ch.key]))
            if both_main[ch.key].duplicated().any() and (mt is None or ct is None):
                continue  # repeated keys without dates could match a row's own later events
            chm = Child(ch.name, ch.df, key=ch.key, time=ct, drop=ch.drop)
            t = time.time()
            Mf = match_features(both_main, ch.key, chm, cols, main_time=mt if ct is not None else None)
            print(f"table {ch.name}: same-{'/'.join(cols)} matches{' before ' + mt if mt and ct else ''}: "
                  f"{Mf.shape[1]} columns in {time.time() - t:.0f}s", flush=True)
            A.append(Mf)
        for lk in lookups:
            L = lookup_features(both_main, lk.df, lk.key, lk.name)
            print(f"table {lk.name}: {L.shape[1]} lookup columns", flush=True)
            A.append(L)
        for Fa in A:
            rel_tr = pd.concat([rel_tr, Fa.iloc[:len(tr)].reset_index(drop=True)], axis=1)
            rel_te = pd.concat([rel_te, Fa.iloc[len(tr):].reset_index(drop=True)], axis=1)
        print(f"related tables: {rel_tr.shape[1]} columns in {time.time() - t0:.0f}s", flush=True)

    Xtr = tr.drop(columns=[a.id]) if a.id else tr
    Xte = te.drop(columns=[a.id]) if a.id else te
    Xte = Xte[Xtr.columns]
    if a.forecast == "auto":
        from tabularaml.generate.forecast import ForecastFeatures
        t = time.time()
        ff = ForecastFeatures(**json.loads(a.forecast_kw)).fit(Xtr, y.to_numpy(), Xte)
        if ff.active_:
            Ftr, Fte = ff.transform(Xtr), ff.transform(Xte)
            Xtr = pd.concat([Xtr.reset_index(drop=True), Ftr.reset_index(drop=True)], axis=1)
            Xte = pd.concat([Xte.reset_index(drop=True), Fte.reset_index(drop=True)], axis=1)
            print(f"forecast: {Ftr.shape[1]} columns in {time.time() - t:.0f}s", flush=True)
    if rel_tr is not None:
        # The search sees the related columns a quick model uses most; all are written out.
        import lightgbm as lgb
        both = pd.concat([Xtr.reset_index(drop=True), rel_tr], axis=1)
        for c in both.columns:
            if not (pd.api.types.is_numeric_dtype(both[c]) or isinstance(both[c].dtype, pd.CategoricalDtype)):
                both[c] = both[c].astype(str).astype("category")
        b = lgb.train(dict(objective="binary" if y.nunique() == 2 else "regression", learning_rate=0.1,
                           num_leaves=31, feature_fraction=0.5, verbose=-1), lgb.Dataset(both, np.log1p(y) if a.log_target else y), 300)
        gain = pd.Series(b.feature_importance("gain"), index=both.columns)[rel_tr.columns]
        top = list(gain.sort_values(ascending=False).index[:a.top_related])
        Xtr = both[list(Xtr.columns) + top]
        Xte = pd.concat([Xte.reset_index(drop=True), rel_te[top]], axis=1)
    forge = FeatureForge(task=a.task, time_budget=a.budget, log_target=a.log_target, **json.loads(a.forge_kw)).fit(Xtr, y, X_unlabeled=Xte)
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
