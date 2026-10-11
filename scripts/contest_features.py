"""Contest feature pipeline: related tables + child-row models + FeatureForge.

Features only; train any AutoML (AutoGluon, ...) on the outputs.

    python scripts/contest_features.py --train application_train.csv --test application_test.csv \\
        --target TARGET --id SK_ID_CURR \\
        --table bureau=bureau.csv --table prev=previous_application.csv \\
        --table inst=installments_payments.csv --out-dir features/

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
from tabularaml.generate.lists import as_lists, list_features, unit_ratio_features  # noqa: E402
from tabularaml.generate.returns import return_features  # noqa: E402
from tabularaml.generate.panel import panel_features  # noqa: E402
from tabularaml.generate.stream import aggregation_bytes, chunks, memory_bytes  # noqa: E402
from tabularaml.generate.history import event_log_features, history_features, repeats  # noqa: E402
from tabularaml.generate.relational import (Child, RelatedTables, asof_features, child_model_features,  # noqa: E402
                                             lookup_features, match_features)
from tabularaml.generate.strs import as_text  # noqa: E402

TIME_HINTS = ("days", "day", "month", "date", "time", "week", "year")


def log_y(y: pd.Series) -> np.ndarray:
    """The search's log target: log of a small positive target (FeatureForge does the same), else log1p."""
    v = y.to_numpy(dtype=float)
    return np.log(v) if np.nanmin(v) > 0 and np.nanmedian(v) < 1 else np.log1p(v)


def read(path: str) -> pd.DataFrame:
    return normalise(pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False))


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object and isinstance(df[c].dropna().iloc[:1].tolist()[0] if df[c].notna().any() else None,
                                                (list, tuple, np.ndarray)):
            # List cells (amenities, photo URLs): items joined, so they stay readable as a list column.
            df[c] = df[c].map(lambda v: " ; ".join(str(i).replace(";", ",") for i in v)
                              if isinstance(v, (list, tuple, np.ndarray)) else v)
        if df[c].dtype == object:
            # Mixed 0 / "0" (pandas reads a CSV column chunk by chunk) become one level.
            df[c] = df[c].where(df[c].isna(), as_text(df[c])).astype("category")
        elif df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def parse_table(spec: str, main: pd.DataFrame, id_col: str | None) -> Child:
    name, rest = spec.split("=", 1)
    parts = rest.split(":")
    key = parts[1] if len(parts) > 1 and parts[1] else None
    need = aggregation_bytes(parts[0], key) if key else None
    big = need is not None and need > STREAM_SHARE * memory_bytes() and stream_rows(parts[0]) > STREAM_MIN_ROWS
    if big and key in main.columns and main[key].is_unique:
        # (one main row per key only: a key repeating in the main table is an event log aggregated as of each row)
        # Too big to aggregate whole (Optiver's 160M-row order book): read a key range at a time.
        from tabularaml.generate.stream import _dataset
        df = normalise(_dataset(parts[0]).head(200_000).to_pandas())
        tcol = parts[2] if len(parts) > 2 else next(
            (c for c in df.columns if c != key and pd.api.types.is_numeric_dtype(df[c])
             and any(h in c.lower() for h in TIME_HINTS)), None)
        drop = [c for c in df.columns if c != key and (c.upper().startswith(("SK_ID", "ID_")) or c.lower().endswith("_id"))]
        print(f"table {name}: {_dataset(parts[0]).count_rows()} rows, key={key}, read by key range "
              f"(whole: {need / 2**30:.0f} GB to aggregate)", flush=True)
        return Child(name, df, key=key, time=tcol, drop=drop, source=parts[0])
    df = read(parts[0])
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


STREAM_SHARE, STREAM_BUDGET_SHARE = 0.4, 0.2   # of the machine's memory
# Whole tables aggregated fine up to Elo's 29M transactions (and Amex's 5.5M wide statements); Optiver's book ran a
# 15 GB machine out of memory from 34M rows on.
STREAM_MIN_ROWS = 30_000_000


def stream_rows(path: str) -> int:
    from tabularaml.generate.stream import _dataset
    return _dataset(path).count_rows()


def streamed_features(ch: Child, history: bool) -> list[pd.DataFrame]:
    """RelatedTables, price-path and latest-state history features of a streamed child, a key range at a time
    (each key's rows sit in one range, so this equals aggregating the table whole)."""
    rt, plan, hist = RelatedTables([], drop_const=False), {}, None
    F, R, H = [], [], []
    t = time.time()
    for i, df in enumerate(chunks(ch.source, ch.key, max(1, int(STREAM_BUDGET_SHARE * memory_bytes())), normalise)):
        c = Child(ch.name, df, key=ch.key, time=ch.time, drop=ch.drop)
        F.append(rt._aggregate(c, ch.key))
        R.append(return_features(df.drop(columns=[d for d in ch.drop if d in df.columns]), ch.key, ch.name,
                                 ch.time, plan=plan))
        hist = (history and repeats(c)) if hist is None else hist
        if hist:
            H.append(history_features(c))
        print(f"table {ch.name}: key range {i + 1} ({len(df)} rows) aggregated, {time.time() - t:.0f}s", flush=True)
        del df, c
    F = pd.concat(F)
    F = F[[c for c in F.columns if F[c].notna().mean() > 0.01 and F[c].nunique() > 1]]
    return [x for x in (F, pd.concat(R) if R and R[0].shape[1] else None, pd.concat(H) if H else None) if x is not None]


def date_col(df: pd.DataFrame) -> str | None:
    """The first text column holding calendar dates (2013-04-22, 2016-01-01 19:00:00)."""
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        v = as_text(df[c].dropna().head(1000))
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
        vals = pd.unique(as_text(main[c].dropna()))
        if len(vals) and np.isin(vals, pd.unique(as_text(s.dropna()))).mean() >= 0.5:
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


CM_CHECK_PARENTS, CM_CHECK_ROWS = 60_000, 300_000


def child_models_help(main, rel, cm, y, task, max_rows=60_000, seed=0):
    """Keep child-row models only when they add to the aggregates: a quick 3-fold LightGBM on
    training rows with and without them (Home Credit +0.28 AUC points; on Elo they cost
    hours and lowered the score, the card-level label being noise for each transaction)."""
    import lightgbm as lgb
    from sklearn.model_selection import KFold, StratifiedKFold
    t = time.time()
    base = pd.concat([main.reset_index(drop=True)] + rel, axis=1)
    full = pd.concat([base] + cm, axis=1)
    for X in (base, full):
        for c in X.columns:
            if not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
                X[c] = as_text(X[c]).astype("category")
    yv = y.to_numpy(dtype=float)
    rows = np.arange(len(yv))
    if len(rows) > max_rows:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_rows, replace=False))
    binary = (task or ("binary" if y.nunique() == 2 else "regression")) == "binary"
    P = dict(objective="binary" if binary else "regression", learning_rate=0.1, num_leaves=31,
             min_child_samples=50, feature_fraction=0.5, verbose=-1, num_threads=4, seed=seed)
    split = (StratifiedKFold(3, shuffle=True, random_state=seed).split(rows, yv[rows]) if binary
             else KFold(3, shuffle=True, random_state=seed).split(rows))
    gains = []
    for fi, vi in split:
        fi, vi = rows[fi], rows[vi]
        loss = []
        for X in (base, full):
            b = lgb.train(P, lgb.Dataset(X.iloc[fi], yv[fi]), 1000, valid_sets=[lgb.Dataset(X.iloc[vi], yv[vi])],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            loss.append(b.best_score["valid_0"]["binary_logloss" if binary else "l2"])
        gains.append((loss[0] - loss[1]) / loss[0])
    keep = float(np.mean(gains)) > 0 and sum(g > 0 for g in gains) >= 2
    print(f"child models: {100 * np.mean(gains):+.2f}% CV loss over the aggregates "
          f"({', '.join(f'{100 * g:+.2f}%' for g in gains)}) -> {'kept' if keep else 'dropped'} "
          f"in {time.time() - t:.0f}s", flush=True)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--id", default=None)
    ap.add_argument("--table", action="append", default=[], help="name=path[:key[:time]]")
    ap.add_argument("--time", default=None, help="main-table time column for as-of aggregation (default: auto)")
    ap.add_argument("--child-models", action=argparse.BooleanOptionalAction, default=True,
                    help="out-of-fold child-row model features for every keyed child table "
                         "(Home Credit 0.7976 -> 0.8004; --no-child-models turns them off)")
    ap.add_argument("--task", default=None, choices=["regression", "binary", "multiclass"])
    ap.add_argument("--log-target", action="store_true", help="search on log1p(target) (RMSLE-scored contests)")
    ap.add_argument("--budget", type=float, default=900, help="FeatureForge time budget (s)")
    ap.add_argument("--top-related", type=int, default=200,
                    help="related-table columns handed to FeatureForge's search (all are kept in the output)")
    ap.add_argument("--out-dir", default="features")
    ap.add_argument("--related-cache", default=None,
                    help="directory to save the related-table features to, or load them from when present "
                         "(arms that differ only in FeatureForge settings share them)")
    ap.add_argument("--forge-kw", default="{}", help="JSON of extra FeatureForge arguments")
    ap.add_argument("--forecast", default="auto", choices=["auto", "off"],
                    help="forecasting family (tabularaml/generate/forecast.py): on when the data has a date, "
                         "repeating entity keys and test rows after the training period")
    ap.add_argument("--forecast-kw", default="{}", help="JSON of extra ForecastFeatures arguments")
    ap.add_argument("--history", default="auto", choices=["auto", "off"],
                    help="history families (tabularaml/generate/history.py), on from structure: latest state of keyed "
                         "child tables with a time column (last, last - mean, last - previous), and outcome history "
                         "of as-of event logs that carry the target (earlier outcomes, strictly before each row)")
    ap.add_argument("--panel", default="auto", choices=["auto", "off"],
                    help="intraday panels: each entity against all at the same moment and its own last steps of the "
                         "day, tabularaml/generate/panel.py; on from structure")
    ap.add_argument("--lists", default="auto", choices=["auto", "off"],
                    help="list columns (item counts and common-item indicators) and per-unit amounts (a skewed amount "
                         "over small counts and their total), tabularaml/generate/lists.py; on from structure")
    a = ap.parse_args()

    t0 = time.time()
    tr, te = read(a.train), read(a.test)
    y = tr.pop(a.target)
    ids_tr = tr[a.id].to_numpy() if a.id else np.arange(len(tr))
    ids_te = te[a.id].to_numpy() if a.id else np.arange(len(tr), len(tr) + len(te))
    rel_tr = rel_te = None
    cache = Path(a.related_cache) if a.related_cache else None
    if cache is not None and (cache / "rel_train.parquet").exists():
        rel_tr, rel_te = pd.read_parquet(cache / "rel_train.parquet"), pd.read_parquet(cache / "rel_test.parquet")
        tr, te = pd.read_parquet(cache / "main_train.parquet"), pd.read_parquet(cache / "main_test.parquet")
        print(f"related tables: {rel_tr.shape[1]} columns from {cache}", flush=True)
    elif a.table:
        children = [parse_table(s, tr, a.id) for s in a.table]
        lookups = [ch for ch in children if "__lookup__" in ch.drop]
        children = [ch for ch in children if ch not in lookups]
        # One row per key: attributes of the main row (a tube's dimensions, its bill of
        # materials), joined as columns before anything is aggregated or looked up.
        for ch in [ch for ch in children if ch.source is None and ch.key in tr.columns
                   and not ch.df[ch.key].duplicated().any()]:
            cols = [c for c in ch.df.columns if c == ch.key or c not in tr.columns]
            tr = tr.merge(ch.df[cols], on=ch.key, how="left")
            te = te.merge(ch.df[cols], on=ch.key, how="left")
            children.remove(ch)
            print(f"table {ch.name}: one row per {ch.key}, joined as columns", flush=True)
        both_main = pd.concat([tr, te[[c for c in tr.columns if c in te.columns]]], ignore_index=True)
        asof = {ch.name: main_time(tr, ch, a.time) for ch in children
                if ch.source is None and ch.time is not None and ch.key in tr.columns and tr[ch.key].duplicated().any()}
        asof = {k: v for k, v in asof.items() if v is not None}
        keyed = [ch for ch in children if ch.name not in asof]
        F = pd.DataFrame()
        A = []
        for ch in children:
            if ch.name in asof:
                print(f"table {ch.name}: as of {asof[ch.name]} per main row", flush=True)
                A.append(asof_features(both_main, ch.key, asof[ch.name], ch))
                if a.history == "auto" and a.target in ch.df.columns:
                    # The log carries the outcome itself (earlier answers): outcome history strictly before
                    # each row, overall and on the row's own item (the shared id-like column with most levels).
                    shared = [c for c in both_main.columns if c in ch.df.columns and c not in (ch.key, asof[ch.name])
                              and both_main[c].nunique() > 50]
                    item = max(shared, key=lambda c: both_main[c].nunique()) if shared else None
                    t = time.time()
                    El = event_log_features(both_main, ch, a.target, asof[ch.name], item=item,
                                            outcome_values=sorted(pd.unique(y.dropna())))
                    print(f"table {ch.name}: outcome history (item={item}), {El.shape[1]} columns in {time.time() - t:.0f}s", flush=True)
                    A.append(El)
        cm_tables = []
        if a.child_models and keyed:
            # as-of children would see later rows' outcomes; labels per key are ambiguous
            # when the key repeats
            # (not for a table read by key range: its rows never sit in memory together)
            cm_tables = [ch for ch in keyed if tr[ch.key].is_unique and ch.source is None]
        for ch in keyed:
            if ch.source is not None:
                for j, Fs in enumerate(streamed_features(ch, a.history == "auto")):
                    Fs = Fs.reindex(both_main[ch.key].to_numpy()).set_index(both_main.index)
                    for c in [c for c in Fs.columns if c.endswith("__count")]:
                        Fs[c] = Fs[c].fillna(0)
                    A.insert(j, Fs)
                continue
            Fk = RelatedTables([ch]).features()
            Fk = Fk.reindex(both_main[ch.key].to_numpy()).set_index(both_main.index)
            for c in [c for c in Fk.columns if c.endswith("__count")]:
                Fk[c] = Fk[c].fillna(0)
            A.insert(0, Fk)
            # Price paths (an order book, a trade log): realized volatility per parent.
            t = time.time()
            Rk = return_features(ch.df.drop(columns=[c for c in ch.drop if c in ch.df.columns]), ch.key, ch.name, ch.time)
            if Rk.shape[1]:
                A.insert(1, Rk.reindex(both_main[ch.key].to_numpy()).set_index(both_main.index))
                print(f"table {ch.name}: price paths, {Rk.shape[1]} columns in {time.time() - t:.0f}s", flush=True)
            if a.history == "auto" and repeats(ch):
                t = time.time()
                Hk = history_features(ch).reindex(both_main[ch.key].to_numpy()).set_index(both_main.index)
                print(f"table {ch.name}: latest-state history, {Hk.shape[1]} columns in {time.time() - t:.0f}s", flush=True)
                A.insert(1, Hk)
        rel_tr, rel_te = pd.DataFrame(index=range(len(tr))), pd.DataFrame(index=range(len(te)))
        # Child rows that also share the main row's code values (a customer's purchases of the
        # offer's brand), counted before the main row's date when both tables carry dates.
        mt = date_col(both_main)
        for ch in [ch for ch in keyed if ch.source is None] + [ch for ch in children if ch.name in asof]:
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
        if cm_tables:
            task = a.task or ("binary" if y.nunique() == 2 else "regression")
            obj = "binary" if task == "binary" else "regression"
            # Decide on a sample of training parents with few child rows per fit first: on
            # Elo (29M transactions) the full models took over an hour and lowered the score.
            rng = np.random.default_rng(0)
            pos = np.arange(len(tr))
            if len(pos) > CM_CHECK_PARENTS:
                pos = np.sort(rng.choice(pos, CM_CHECK_PARENTS, replace=False))
            t = time.time()
            cm_s = []
            for ch in cm_tables:
                ks = tr[ch.key].to_numpy()[pos]
                sub = Child(ch.name, ch.df[ch.df[ch.key].isin(set(ks))], key=ch.key, time=ch.time,
                            children=ch.children, drop=ch.drop)
                Ms = child_model_features(sub, pd.Series(y.to_numpy()[pos], index=ks), [], task=obj,
                                          max_rows=CM_CHECK_ROWS)
                cm_s.append(Ms.reindex(ks).reset_index(drop=True))
            print(f"child models on {len(pos)} sampled parents: {time.time() - t:.0f}s", flush=True)
            keep = child_models_help(tr.drop(columns=[c for c in (a.id, a.target) if c and c in tr.columns]).iloc[pos],
                                     [Fa.iloc[pos].reset_index(drop=True) for Fa in A], cm_s,
                                     y.iloc[pos].reset_index(drop=True), a.task)
            for ch in cm_tables if keep else []:
                t = time.time()
                yk = pd.Series(y.to_numpy(), index=tr[ch.key].to_numpy())
                M = child_model_features(ch, yk, te[ch.key].to_numpy(), task=obj)
                A.append(M.reindex(both_main[ch.key].to_numpy()).set_index(both_main.index))
                print(f"child model {ch.name}: {time.time() - t:.0f}s", flush=True)
        for Fa in A:
            rel_tr = pd.concat([rel_tr, Fa.iloc[:len(tr)].reset_index(drop=True)], axis=1)
            rel_te = pd.concat([rel_te, Fa.iloc[len(tr):].reset_index(drop=True)], axis=1)
        print(f"related tables: {rel_tr.shape[1]} columns in {time.time() - t0:.0f}s", flush=True)
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
            rel_tr.to_parquet(cache / "rel_train.parquet")
            rel_te.to_parquet(cache / "rel_test.parquet")
            tr.to_parquet(cache / "main_train.parquet")  # with any one-row-per-key tables joined
            te.to_parquet(cache / "main_test.parquet")
        del A, both_main, children, keyed

    Xtr = tr.drop(columns=[a.id]) if a.id else tr
    Xte = te.drop(columns=[a.id]) if a.id else te
    Xte = Xte[Xtr.columns]
    n_main = tr.shape[1]
    if a.lists == "auto":
        t = time.time()
        Ltr, Lte, found = list_features(Xtr, Xte)
        for c in found:  # downstream families read them as text: items joined
            J = as_lists(pd.concat([Xtr[c], Xte[c]], ignore_index=True)).map(" ; ".join).to_numpy()
            Xtr[c], Xte[c] = J[:len(Xtr)], J[len(Xtr):]
        Utr, Ute = unit_ratio_features(Xtr, Xte)
        if Ltr.shape[1] or Utr.shape[1]:
            Xtr = pd.concat([Xtr.reset_index(drop=True), Ltr, Utr], axis=1)
            Xte = pd.concat([Xte.reset_index(drop=True), Lte, Ute], axis=1)
            print(f"lists: {found} -> {Ltr.shape[1]} columns; per-unit amounts: {Utr.shape[1]} columns "
                  f"in {time.time() - t:.0f}s", flush=True)
    if a.panel == "auto":
        t = time.time()
        Ptr, Pte, P = panel_features(Xtr, Xte)
        if P is not None:
            Xtr = pd.concat([Xtr.reset_index(drop=True), Ptr], axis=1)
            Xte = pd.concat([Xte.reset_index(drop=True), Pte], axis=1)
            print(f"panel: {P} -> {Ptr.shape[1]} columns in {time.time() - t:.0f}s", flush=True)
    del tr, te  # a second copy of both tables is gigabytes on IEEE-CIS
    forecasting = False
    if a.forecast == "auto":
        from tabularaml.generate.forecast import ForecastFeatures
        t = time.time()
        ff = ForecastFeatures(**json.loads(a.forecast_kw)).fit(Xtr, y.to_numpy(), Xte)
        if ff.active_:
            forecasting = True
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
                both[c] = as_text(both[c]).astype("category")
        b = lgb.train(dict(objective="binary" if y.nunique() == 2 else "regression", learning_rate=0.1,
                           num_leaves=31, feature_fraction=0.5, verbose=-1), lgb.Dataset(both, log_y(y) if a.log_target else y), 300)
        gain = pd.Series(b.feature_importance("gain"), index=both.columns)[rel_tr.columns]
        top = list(gain.sort_values(ascending=False).index[:a.top_related])
        Xtr = both[list(Xtr.columns) + top]
        Xte = pd.concat([Xte.reset_index(drop=True), rel_te[top]], axis=1)
        del both, b
    if forecasting:
        # On top of the forecasting columns the search's gate rows stop resembling the test
        # horizon: its picks lost held-out (Favorita 0.669 -> 0.689, Recruit 0.519 -> 0.525,
        # Rossmann kept none), so a forecasting contest gets the forecasting columns alone.
        print("forecasting family active: FeatureForge search skipped", flush=True)
        out_tr, out_te = Xtr.reset_index(drop=True).copy(), Xte.reset_index(drop=True).copy()
    else:
        forge = FeatureForge(task=a.task, time_budget=a.budget, log_target=a.log_target,
                             **json.loads(a.forge_kw)).fit(Xtr, y, X_unlabeled=Xte)
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
    print(f"done in {time.time() - t0:.0f}s: {out_tr.shape[1] - n_main - 1} columns added -> {out}/")


if __name__ == "__main__":
    main()
