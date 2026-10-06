"""Measure feature-engineering lift on an untouched outer holdout.

For each dataset and repeat:
  1. Outer 80/20 split (stratified for classification). The 20% is never
     seen by feature search, selection, encoders or early stopping.
  2. Each FE arm is fitted on the 80% only; target encodings on training
     rows are out-of-fold.
  3. The same downstream model (``ContestSolver``: 5-fold bagged LightGBM by
     default, test predictions averaged over folds) is trained on raw and on
     engineered features and scored on the holdout.

Lift is reported as percent reduction of the holdout loss (positive = FE
helps) together with FE wall time.

    python scripts/bench_fe.py --arms raw forge --out reports/fe_bench.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabularaml.benchmarks.contest_suite import CONTEST, STRUCTURED, SUITE, load_suite_dataset  # noqa: E402
from tabularaml.contest import ContestSolver, get_metric  # noqa: E402


def fe_raw(X_tr, y_tr, X_te, task, metric, seed, budget, threads):
    return X_tr, X_te, {}


FORGE_KW: dict = {}  # extra FeatureForge arguments from --forge-kw
TRANSDUCTIVE = False  # --transductive: hand the holdout's features (never its labels) to FeatureForge


def fe_forge(X_tr, y_tr, X_te, task, metric, seed, budget, threads):
    from tabularaml.generate.forge import FeatureForge
    forge = FeatureForge(task=task, log_target=(metric == "rmsle"), time_budget=budget,
                         random_state=seed, n_jobs=threads, verbose=True, **FORGE_KW).fit(
        X_tr, y_tr, X_unlabeled=X_te if TRANSDUCTIVE else None)
    info = dict(n_added=len(forge.new_columns_), gate=forge.gate_passed_,
                base_cv=forge.base_cv_loss_, search_cv=forge.search_cv_loss_,
                features=forge.new_columns_[:80])
    return forge.transform_train(X_tr), forge.transform(X_te), info


def fe_tabularaml(X_tr, y_tr, X_te, task, metric, seed, budget, threads):
    """The existing genetic FeatureGenerator (lite preset, budget-capped)."""
    from tabularaml.generate.features import FeatureGenerator
    from tabularaml.eval.scorers import rmse, binary_crossentropy, categorical_crossentropy
    scorer = rmse if task == "regression" else (binary_crossentropy if task == "binary" else categorical_crossentropy)
    gen = FeatureGenerator(task="regression" if task == "regression" else "classification",
                           scorer=scorer, mode="lite", time_budget=budget, use_gpu=False,
                           log_file=None, random_state=seed, n_jobs=threads)
    # The genetic engine predates pandas' string dtype; hand it object columns.
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for c in X_tr.columns:
        if pd.api.types.is_string_dtype(X_tr[c]) and X_tr[c].dtype != object:
            X_tr[c], X_te[c] = X_tr[c].astype(object), X_te[c].astype(object)
    gen.generate(X_tr, y_tr)
    gen.fit(X_tr, y_tr)
    A, B = gen.transform(X_tr), gen.transform(X_te)
    return A, B, dict(n_added=A.shape[1] - X_tr.shape[1])


def _load_openfe_adapter():
    """Import the adapter without the benchmark package's __init__ (which needs ``openml``)."""
    import importlib.util
    import types
    root = Path(__file__).resolve().parents[1] / "tabularaml" / "benchmarks" / "feature_gen" / "adapters"
    for pkg, path in (("tabularaml.benchmarks.feature_gen", root.parent), ("tabularaml.benchmarks.feature_gen.adapters", root)):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(path)]
            sys.modules[pkg] = m
    for name in ("base", "openfe_adapter"):
        full = f"tabularaml.benchmarks.feature_gen.adapters.{name}"
        if full not in sys.modules:
            spec = importlib.util.spec_from_file_location(full, root / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[full] = mod
            spec.loader.exec_module(mod)
    return sys.modules["tabularaml.benchmarks.feature_gen.adapters.openfe_adapter"].OpenFEAdapter


def fe_openfe(X_tr, y_tr, X_te, task, metric, seed, budget, threads):
    """OpenFE (ICML 2023) through the repo's adapter. Its transform computes
    aggregates over train + test together (upstream behaviour)."""
    OpenFEAdapter = _load_openfe_adapter()
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for c in X_tr.columns:
        if X_tr[c].dtype == object or pd.api.types.is_string_dtype(X_tr[c]):
            cats = pd.Index(pd.unique(pd.concat([X_tr[c], X_te[c]]).astype(str)))
            X_tr[c] = pd.Categorical(X_tr[c].astype(str), categories=cats)
            X_te[c] = pd.Categorical(X_te[c].astype(str), categories=cats)
    yt = np.log1p(y_tr) if metric == "rmsle" else y_tr
    ad = OpenFEAdapter("regression" if task == "regression" else "classification", int(budget), seed, n_jobs=threads)
    A = ad.fit_transform(X_tr, yt)
    B = ad.transform(X_te)
    return A, B[A.columns], dict(n_added=A.shape[1] - X_tr.shape[1])


ARMS = {"raw": fe_raw, "forge": fe_forge, "tabularaml": fe_tabularaml, "openfe": fe_openfe}


def evaluate_repo(X_tr, y_tr, X_te, y_te, task, metric, seed, threads):
    """The repo's original evaluator: one early-stopped XGBoost with fixed params."""
    from tabularaml.benchmarks.feature_gen.evaluator import (
        build_base_learner, split_early_stopping_validation)
    from tabularaml.contest.solver import prepare_frames
    X_tr, X_te = prepare_frames(X_tr, X_te)
    t = "regression" if task == "regression" else "classification"
    yt = np.log1p(y_tr) if metric == "rmsle" else y_tr
    X_fit, X_val, y_fit, y_val = split_early_stopping_validation(X_tr, yt, t, seed)
    model = build_base_learner(t, int(pd.Series(y_tr).nunique()), seed, n_jobs=threads)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    if task == "regression":
        p = model.predict(X_te)
        p = np.expm1(p) if metric == "rmsle" else p
    else:
        p = model.predict_proba(X_te)
        p = p[:, 1] if task == "binary" else p
    return get_metric(metric)(y_te, p), np.nan


AG_KW: dict = {"presets": "medium_quality", "time_limit": 120}  # --ag-preset / --ag-time


def evaluate_autogluon(X_tr, y_tr, X_te, y_te, task, metric, seed, threads):
    """AutoGluon, the AutoML a contest pipeline would run after feature engineering."""
    import shutil
    import tempfile
    from autogluon.tabular import TabularPredictor
    ag_metric = {"logloss": "log_loss", "rmse": "root_mean_squared_error",
                 "rmsle": "root_mean_squared_error"}.get(metric, metric)
    ag_task = {"binary": "binary", "multiclass": "multiclass", "regression": "regression"}[task]
    yt = np.log1p(y_tr) if metric == "rmsle" else y_tr
    tr = X_tr.copy()
    tr["__y__"] = np.asarray(yt)
    path = tempfile.mkdtemp(prefix="ag_")
    try:
        pred = TabularPredictor("__y__", problem_type=ag_task, eval_metric=ag_metric, path=path,
                                verbosity=0).fit(tr, presets=AG_KW["presets"], time_limit=AG_KW["time_limit"],
                                                 num_cpus=threads, ag_args_fit={"random_seed": seed})
        if task == "regression":
            p = pred.predict(X_te).to_numpy()
            p = np.expm1(p) if metric == "rmsle" else p
        else:
            P = pred.predict_proba(X_te)
            P = P[sorted(P.columns)].to_numpy()
            p = P[:, 1] if task == "binary" else P
        return get_metric(metric)(y_te, p), np.nan
    finally:
        shutil.rmtree(path, ignore_errors=True)


def evaluate(X_tr, y_tr, X_te, y_te, task, metric, seed, models, threads, judge="solver"):
    if judge == "autogluon":
        return evaluate_autogluon(X_tr, y_tr, X_te, y_te, task, metric, seed, threads)
    if judge == "repo":
        return evaluate_repo(X_tr, y_tr, X_te, y_te, task, metric, seed, threads)
    solver = ContestSolver(task=task, metric=metric, models=models, n_folds=5, seeds=(seed,),
                           n_jobs=threads, verbose=False).fit(X_tr, y_tr, X_te)
    m = get_metric(metric)
    return m(y_te, solver.test_ensemble_), solver.oof_ensemble_score_


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--suite", default="public", choices=["public", "contest", "structured"],
                    help="public: PMLB/AutoGluon tables; contest: real competition tables from OpenML")
    ap.add_argument("--arms", nargs="*", default=["raw", "forge"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--models", nargs="*", default=["lgbm"])
    ap.add_argument("--budget", type=float, default=300)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--judge", default="solver", choices=["solver", "repo", "autogluon"],
                    help="downstream model: ContestSolver(--models), the repo's fixed XGBoost, or AutoGluon")
    ap.add_argument("--transductive", action="store_true",
                    help="contest setting: FeatureForge sees the test rows' features (counts, group stats)")
    ap.add_argument("--ag-preset", default="medium_quality")
    ap.add_argument("--ag-time", type=float, default=120, help="AutoGluon time limit per fit (s)")
    ap.add_argument("--tag", default="", help="suffix for non-raw arm names (algorithm versions)")
    ap.add_argument("--out", type=Path, default=Path("reports/fe_bench.csv"))
    ap.add_argument("--forge-kw", default="{}", help='JSON of extra FeatureForge arguments, e.g. \'{"top_keys": 16}\'')
    args = ap.parse_args()
    FORGE_KW.update(json.loads(args.forge_kw))
    AG_KW.update(presets=args.ag_preset, time_limit=args.ag_time)
    global TRANSDUCTIVE
    TRANSDUCTIVE = args.transductive
    if args.datasets is None:
        args.datasets = list({"contest": CONTEST, "structured": STRUCTURED}.get(args.suite, SUITE))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        prev = pd.read_csv(args.out)
        done = set(zip(prev.dataset, prev.seed, prev.arm))

    for name in args.datasets:
        X, y, task, metric = load_suite_dataset(name)
        for seed in args.seeds:
            strat = y if task != "regression" else None
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=strat)
            X_tr, X_te = X_tr.reset_index(drop=True), X_te.reset_index(drop=True)
            y_tr, y_te = y_tr.reset_index(drop=True), y_te.reset_index(drop=True).to_numpy()
            for base_arm in args.arms:
                arm = f"{base_arm}_{args.tag}" if args.tag else base_arm
                if args.judge != "solver":
                    arm = f"{arm}@{'ag' if args.judge == 'autogluon' else args.judge}"
                if (name, seed, arm) in done:
                    continue
                t0 = time.time()
                try:
                    A, B, info = ARMS[base_arm](X_tr, y_tr, X_te, task, metric, seed, args.budget, args.threads)
                    fe_secs = time.time() - t0
                    test, oof = evaluate(A, y_tr, B, y_te, task, metric, seed, args.models, args.threads, args.judge)
                    status, err = "ok", ""
                except Exception as exc:  # keep the suite running
                    import traceback
                    traceback.print_exc()
                    fe_secs, test, oof, info, status, err = time.time() - t0, np.nan, np.nan, {}, "error", repr(exc)[:300]
                row = dict(dataset=name, seed=seed, arm=arm, task=task, metric=metric,
                           n_train=len(X_tr), n_raw=X.shape[1], test_score=test, oof_score=oof,
                           fe_seconds=fe_secs, n_added=info.get("n_added", 0),
                           gate=info.get("gate"), status=status, error=err,
                           info=json.dumps({k: v for k, v in info.items() if k != "n_added"}, default=str))
                pd.DataFrame([row]).to_csv(args.out, mode="a", header=not args.out.exists(), index=False)
                print(f"### {name:<12} seed={seed} {arm:<10} {metric}={test:.6f} "
                      f"fe={fe_secs:.0f}s added={row['n_added']} gate={row['gate']}", flush=True)


if __name__ == "__main__":
    main()
