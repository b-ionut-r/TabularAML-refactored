"""Subprocess entry point for a single (dataset_id, framework, seed) benchmark run.

Invoked by BenchmarkRunner as:
    python -m tabularaml.benchmarks.feature_gen._worker --spec '<json>'

Emits the result row as a single JSON line on stdout so the runner can parse
it back. All adapter/framework errors are caught and returned as a row with
status != "ok"; only truly exceptional failures (e.g. import error in the
worker itself) cause a non-zero exit code.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def _make_row(spec: dict, **overrides) -> dict:
    row = {
        "dataset_id": int(spec["dataset_id"]),
        "task": spec["task"],
        "framework": spec["framework"],
        "seed": int(spec["seed"]),
        "time_budget_s": int(spec.get("time_budget_s", 0)),
        "n_train": None,
        "n_test": None,
        "n_features_before": None,
        "n_features_after": None,
        "n_added": None,
        "score_holdout": None,
        "wall_time_fit": None,
        "wall_time_transform": None,
        "wall_time_total": None,
        "peak_rss_mb": None,
        "n_boost_rounds": None,
        "status": "crash",
        "error_msg": "",
        "adapter_version": "",
        "internal_log_json": "",
    }
    row.update(overrides)
    return row


def _load_dataset(dataset_id: int, task: str):
    from tabularaml.utils.datasets import Dataset
    ds = Dataset.fetch_by_id(int(dataset_id))
    if ds is None:
        raise RuntimeError(f"Dataset {dataset_id} could not be fetched")
    X = ds.X
    y = ds.y
    # Prefer the label recorded in the manifest if task mismatch is possible.
    return X, y, task or ds.task


def _preprocess(X: pd.DataFrame, y, task: str):
    """Framework-agnostic preprocessing done BEFORE any framework sees the data."""
    X = X.copy()
    # Drop all-constant cols + duplicate cols.
    const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if const_cols:
        X = X.drop(columns=const_cols)
    X = X.loc[:, ~X.columns.duplicated(keep="first")]
    # Densify any sparse columns (sparse pandas dtype crashes most FE frameworks).
    for c in X.columns:
        if hasattr(X[c], "sparse"):
            X[c] = X[c].sparse.to_dense()
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Cast object → category.
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype("category")

    y_ser = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    n_classes = 0
    if task == "classification":
        codes, _ = pd.factorize(y_ser, sort=True)
        y_out = codes.astype(int)
        n_classes = int(pd.Series(y_out).nunique())
    else:
        y_out = pd.to_numeric(y_ser, errors="coerce").astype(float).values
        # If any NaN in y after coercion, drop those rows.
        mask = ~np.isnan(y_out)
        if not mask.all():
            X = X.loc[mask].reset_index(drop=True)
            y_out = y_out[mask]
    return X, y_out, n_classes


def _peak_rss_mb() -> float:
    try:
        import psutil
        return float(psutil.Process(os.getpid()).memory_info().rss) / (1024 * 1024)
    except Exception:
        return float("nan")


def run(spec: dict) -> dict:
    import urllib.error
    from sklearn.model_selection import train_test_split
    from tabularaml.benchmarks.feature_gen.adapters import get_adapter_cls
    from tabularaml.benchmarks.feature_gen.adapters.base import _check_contract, ContractViolationError
    from tabularaml.benchmarks.feature_gen.evaluator import (
        score_on_holdout,
        select_scorer,
        split_early_stopping_validation,
    )
    from tabularaml.benchmarks.feature_gen.wandb_logger import (
        wandb_run, derive_tags, log_row,
    )

    row = _make_row(spec)
    t0_total = time.time()

    try:
        try:
            X_raw, y_raw, task = _load_dataset(spec["dataset_id"], spec["task"])
        except (urllib.error.URLError, RuntimeError) as e:
            if "could not be fetched" in str(e) or "urlopen error" in str(e):
                row["status"] = "dataset_fetch_failed"
                row["error_msg"] = str(e)
                return row
            raise
        X, y, n_classes = _preprocess(X_raw, y_raw, task)
        row["n_features_before"] = int(X.shape[1])
        if X.shape[1] == 0:
            row["status"] = "degenerate_dataset"
            row["error_msg"] = "Dataset has 0 features after preprocessing"
            return row

        stratify = y if (task == "classification" and len(np.unique(y)) > 1) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=int(spec["seed"]),
            stratify=stratify,
        )
        row["n_train"] = int(len(X_train))
        row["n_test"] = int(len(X_test))

        wandb_cfg = {
            "dataset_id": int(spec["dataset_id"]),
            "task": task,
            "framework": spec["framework"],
            "seed": int(spec["seed"]),
            "time_budget_s": int(spec["time_budget_s"]),
            "n_rows_total": int(len(X)),
            "n_cols_total": int(X.shape[1]),
            "n_classes": int(n_classes),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            **{k: spec.get(k) for k in ("mode", "framework_kwargs") if spec.get(k) is not None},
        }
        tags = derive_tags(spec["framework"], task, int(len(X)), int(X.shape[1]))
        wb_enabled = bool(spec.get("wandb_enabled", True))
        run_name = f"{spec['framework']}_{task}_tid{spec['dataset_id']}_seed{spec['seed']}"
        group = f"tid{spec['dataset_id']}"

        with wandb_run(
            project=spec.get("wandb_project", "tabularaml-fe-benchmark"),
            entity=spec.get("wandb_entity"),
            run_name=run_name,
            group=group,
            tags=tags,
            config=wandb_cfg,
            job_type="worker",
            enabled=wb_enabled,
        ) as wb:
            adapter_cls = get_adapter_cls(spec["framework"])

            adapter_kwargs = dict(spec.get("framework_kwargs") or {})
            if spec["framework"] == "tabularaml" and "mode" not in adapter_kwargs:
                adapter_kwargs["mode"] = spec.get("mode", "medium")

            adapter = adapter_cls(
                task=task,
                time_budget_s=int(spec["time_budget_s"]),
                random_state=int(spec["seed"]),
                n_jobs=int(spec.get("n_jobs", -1)),
                **adapter_kwargs,
            )
            row["adapter_version"] = getattr(adapter_cls, "version", "")

            if task == "regression" and not adapter.supports_regression:
                row["status"] = "unsupported_task"
                row["error_msg"] = "adapter does not support regression"
                return row
            if task == "classification" and not adapter.supports_classification:
                row["status"] = "unsupported_task"
                row["error_msg"] = "adapter does not support classification"
                return row
            if n_classes > 2 and not adapter.supports_multiclass:
                row["status"] = "unsupported_task"
                row["error_msg"] = "adapter does not support multiclass"
                return row

            t_fit_start = time.time()
            X_train_fit, X_train_es, y_train_fit, y_train_es = split_early_stopping_validation(
                X_train,
                y_train,
                task=task,
                seed=int(spec["seed"]),
            )

            X_train_fe = adapter.fit_transform(X_train_fit, pd.Series(y_train_fit))
            row["wall_time_fit"] = time.time() - t_fit_start

            t_tr_start = time.time()
            X_es_fe = adapter.transform(X_train_es)
            X_test_fe = adapter.transform(X_test)
            row["wall_time_transform"] = time.time() - t_tr_start

            _check_contract(X_train_fe, X_es_fe, len(X_train_fit), len(X_train_es))
            _check_contract(X_train_fe, X_test_fe, len(X_train_fit), len(X_test))

            row["n_features_after"] = int(X_train_fe.shape[1])
            row["n_added"] = int(adapter.get_feature_count_added())

            score, n_rounds = score_on_holdout(
                X_train_fe,
                y_train_fit,
                X_es_fe,
                y_train_es,
                X_test_fe,
                y_test,
                task=task, n_classes=n_classes, seed=int(spec["seed"]),
                n_jobs=int(spec.get("n_jobs", 1)),
            )
            scorer = select_scorer(task, n_classes)
            row["score_holdout"] = float(score)
            row["n_boost_rounds"] = int(n_rounds)
            row["status"] = "ok"
            row["peak_rss_mb"] = _peak_rss_mb()

            try:
                row["internal_log_json"] = json.dumps(adapter.get_internal_log(), default=str)
            except Exception:
                row["internal_log_json"] = ""

            row["scorer_name"] = scorer.name
            row["scorer_greater_is_better"] = bool(scorer.greater_is_better)

            log_row(wb, row)

    except ContractViolationError as e:
        row["status"] = "contract_violation"
        row["error_msg"] = str(e)[:500]
    except MemoryError as e:
        row["status"] = "oom"
        row["error_msg"] = f"MemoryError: {str(e)[:200]}"
    except Exception as e:
        cls_name = type(e).__name__
        if cls_name == "_AutofeatInternalNaNError":
            row["status"] = "autofeat_internal_nan"
        elif cls_name == "_AutofeatUpstreamBugError":
            row["status"] = "autofeat_upstream_bug"
        elif cls_name == "_FeaturetoolsUpstreamBugError":
            row["status"] = "featuretools_upstream_bug"
        elif cls_name == "_OpenFEUpstreamBugError":
            row["status"] = "openfe_upstream_bug"
        else:
            row["status"] = "crash"
        row["error_msg"] = f"{cls_name}: {str(e)[:400]}"
        traceback.print_exc(file=sys.stderr)

    row["wall_time_total"] = time.time() - t0_total
    return row


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, help="JSON-encoded RunSpec")
    p.add_argument("--out", required=False, default=None,
                   help="Optional path to write the single-row JSON result (fallback to stdout)")
    args = p.parse_args(argv)

    spec = json.loads(args.spec)
    try:
        row = run(spec)
    except BaseException as e:
        row = _make_row(spec, status="crash",
                        error_msg=f"{type(e).__name__}: {traceback.format_exc()[:1200]}")

    payload = json.dumps(row, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload, encoding="utf-8")
    sys.stdout.write(payload + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
