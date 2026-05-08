#!/usr/bin/env python3
"""
Run TabularAML feature generation without the Hugging Face/Flask UI.

Defaults mirror the settings shown in the UI screenshot:
Regression target, Pearson metric, Extreme mode, TimeSeriesSplit on ``moon``,
80 generations, 100 parents, 360 children, 24h budget, GPU/adaptive enabled.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.utils.multiclass import type_of_target

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tabularaml.eval.scorers import PREDEFINED_CLS_SCORERS, PREDEFINED_REG_SCORERS
from tabularaml.eval.splitters import PurgedTimeSeriesSplit
from tabularaml.generate.features import FeatureGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless TabularAML feature generation using tabularaml.generate.features.FeatureGenerator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataset", required=True, help="Input dataset path (.parquet, .csv, .json, .xlsx/.xls).")
    parser.add_argument("--target", default="target", help="Target column name.")
    parser.add_argument("--task", choices=["auto", "regression", "classification"], default="regression")
    parser.add_argument("--metric", default="pearson", help="Metric key, e.g. pearson, rmse, r2, accuracy.")

    parser.add_argument("--mode", default="extreme", help="FeatureGenerator mode.")
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--parents", type=int, default=100)
    parser.add_argument("--children", type=int, default=360)
    parser.add_argument("--min-pct-gain", type=float, default=0.002)
    parser.add_argument("--early-stop-iter", type=int, default=12)
    parser.add_argument("--early-stop-child-eval", type=int, default=80)
    parser.add_argument("--max-new-feats", type=float, default=1.0)
    parser.add_argument("--ranking-method", choices=["multi_criteria", "shap", "none"], default="multi_criteria")

    parser.add_argument("--cv-strategy", choices=["kfold", "groupfold", "timeseries"], default="timeseries")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--time-column", default="moon", help="Column used for TimeSeriesSplit ordering/grouping.")
    parser.add_argument("--group-column", default=None, help="Column used for GroupKFold. Defaults to --time-column.")
    parser.add_argument("--embargo-gap", type=int, default=4, help="TimeSeriesSplit gap.")
    parser.add_argument(
        "--keep-split-column",
        action="store_true",
        help="Keep the time/group split column in X. By default it is dropped to match the UI leakage guard.",
    )

    parser.add_argument("--time-budget-minutes", type=int, default=1440)
    parser.add_argument("--search-sample-size", type=int, default=234011)
    parser.add_argument("--use-gpu", dest="use_gpu", action="store_true", default=True)
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    parser.add_argument("--adaptive", dest="adaptive", action="store_true", default=True)
    parser.add_argument("--no-adaptive", dest="adaptive", action="store_false")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)

    parser.add_argument("--save-generator", default="cache/feature_generator.pkl")
    parser.add_argument(
        "--log-file",
        default="cache/feature_generation_headless.log",
        help="Path for FeatureGenerator and script logs.",
    )
    parser.add_argument("--append-log", action="store_true", help="Append to --log-file instead of starting a fresh log.")
    parser.add_argument(
        "--output-dataset",
        default=None,
        help="Optional path to save the generated feature matrix plus target. Format follows extension.",
    )
    parser.add_argument("--summary-json", default=None, help="Optional path to write run summary JSON.")
    parser.add_argument("--meta-validation-frac", type=float, default=0.15,
                        help="Fraction of data held out for meta-validation overfitting check. Set to 0 to disable.")

    return parser.parse_args()


def load_dataframe(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(dataset_path)
    if suffix == ".csv":
        return pd.read_csv(dataset_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(dataset_path)
    if suffix == ".json":
        return pd.read_json(dataset_path)
    raise ValueError(f"Unsupported dataset extension: {suffix}")


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(output_path, index=False)
    elif suffix == ".json":
        df.to_json(output_path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported output extension: {suffix}")


def resolve_scorer(task: str, metric: str, y: pd.Series):
    inferred_task = "regression" if type_of_target(y) == "continuous" else "classification"
    resolved_task = inferred_task if task == "auto" else task

    if metric == "auto":
        return resolved_task, None

    scorer = (
        PREDEFINED_REG_SCORERS.get(metric)
        if resolved_task == "regression"
        else PREDEFINED_CLS_SCORERS.get(metric)
    )

    if scorer is None and resolved_task == "classification":
        n_classes = int(y.nunique(dropna=True))
        if metric == "binary_crossentropy" and n_classes > 2:
            scorer = PREDEFINED_CLS_SCORERS.get("categorical_crossentropy")
        elif metric == "categorical_crossentropy" and n_classes == 2:
            scorer = PREDEFINED_CLS_SCORERS.get("binary_crossentropy")
        elif metric == "binary_roc_auc" and n_classes > 2:
            scorer = PREDEFINED_CLS_SCORERS.get("categorical_roc_auc")
        elif metric == "categorical_roc_auc" and n_classes == 2:
            scorer = PREDEFINED_CLS_SCORERS.get("binary_roc_auc")

    if scorer is None:
        raise ValueError(f'Invalid metric "{metric}" for task "{resolved_task}".')

    return resolved_task, scorer


def build_cv(args: argparse.Namespace, df: pd.DataFrame):
    groups = None
    split_column = None
    cv_obj: int | GroupKFold | PurgedTimeSeriesSplit = args.cv_folds

    if args.cv_strategy == "groupfold":
        split_column = args.group_column or args.time_column
        if not split_column:
            raise ValueError("--group-column is required for GroupKFold.")
        if split_column not in df.columns:
            raise ValueError(f'Group column "{split_column}" not found.')
        groups = df[split_column].to_numpy()
        cv_obj = GroupKFold(n_splits=args.cv_folds)

    elif args.cv_strategy == "timeseries":
        split_column = args.time_column
        if not split_column:
            raise ValueError("--time-column is required for TimeSeriesSplit.")
        if split_column not in df.columns:
            raise ValueError(f'Time column "{split_column}" not found.')
        groups = df[split_column].to_numpy()
        unique_periods = np.sort(np.unique(groups))
        base = TimeSeriesSplit(n_splits=args.cv_folds, gap=args.embargo_gap)
        cv_obj = PurgedTimeSeriesSplit(base, unique_periods, groups)

    return cv_obj, groups, split_column


def as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [as_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def prepare_log_file(path: str | Path | None, append: bool) -> Path | None:
    if not path:
        return None

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not append:
        log_path.write_text("", encoding="utf-8")
    return log_path


def log_message(message: str, log_path: Path | None = None) -> None:
    print(message)
    if log_path:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")


def main() -> int:
    args = parse_args()
    started_at = time.time()
    log_path = prepare_log_file(args.log_file, args.append_log)
    log_message(f"Headless feature generation log: {time.strftime('%Y-%m-%d %H:%M:%S')}", log_path)

    df = load_dataframe(args.dataset)
    if args.target not in df.columns:
        raise ValueError(f'Target column "{args.target}" not found. Available columns: {list(df.columns)}')

    y = df[args.target]
    X = df.drop(columns=[args.target]).copy()
    task, scorer = resolve_scorer(args.task, args.metric, y)
    cv_obj, groups, split_column = build_cv(args, df)

    if split_column and split_column in X.columns and not args.keep_split_column:
        log_message(f"Dropping split column '{split_column}' from features to match UI leakage guard.", log_path)
        X = X.drop(columns=[split_column])

    save_generator_path = Path(args.save_generator)
    save_generator_path.parent.mkdir(parents=True, exist_ok=True)

    generator_params = {
        "mode": args.mode,
        "task": task,
        "scorer": scorer,
        "n_generations": args.generations,
        "n_parents": args.parents,
        "n_children": args.children,
        "min_pct_gain": args.min_pct_gain,
        "early_stopping_iter": args.early_stop_iter,
        "early_stopping_child_eval": args.early_stop_child_eval,
        "max_new_feats": args.max_new_feats,
        "ranking_method": args.ranking_method,
        "cv": cv_obj,
        "groups": groups,
        "time_budget": args.time_budget_minutes * 60,
        "search_sample_size": args.search_sample_size,
        "use_gpu": args.use_gpu,
        "adaptive": args.adaptive,
        "save_path": str(save_generator_path),
        "log_file": str(log_path) if log_path else None,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs,
        "meta_validation_frac": args.meta_validation_frac,
    }

    log_message("Starting headless FeatureGenerator search with parameters:", log_path)
    printable_params = {
        key: value
        for key, value in generator_params.items()
        if key not in {"scorer", "cv", "groups"}
    }
    printable_params["metric"] = args.metric
    printable_params["cv_strategy"] = args.cv_strategy
    printable_params["cv"] = type(cv_obj).__name__ if not isinstance(cv_obj, int) else cv_obj
    printable_params["groups"] = "provided" if groups is not None else None
    log_message(json.dumps(as_jsonable(printable_params), indent=2), log_path)

    generator = FeatureGenerator(**generator_params)
    X_generated, _, generation_features, interactions = generator.search(X, y)
    generator.save(str(save_generator_path))

    output_dataset_path = None
    if args.output_dataset:
        output_dataset = X_generated.copy()
        output_dataset[args.target] = y.loc[output_dataset.index].to_numpy()
        save_dataframe(output_dataset, args.output_dataset)
        output_dataset_path = str(Path(args.output_dataset))

    summary = {
        "dataset": str(Path(args.dataset)),
        "target": args.target,
        "task": task,
        "metric": args.metric,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "initial_features": int(getattr(generator, "n_init_feats", X.shape[1])),
        "added_features": int(getattr(generator, "n_added_feats", 0) or 0),
        "final_features": int(getattr(generator, "n_final_feats", X_generated.shape[1])),
        "initial_validation_metric": getattr(generator, "initial_val_metric", None),
        "final_validation_metric": getattr(generator, "final_metric", None),
        "gain": getattr(generator, "gain", None),
        "percent_gain": (getattr(generator, "pct_gain", 0.0) or 0.0) * 100,
        "best_generation": generator.state["best"].get("gen_num") if hasattr(generator, "state") else None,
        "generation_feature_count": len(generation_features),
        "interaction_count": len(interactions),
        "generator_path": str(save_generator_path),
        "output_dataset": output_dataset_path,
    }

    log_message("Completed headless feature generation:", log_path)
    log_message(json.dumps(as_jsonable(summary), indent=2), log_path)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(as_jsonable(summary), indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())