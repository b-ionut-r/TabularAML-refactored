"""Unified dataset loader for targeted benchmark suites.

Supports two sources:
  - "openml_task": fetch by OpenML task ID via the openml Python package.
  - "pmlb":        fetch by PMLB dataset name via the pmlb Python package.

Both paths return a LoadedDataset with a consistent interface.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .registry import DatasetSpec


def _pmlb_cache_dir() -> str:
    """Return a stable local cache directory for PMLB downloads."""
    d = Path(os.environ.get("PMLB_CACHE_DIR", Path.home() / ".cache" / "pmlb"))
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


@dataclass
class LoadedDataset:
    X: pd.DataFrame
    y: pd.Series
    task: str       # "classification" | "regression" | "multiclass"
    name: str
    n_classes: int  # 0 for regression, >=2 for classification/multiclass


def load_dataset(spec: DatasetSpec) -> LoadedDataset:
    if spec.source == "pmlb":
        return _load_pmlb(spec)
    if spec.source == "openml_task":
        return _load_openml_task(spec)
    raise ValueError(f"Unknown dataset source: {spec.source!r}")


def _load_openml_task(spec: DatasetSpec) -> LoadedDataset:
    import openml

    task_id = int(spec.id)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        task = openml.tasks.get_task(task_id)
        X, y = task.get_X_and_y(dataset_format="dataframe")

    X = pd.DataFrame(X)
    y_ser = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    task_type, n_classes = _infer_task(y_ser, spec.task)
    return LoadedDataset(X=X, y=y_ser, task=task_type, name=spec.name, n_classes=n_classes)


def _load_pmlb(spec: DatasetSpec) -> LoadedDataset:
    import pmlb

    df = pmlb.fetch_data(spec.id, local_cache_dir=_pmlb_cache_dir())
    if "target" not in df.columns:
        raise RuntimeError(f"PMLB dataset {spec.id!r} has no 'target' column")

    y_ser = df["target"].reset_index(drop=True)
    X = df.drop(columns=["target"]).reset_index(drop=True)

    task_type, n_classes = _infer_task(y_ser, spec.task)
    return LoadedDataset(X=X, y=y_ser, task=task_type, name=spec.name, n_classes=n_classes)


def _infer_task(y: pd.Series, hint: str) -> tuple[str, int]:
    """Derive final task label and n_classes from the target series.

    The registry hint is used as a tiebreaker when the target dtype is
    ambiguous (e.g. integer-coded labels could be classification or regression).
    """
    n_unique = int(y.nunique())

    if hint == "regression":
        return "regression", 0

    if hint in ("classification", "multiclass") or n_unique <= 20:
        codes = np.unique(y.dropna().values)
        n_cls = len(codes)
        task_type = "multiclass" if n_cls > 2 else "classification"
        return task_type, n_cls

    # Fallback: treat as regression when many unique numeric values
    if pd.api.types.is_numeric_dtype(y):
        return "regression", 0

    n_cls = n_unique
    task_type = "multiclass" if n_cls > 2 else "classification"
    return task_type, n_cls
