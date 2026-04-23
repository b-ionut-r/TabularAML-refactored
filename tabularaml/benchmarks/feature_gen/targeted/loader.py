"""Unified dataset loader for targeted benchmark suites.

Supports two sources:
  - "openml_task": fetch by OpenML task ID via the openml Python package.
  - "openml_dataset": fetch by OpenML dataset ID via the openml Python package.
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


class DatasetLoadError(RuntimeError):
    """Raised when a configured dataset cannot be fetched or decoded."""


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
    if spec.source == "openml_dataset":
        return _load_openml_dataset(spec)
    if spec.source == "openml_task":
        return _load_openml_task(spec)
    raise ValueError(f"Unknown dataset source: {spec.source!r}")


def _acquire_cache_lock(dataset_id: str):
    import tempfile
    try:
        from filelock import FileLock
        lock_path = Path(tempfile.gettempdir()) / f"dataset_cache_{dataset_id}.lock"
        return FileLock(str(lock_path))
    except ImportError:
        class _NoLock:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return _NoLock()

def _load_openml_task(spec: DatasetSpec) -> LoadedDataset:
    import openml

    task_id = int(spec.id)
    with _acquire_cache_lock(f"openml_{task_id}"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                task = openml.tasks.get_task(task_id)
                X, y = task.get_X_and_y(dataset_format="dataframe")
            except Exception as exc:
                if _looks_like_unknown_openml_task(exc):
                    return _load_openml_dataset(spec)
                raise DatasetLoadError(
                    f"OpenML task {task_id} could not be loaded: {exc}"
                ) from exc

    X = pd.DataFrame(X)
    y_ser = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    task_type, n_classes = _infer_task(y_ser, spec.task)
    return LoadedDataset(X=X, y=y_ser, task=task_type, name=spec.name, n_classes=n_classes)


def _looks_like_unknown_openml_task(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "unknown task" in text or ("returned code 151" in text and "/task/" in text)


def _default_openml_target(dataset) -> str | None:
    target = getattr(dataset, "default_target_attribute", None)
    if target is None:
        return None
    if isinstance(target, (list, tuple)):
        values = [str(v).strip() for v in target if str(v).strip()]
    else:
        values = [part.strip() for part in str(target).split(",") if part.strip()]
    if not values:
        return None
    return values[0]


def _load_openml_dataset(spec: DatasetSpec) -> LoadedDataset:
    import openml

    dataset_id = int(spec.id)
    with _acquire_cache_lock(f"openml_dataset_{dataset_id}"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                dataset = openml.datasets.get_dataset(
                    dataset_id,
                    download_data=True,
                    download_qualities=False,
                    download_features_meta_data=False,
                )
                target = _default_openml_target(dataset)
                if not target:
                    raise DatasetLoadError(
                        f"OpenML dataset {dataset_id} has no default target attribute"
                    )
                X, y, _, _ = dataset.get_data(
                    dataset_format="dataframe",
                    target=target,
                )
            except DatasetLoadError:
                raise
            except Exception as exc:
                raise DatasetLoadError(
                    f"OpenML dataset {dataset_id} could not be loaded: {exc}"
                ) from exc

    if y is None:
        raise DatasetLoadError(f"OpenML dataset {dataset_id} returned no target column")

    X = pd.DataFrame(X)
    y_ser = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    task_type, n_classes = _infer_task(y_ser, spec.task)
    return LoadedDataset(X=X, y=y_ser, task=task_type, name=spec.name, n_classes=n_classes)


def _load_pmlb(spec: DatasetSpec) -> LoadedDataset:
    import pmlb

    with _acquire_cache_lock(f"pmlb_{spec.id}"):
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
