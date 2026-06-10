"""Build the benchmark manifest by probing the existing OpenML id pools.

Walks `tabularaml/utils/cls_ids.csv` (~3500) + `reg_ids.csv` (~1250), fetches
dataset metadata via openml.datasets.get_dataset(..., download_data=False), and
keeps only datasets that can reasonably be benchmarked across every adapter
(including the fussy ones: AutoFeat for NaN-free numeric requirements,
Featuretools for size-based blow-up).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
from tqdm.auto import tqdm
try:
    # Optional: only needed when (re)building the manifest from OpenML.
    import openml
except ImportError:
    openml = None
import warnings


_BASE = Path(__file__).resolve().parent
_UTILS = Path(__file__).resolve().parents[2] / "utils"
_DEFAULT_MANIFEST = _BASE / "manifest.csv"


def _load_id_pools() -> pd.DataFrame:
    cls = pd.read_csv(_UTILS / "cls_ids.csv")["tid"].astype(int).tolist()
    reg = pd.read_csv(_UTILS / "reg_ids.csv")["tid"].astype(int).tolist()
    return pd.DataFrame(
        [{"tid": t, "task": "classification"} for t in cls]
        + [{"tid": t, "task": "regression"} for t in reg]
    )


def _probe_one(tid: int, task: str):
    """Return a dict describing the dataset or None if it should be skipped."""
    if openml is None:
        return None, "openml_not_installed"
    try:
        task_obj = openml.tasks.get_task(tid, download_data=False, download_qualities=True)
    except Exception as e:
        return None, f"task_fetch_failed: {type(e).__name__}"
    try:
        ds = openml.datasets.get_dataset(task_obj.dataset_id, download_data=False)
    except Exception as e:
        return None, f"dataset_fetch_failed: {type(e).__name__}"
    q = ds.qualities or {}

    def _safe_int(k, default=0):
        v = q.get(k, default)
        if pd.isna(v):
            return default
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return default

    def _safe_float(k, default=0.0):
        v = q.get(k, default)
        if pd.isna(v):
            return default
        try:
            return float(v)
        except (ValueError, TypeError):
            return default

    row = {
        "tid": int(tid),
        "did": int(task_obj.dataset_id),
        "task": task,
        "n_rows": _safe_int("NumberOfInstances"),
        "n_cols": _safe_int("NumberOfFeatures"),
        "n_numeric": _safe_int("NumberOfNumericFeatures"),
        "n_categorical": _safe_int("NumberOfSymbolicFeatures"),
        "n_classes": _safe_int("NumberOfClasses"),
        "pct_missing": _safe_float("PercentageOfMissingValues") / 100.0,
        "name": str(ds.name)[:64],
    }
    return row, "ok"


def build_manifest(
    min_rows: int = 500,
    max_rows: int = 50_000,
    max_features: int = 200,
    max_missing_pct: float = 0.5,
    min_numeric_features: int = 1,
    max_classes_multiclass: int = 20,
    out_path: Optional[Path] = None,
    incremental: bool = True,
) -> pd.DataFrame:
    """Probe all ids, apply filters, and write manifest.csv.

    `incremental=True` loads an existing out_path first and only probes new tids,
    so re-runs after a crash resume cheaply.
    """
    out_path = Path(out_path) if out_path else _DEFAULT_MANIFEST
    pools = _load_id_pools()

    existing = pd.DataFrame()
    if incremental and out_path.exists():
        existing = pd.read_csv(out_path)
        done_tids = set(existing["tid"].astype(int).tolist())
        pools = pools[~pools["tid"].isin(done_tids)].reset_index(drop=True)

    rows = []
    errors = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _, r in tqdm(pools.iterrows(), total=len(pools), desc="probing OpenML"):
            row, status = _probe_one(int(r["tid"]), r["task"])
            if row is not None:
                rows.append(row)
            else:
                errors.append({"tid": int(r["tid"]), "task": r["task"], "status": status})

    probed = pd.DataFrame(rows)
    all_probed = pd.concat([existing, probed], ignore_index=True) if len(existing) else probed
    all_probed = all_probed.drop_duplicates(subset=["tid"], keep="last").reset_index(drop=True)

    # Filter.
    f = all_probed.copy()
    f = f[(f["n_rows"] >= min_rows) & (f["n_rows"] <= max_rows)]
    f = f[f["n_cols"] <= max_features]
    f = f[f["n_numeric"] >= min_numeric_features]
    f = f[f["pct_missing"] <= max_missing_pct]
    cls_mask = (f["task"] == "classification") & (f["n_classes"].between(2, max_classes_multiclass))
    reg_mask = f["task"] == "regression"
    f = f[cls_mask | reg_mask].reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(out_path, index=False)

    if errors:
        err_path = out_path.with_name(out_path.stem + "_errors.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False)

    return f


def load_manifest(path: Optional[Path] = None) -> pd.DataFrame:
    path = Path(path) if path else _DEFAULT_MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {path}. Run scripts/build_manifest.py first."
        )
    return pd.read_csv(path)


def subset_manifest(manifest: pd.DataFrame, subset: str) -> pd.DataFrame:
    subset = subset.lower()
    if subset == "full":
        return manifest.reset_index(drop=True)
    counts = {"smoke": 3, "small": 50, "medium": 1000}
    if subset not in counts:
        raise ValueError(f"Unknown subset {subset!r}; use smoke|small|medium|full")
    n = counts[subset]
    # Stratified by task so classification and regression are both represented.
    parts = []
    for task in ["regression", "classification"]:
        sub = manifest[manifest["task"] == task]
        k = min(len(sub), n // 2 + 1)
        # Select randomly instead of picking the smallest, using a fixed seed for consistency
        sub = sub.sample(n=k, random_state=42)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).head(n).reset_index(drop=True)
