"""W&B integration for the cross-framework FE benchmark.

Designed to be safely no-op when wandb isn't installed or login is absent -
the runner stays functional either way, all real analysis is driven off
master.csv.

Sections
--------
1. Guard helpers      - _wandb_available / _wandb_enabled / _bucket
2. Per-worker helpers - wandb_run, log_row, log_artifact, derive_tags
                        (called from _worker.py subprocesses)
3. Artifact sync      - download_results_artifact
                        (called by runner at startup to rehydrate master.csv)
4. Orchestrator viz   - dataframe/table/plot/figure helpers + OrchestratorRun
                        (long-lived run that owns artifact uploads and charts)
"""
from __future__ import annotations

import os
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Guard helpers
# ---------------------------------------------------------------------------

def _wandb_available() -> bool:
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False

def _wandb_enabled() -> bool:
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    return _wandb_available()

def _bucket(n: int, edges=(1_000, 10_000, 50_000)) -> str:
    labels = ["<1k", "<10k", "<50k", ">=50k"]
    for edge, label in zip(edges, labels):
        if n < edge:
            return label
    return labels[-1]

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_") or "unknown"

def _task_sort_key(task: Any) -> tuple[int, str]:
    task_str = str(task)
    order = {"classification": 0, "regression": 1}
    return order.get(task_str, 99), task_str

def _label_with_task(framework: Any, task: Any) -> str:
    task_str = str(task)
    short = {"classification": "cls", "regression": "reg"}.get(task_str, task_str)
    return f"{framework} [{short}]"

# ---------------------------------------------------------------------------
# 2. Per-worker helpers  (called from _worker.py subprocesses)
# ---------------------------------------------------------------------------

@contextmanager
def wandb_run(
    *,
    project: str,
    entity: Optional[str],
    run_name: str,
    group: str,
    tags: list,
    config: Dict[str, Any],
    job_type: Optional[str] = None,
    enabled: bool = True,
):
    if not enabled or not _wandb_enabled():
        yield None
        return

    import wandb
    init_kwargs = dict(
        project=project,
        entity=entity,
        name=run_name,
        group=group,
        tags=tags,
        config=config,
        reinit=True,
        settings=wandb.Settings(start_method="thread", init_timeout=300),
    )
    if job_type:
        init_kwargs["job_type"] = job_type
    run = wandb.init(**init_kwargs)
    try:
        yield run
    finally:
        try:
            wandb.finish()
        except Exception:
            pass

def log_row(run, row: Dict[str, Any]) -> None:
    if run is None:
        return
    import wandb

    summary: Dict[str, Any] = {}
    for k, v in row.items():
        summary[k] = v if isinstance(v, (int, float, bool)) or v is None else str(v)
    try:
        wandb.run.summary.update(summary)
    except Exception:
        pass

def log_artifact(name: str, artifact_type: str, paths: list) -> None:
    if not _wandb_enabled():
        return
    import wandb
    if wandb.run is None:
        return

    artifact = wandb.Artifact(name=name, type=artifact_type)
    for p in paths:
        p = Path(p)
        if p.is_dir():
            artifact.add_dir(str(p))
        elif p.exists():
            artifact.add_file(str(p))
    wandb.log_artifact(artifact)

def derive_tags(framework: str, task: str, n_rows: int, n_cols: int) -> list:
    return [
        framework,
        task,
        f"nrows_{_bucket(n_rows)}",
        f"ncols_{_bucket(n_cols, edges=(20, 50, 100))}",
    ]

# ---------------------------------------------------------------------------
# 3. Artifact sync  (called by runner at startup)
# ---------------------------------------------------------------------------

def download_results_artifact(
    *, project: str, entity: Optional[str], artifact_name: str,
    out_dir: Path, alias: str = "latest",
) -> bool:
    if not _wandb_enabled():
        return False

    import wandb

    qualified = (
        f"{entity}/{project}/{artifact_name}:{alias}" if entity
        else f"{project}/{artifact_name}:{alias}"
    )
    try:
        artifact = wandb.Api().artifact(qualified)
    except Exception as e:
        print(f"[wandb] no prior artifact ({qualified}): {type(e).__name__}: {e}")
        return False
    try:
        staging = Path(artifact.download())
    except Exception as e:
        print(f"[wandb] artifact download failed: {e}")
        return False

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in staging.rglob("*"):
        if src.is_file():
            dst = out_dir / src.relative_to(staging)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    print(f"[wandb] pulled {copied} file(s) from {qualified} into {out_dir}")
    return copied > 0

# ---------------------------------------------------------------------------
# 4. Orchestrator visualisation helpers + OrchestratorRun
# ---------------------------------------------------------------------------

_RESULTS_TABLE_COLS: List[str] = [
    "dataset_id", "task", "framework", "seed", "scorer_name",
    "score_holdout", "score_nofe_same_seed", "pct_improvement",
    "n_train", "n_test",
    "n_features_before", "n_features_after", "n_added",
    "wall_time_fit", "wall_time_transform", "wall_time_total",
    "peak_rss_mb", "n_boost_rounds",
    "status", "error_msg",
]

_PER_DATASET_TABLE_COLS: List[str] = [
    "dataset_id", "task", "framework", "scorer_name", "framework_task",
    "n_seeds",
    "score_holdout_mean", "score_holdout_std",
    "pct_improvement_mean", "pct_improvement_std",
    "wall_time_fit_mean", "wall_time_total_mean",
    "peak_rss_mb_mean", "n_added_mean",
]

_TASK_SUMMARY_COLS: List[str] = [
    "framework", "task", "framework_task",
    "n_datasets", "n_attempts", "n_ok_runs", "n_non_ok_runs", "non_ok_rate",
    "win_rate",
    "pct_improvement_mean", "pct_improvement_median", "pct_improvement_std",
    "wall_time_fit_mean", "wall_time_total_mean",
    "peak_rss_mb_mean", "n_added_mean", "n_seeds_mean",
]

_SCORER_SUMMARY_COLS: List[str] = [
    "framework", "task", "scorer_name", "framework_metric",
    "n_datasets",
    "score_holdout_mean", "score_holdout_median", "score_holdout_std",
    "wall_time_total_mean",
]

_NUMERIC_RESULT_COLS = (
    "seed", "score_holdout", "score_nofe_same_seed", "pct_improvement",
    "n_train", "n_test",
    "n_features_before", "n_features_after", "n_added",
    "wall_time_fit", "wall_time_transform", "wall_time_total",
    "peak_rss_mb", "n_boost_rounds",
)

def _py_scalar(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)

def _normalize_results_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy() if len(df) else pd.DataFrame(columns=_RESULTS_TABLE_COLS)
    for col in _RESULTS_TABLE_COLS:
        if col not in frame.columns:
            frame[col] = None
    for col in _NUMERIC_RESULT_COLS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    
    if "dataset_id" in frame.columns:
        frame["dataset_id"] = frame["dataset_id"].astype(str)
        
    # Safe fallback for object columns
    for col in ("task", "framework", "scorer_name", "status", "error_msg"):
        frame[col] = frame[col].replace({pd.NA: None})
        
    return frame[_RESULTS_TABLE_COLS].sort_values(
        by=["dataset_id", "framework", "seed"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

def _build_per_dataset_frame(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame(columns=_PER_DATASET_TABLE_COLS)

    grouped = ok.groupby(
        ["dataset_id", "task", "framework", "scorer_name"],
        as_index=False,
        dropna=False,
    ).agg(
        n_seeds=("seed", "nunique"),
        score_holdout_mean=("score_holdout", "mean"),
        score_holdout_std=("score_holdout", "std"),
        pct_improvement_mean=("pct_improvement", "mean"),
        pct_improvement_std=("pct_improvement", "std"),
        wall_time_fit_mean=("wall_time_fit", "mean"),
        wall_time_total_mean=("wall_time_total", "mean"),
        peak_rss_mb_mean=("peak_rss_mb", "mean"),
        n_added_mean=("n_added", "mean"),
    )
    grouped["framework_task"] = grouped.apply(
        lambda r: _label_with_task(r["framework"], r["task"]),
        axis=1,
    )
    return grouped[_PER_DATASET_TABLE_COLS].sort_values(
        by=["task", "framework", "dataset_id"],
        key=lambda s: s.map(_task_sort_key) if s.name == "task" else s,
        kind="stable",
    ).reset_index(drop=True)

def _build_task_summary_frame(df: pd.DataFrame, per_dataset: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=_TASK_SUMMARY_COLS)

    attempts = df.groupby(["framework", "task"], as_index=False).size().rename(
        columns={"size": "n_attempts"}
    )
    ok_runs = df[df["status"] == "ok"].groupby(["framework", "task"], as_index=False).size().rename(
        columns={"size": "n_ok_runs"}
    )
    summary = attempts.merge(ok_runs, on=["framework", "task"], how="left")
    summary["n_ok_runs"] = summary["n_ok_runs"].fillna(0).astype(int)
    summary["n_non_ok_runs"] = summary["n_attempts"] - summary["n_ok_runs"]
    summary["non_ok_rate"] = summary["n_non_ok_runs"] / summary["n_attempts"].clip(lower=1)

    if not per_dataset.empty:
        agg = per_dataset.groupby(["framework", "task"], as_index=False).agg(
            n_datasets=("dataset_id", "nunique"),
            pct_improvement_mean=("pct_improvement_mean", "mean"),
            pct_improvement_median=("pct_improvement_mean", "median"),
            pct_improvement_std=("pct_improvement_mean", "std"),
            wall_time_fit_mean=("wall_time_fit_mean", "mean"),
            wall_time_total_mean=("wall_time_total_mean", "mean"),
            peak_rss_mb_mean=("peak_rss_mb_mean", "mean"),
            n_added_mean=("n_added_mean", "mean"),
            n_seeds_mean=("n_seeds", "mean"),
        )
        win_rate_s = (
            per_dataset.dropna(subset=["pct_improvement_mean"])
            .groupby(["framework", "task"])["pct_improvement_mean"]
            .apply(lambda s: float((s > 0).sum()) / len(s) if len(s) > 0 else float("nan"))
            .rename("win_rate")
            .reset_index()
        )
        agg = agg.merge(win_rate_s, on=["framework", "task"], how="left")
        summary = summary.merge(agg, on=["framework", "task"], how="left")
    else:
        for col in (
            "n_datasets", "win_rate",
            "pct_improvement_mean", "pct_improvement_median",
            "pct_improvement_std", "wall_time_fit_mean", "wall_time_total_mean",
            "peak_rss_mb_mean", "n_added_mean", "n_seeds_mean",
        ):
            summary[col] = None

    summary["framework_task"] = summary.apply(
        lambda r: _label_with_task(r["framework"], r["task"]),
        axis=1,
    )
    return summary[_TASK_SUMMARY_COLS].sort_values(
        by=["task", "pct_improvement_mean", "framework"],
        key=lambda s: s.map(_task_sort_key) if s.name == "task" else s,
        ascending=[True, False, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

def _build_scorer_summary_frame(per_dataset: pd.DataFrame) -> pd.DataFrame:
    if per_dataset.empty:
        return pd.DataFrame(columns=_SCORER_SUMMARY_COLS)

    summary = per_dataset.groupby(["framework", "task", "scorer_name"], as_index=False).agg(
        n_datasets=("dataset_id", "nunique"),
        score_holdout_mean=("score_holdout_mean", "mean"),
        score_holdout_median=("score_holdout_mean", "median"),
        score_holdout_std=("score_holdout_mean", "std"),
        wall_time_total_mean=("wall_time_total_mean", "mean"),
    )
    summary["framework_metric"] = summary.apply(
        lambda r: f"{r['framework']} [{r['scorer_name']}]",
        axis=1,
    )
    return summary[_SCORER_SUMMARY_COLS].sort_values(
        by=["task", "scorer_name", "framework"],
        key=lambda s: s.map(_task_sort_key) if s.name == "task" else s,
        kind="stable",
    ).reset_index(drop=True)

def _load_master_frame(master_csv: Optional[Path]) -> pd.DataFrame:
    if master_csv is None or not master_csv.exists():
        return pd.DataFrame(columns=_RESULTS_TABLE_COLS)
    try:
        raw = pd.read_csv(master_csv, dtype={"dataset_id": str})
    except Exception as e:
        print(f"[wandb] failed to read master.csv for reporting: {e}")
        return pd.DataFrame(columns=_RESULTS_TABLE_COLS)
    return _normalize_results_frame(raw)

def _find_master_csv(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        p = Path(p)
        if p.is_file() and p.name == "master.csv":
            return p
    return None

def _to_wandb_table(df: pd.DataFrame):
    import wandb
    columns = list(df.columns)
    if df.empty:
        return wandb.Table(columns=columns)
    data = [[_py_scalar(v) for v in row] for row in df.itertuples(index=False, name=None)]
    return wandb.Table(columns=columns, data=data)

def _ordered_table_frame(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    frame = df.copy() if len(df) else pd.DataFrame(columns=columns)
    for col in columns:
        if col not in frame.columns:
            frame[col] = None
    return frame[columns].reset_index(drop=True)

def _apply_figure_margins(
    fig,
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    wspace: float = 0.2,
) -> None:
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace)

def log_media_placeholder(run, *, key: str, caption: str) -> None:
    if run is None:
        return
    try:
        import numpy as np
        import wandb

        pixel = np.full((2, 2, 3), 247, dtype=np.uint8)
        run.log({key: wandb.Image(pixel, caption=caption)})
    except Exception:
        pass

def _build_native_plots(
    per_dataset: pd.DataFrame,
    task_summary: pd.DataFrame,
    scorer_summary: pd.DataFrame,
) -> Dict[str, Any]:
    plots: Dict[str, Any] = {}

    if not task_summary.empty:
        plots["results_aggregated"] = _to_wandb_table(task_summary)
        for task in sorted(task_summary["task"].dropna().unique(), key=_task_sort_key):
            sub = task_summary[task_summary["task"] == task].reset_index(drop=True)
            plots[f"results_by_framework_{_slug(task)}"] = _to_wandb_table(sub)

    if not scorer_summary.empty:
        plots["results_by_scorer"] = _to_wandb_table(scorer_summary)
        for scorer_name in sorted(scorer_summary["scorer_name"].dropna().unique()):
            sub = scorer_summary[scorer_summary["scorer_name"] == scorer_name].reset_index(drop=True)
            plots[f"results_by_scorer_{_slug(scorer_name)}"] = _to_wandb_table(sub)

    if not per_dataset.empty:
        plots["results_per_dataset"] = _to_wandb_table(per_dataset)
        for task in sorted(per_dataset["task"].dropna().unique(), key=_task_sort_key):
            sub = per_dataset[per_dataset["task"] == task].reset_index(drop=True)
            plots[f"results_per_dataset_{_slug(task)}"] = _to_wandb_table(sub)

    return plots

def _build_message_figure(title: str, message: str):
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor("#f7f4ef")
        ax.set_facecolor("#fdfbf8")
        ax.axis("off")
        ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=18, fontweight="bold", color="#1f2937")
        ax.text(0.5, 0.38, message, ha="center", va="center", fontsize=11, color="#4b5563", wrap=True)
        _apply_figure_margins(fig, left=0.06, right=0.94, bottom=0.08, top=0.92)
        
        img = wandb.Image(fig)
        return img
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)

def _safe_fig(builder, title: str, message: str):
    """Call builder(); if it returns None or raises, return a labelled placeholder."""
    try:
        result = builder()
        if result is not None:
            return result
    except Exception:
        pass
    try:
        return _build_message_figure(title, message)
    except Exception:
        return None

def _framework_color_map(frameworks: List[str]) -> Dict[str, Any]:
    import matplotlib.pyplot as plt
    palette = list(plt.get_cmap("Set2").colors) + list(plt.get_cmap("tab10").colors)
    return {fw: palette[i % len(palette)] for i, fw in enumerate(sorted(frameworks))}

def _build_pct_improvement_figure(task_summary: pd.DataFrame):
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        if task_summary.empty or task_summary["pct_improvement_mean"].dropna().empty:
            return _build_message_figure(
                "Feature Engineering Benchmark",
                "No completed baseline-comparable runs yet. The figure will appear once pct_improvement is available."
            )

        tasks = ["classification", "regression"]
        frameworks = task_summary["framework"].dropna().astype(str).unique().tolist()
        colors = _framework_color_map(frameworks)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
        fig.patch.set_facecolor("#fbf8f3")

        for ax, task in zip(axes, tasks):
            ax.set_facecolor("#fffdfa")
            sub = task_summary[task_summary["task"] == task].copy()
            sub = sub.dropna(subset=["pct_improvement_mean"]).sort_values("pct_improvement_mean", ascending=False)
            
            if sub.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center", fontsize=13, color="#6b7280")
                continue

            y = list(range(len(sub)))
            vals = sub["pct_improvement_mean"].tolist()
            errs = sub["pct_improvement_std"].fillna(0.0).tolist()
            labels = sub["framework"].astype(str).tolist()
            bar_colors = [colors[lbl] for lbl in labels]
            
            ax.barh(y, vals, xerr=errs, color=bar_colors, alpha=0.92, edgecolor="#374151", linewidth=0.6)
            ax.axvline(0, color="#374151", linewidth=1.0, linestyle="--")
            ax.grid(axis="x", linestyle=":", alpha=0.35)
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_title(task.title(), fontsize=14, fontweight="bold")
            ax.set_xlabel("Mean % improvement vs no-FE")
            
            for idx, (_, row) in enumerate(sub.iterrows()):
                val = float(row["pct_improvement_mean"])
                pad = max(abs(val) * 0.02, 0.01)
                x_text = val + pad if val >= 0 else val - pad
                align = "left" if val >= 0 else "right"
                ax.text(
                    x_text, idx, f"{val:.3f}  (n={int(row['n_datasets'])})",
                    va="center", ha=align, fontsize=9, color="#111827"
                )

        fig.suptitle("Feature Engineering Improvement by Task", fontsize=17, fontweight="bold", color="#111827")
        _apply_figure_margins(fig, left=0.16, right=0.98, bottom=0.12, top=0.86, wspace=0.3)
        
        img = wandb.Image(fig)
        return img
    except Exception:
        return None
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)

def _build_pareto_figure(per_dataset: pd.DataFrame):
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        if per_dataset.empty or per_dataset["pct_improvement_mean"].dropna().empty:
            return _build_message_figure("Runtime vs Improvement", "Pareto view will appear after dataset-level results.")

        tasks = ["classification", "regression"]
        frameworks = per_dataset["framework"].dropna().astype(str).unique().tolist()
        colors = _framework_color_map(frameworks)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
        fig.patch.set_facecolor("#fbf8f3")

        for ax, task in zip(axes, tasks):
            ax.set_facecolor("#fffdfa")
            sub = per_dataset[per_dataset["task"] == task].copy()
            sub = sub.dropna(subset=["wall_time_total_mean", "pct_improvement_mean"])
            
            if sub.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center", fontsize=13, color="#6b7280")
                continue

            for fw, grp in sub.groupby("framework"):
                ax.scatter(
                    grp["wall_time_total_mean"], grp["pct_improvement_mean"],
                    s=54, alpha=0.82, color=colors[str(fw)],
                    edgecolors="#374151", linewidths=0.4, label=str(fw),
                )
                
            ax.set_xscale("log")
            ax.axhline(0, color="#374151", linewidth=1.0, linestyle="--")
            ax.grid(True, linestyle=":", alpha=0.3)
            ax.set_title(task.title(), fontsize=14, fontweight="bold")
            ax.set_xlabel("Mean total runtime (s, log scale)")
            ax.set_ylabel("Mean % improvement vs no-FE")

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False)
            
        fig.suptitle("Pareto View: Runtime vs Improvement", fontsize=17, fontweight="bold", color="#111827")
        _apply_figure_margins(fig, left=0.08, right=0.98, bottom=0.14, top=0.82, wspace=0.14)
        
        img = wandb.Image(fig)
        return img
    except Exception as e:
        print(f"[wandb] Error building pareto_figure: {e}")
        return None
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)

def _build_failure_rate_figure(task_summary: pd.DataFrame):
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        if task_summary.empty:
            return _build_message_figure("Failure and Timeout Rates", "No benchmark attempts logged yet.")

        tasks = ["classification", "regression"]
        frameworks = task_summary["framework"].dropna().astype(str).unique().tolist()
        colors = _framework_color_map(frameworks)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
        fig.patch.set_facecolor("#fbf8f3")

        for ax, task in zip(axes, tasks):
            ax.set_facecolor("#fffdfa")
            sub = task_summary[task_summary["task"] == task].copy().sort_values("non_ok_rate", ascending=False)
            
            if sub.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center", fontsize=13, color="#6b7280")
                continue

            x = list(range(len(sub)))
            vals = sub["non_ok_rate"].fillna(0.0).tolist()
            labels = sub["framework"].astype(str).tolist()
            bar_colors = [colors[lbl] for lbl in labels]
            
            ax.bar(x, vals, color=bar_colors, alpha=0.92, edgecolor="#374151", linewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylim(0, max(0.05, min(1.0, max(vals) * 1.25 if vals else 0.05)))
            ax.grid(axis="y", linestyle=":", alpha=0.35)
            ax.set_title(task.title(), fontsize=14, fontweight="bold")
            ax.set_ylabel("Non-OK rate")
            
            for idx, (_, row) in enumerate(sub.iterrows()):
                rate = float(row["non_ok_rate"])
                ax.text(
                    idx, rate + 0.01, f"{rate:.1%}\n{int(row['n_non_ok_runs'])}/{int(row['n_attempts'])}",
                    ha="center", va="bottom", fontsize=8.5, color="#111827",
                )

        fig.suptitle("Crash / Timeout / Unsupported Rate by Task", fontsize=17, fontweight="bold", color="#111827")
        _apply_figure_margins(fig, left=0.08, right=0.98, bottom=0.24, top=0.85, wspace=0.18)

        img = wandb.Image(fig)
        return img
    except Exception:
        return None
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)

def _build_win_rate_figure(task_summary: pd.DataFrame):
    """Bar chart: fraction of datasets where FE beat no-FE, per framework and task."""
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        if task_summary.empty or "win_rate" not in task_summary.columns or task_summary["win_rate"].dropna().empty:
            return None

        tasks = ["classification", "regression"]
        frameworks = task_summary["framework"].dropna().astype(str).unique().tolist()
        colors = _framework_color_map(frameworks)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
        fig.patch.set_facecolor("#fbf8f3")

        for ax, task in zip(axes, tasks):
            ax.set_facecolor("#fffdfa")
            sub = task_summary[task_summary["task"] == task].copy()
            sub = sub.dropna(subset=["win_rate"]).sort_values("win_rate", ascending=False)

            if sub.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center", fontsize=13, color="#6b7280")
                continue

            y = list(range(len(sub)))
            vals = (sub["win_rate"] * 100).tolist()
            labels = sub["framework"].astype(str).tolist()
            bar_colors = [colors[lbl] for lbl in labels]

            ax.barh(y, vals, color=bar_colors, alpha=0.92, edgecolor="#374151", linewidth=0.6)
            ax.axvline(50, color="#374151", linewidth=1.0, linestyle="--")
            ax.set_xlim(0, 110)
            ax.grid(axis="x", linestyle=":", alpha=0.35)
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_title(task.title(), fontsize=14, fontweight="bold")
            ax.set_xlabel("Win rate (% datasets FE > no-FE)")

            for idx, (_, row) in enumerate(sub.iterrows()):
                val = float(row["win_rate"]) * 100
                n = int(row["n_datasets"]) if pd.notna(row.get("n_datasets")) else "?"
                ax.text(
                    min(val + 1.5, 108), idx, f"{val:.0f}%  (n={n})",
                    va="center", ha="left", fontsize=9, color="#111827",
                )

        fig.suptitle("Win Rate: Datasets Where Feature Engineering Helped", fontsize=16, fontweight="bold", color="#111827")
        _apply_figure_margins(fig, left=0.16, right=0.98, bottom=0.12, top=0.86, wspace=0.3)

        img = wandb.Image(fig)
        return img
    except Exception:
        return None
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)


def _build_score_distribution_figure(per_dataset: pd.DataFrame):
    """Box plot of holdout scores per framework, using only the dominant scorer per task.

    Scores are only comparable within the same scorer, so we filter to the
    most-common scorer_name per task before plotting.
    """
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        if per_dataset.empty or per_dataset["score_holdout_mean"].dropna().empty:
            return _build_message_figure("Score Distribution", "No score data available yet.")

        tasks = ["classification", "regression"]
        frameworks = per_dataset["framework"].dropna().astype(str).unique().tolist()
        colors = _framework_color_map(frameworks)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
        fig.patch.set_facecolor("#fbf8f3")

        for ax, task in zip(axes, tasks):
            ax.set_facecolor("#fffdfa")
            task_sub = per_dataset[per_dataset["task"] == task].dropna(subset=["score_holdout_mean"])

            if task_sub.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center", fontsize=13, color="#6b7280")
                continue

            # Find the dominant scorer so absolute values are comparable
            dominant_scorer = (
                task_sub["scorer_name"].value_counts().idxmax()
                if "scorer_name" in task_sub.columns else None
            )
            if dominant_scorer:
                sub = task_sub[task_sub["scorer_name"] == dominant_scorer]
            else:
                sub = task_sub

            fw_order = (
                sub.groupby("framework")["score_holdout_mean"]
                .median()
                .sort_values(ascending=False)
                .index.tolist()
            )
            box_data = [sub[sub["framework"] == fw]["score_holdout_mean"].dropna().tolist() for fw in fw_order]

            bp = ax.boxplot(box_data, patch_artist=True, notch=False, vert=True)
            for patch, fw in zip(bp["boxes"], fw_order):
                patch.set_facecolor(colors.get(str(fw), "#4b5563"))
                patch.set_alpha(0.85)
            for element in bp["whiskers"] + bp["caps"]:
                element.set_color("#374151")
                element.set_linewidth(1.2)
            for median in bp["medians"]:
                median.set_color("#111827")
                median.set_linewidth(2.0)
            for flier in bp["fliers"]:
                flier.set(marker=".", markerfacecolor="#374151", alpha=0.5, markersize=5)

            ax.set_xticks(range(1, len(fw_order) + 1))
            ax.set_xticklabels(fw_order, rotation=20, ha="right")
            ax.grid(axis="y", linestyle=":", alpha=0.35)
            scorer_label = f" ({dominant_scorer})" if dominant_scorer else ""
            ax.set_title(f"{task.title()}{scorer_label}", fontsize=13, fontweight="bold")
            ax.set_ylabel("Holdout score")

        fig.suptitle("Score Distribution by Framework (dominant scorer per task)", fontsize=15, fontweight="bold", color="#111827")
        _apply_figure_margins(fig, left=0.08, right=0.98, bottom=0.22, top=0.86, wspace=0.25)

        img = wandb.Image(fig)
        return img
    except Exception as e:
        print(f"[wandb] Error building score_distribution_figure: {e}")
        return None
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)


def _build_per_dataset_improvement_figure(per_dataset: pd.DataFrame, *, top_n: int = 40):
    """Strip chart of per-dataset pct_improvement per framework.

    The single most informative benchmark plot: shows exactly which datasets
    each framework helps or hurts, with all frameworks overlaid as colored dots.
    Datasets are sorted by mean improvement across frameworks (ascending),
    capped at top_n most-affected datasets to keep the figure readable.
    """
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        # Exclude nofe — it's 0 by definition and clutters the chart
        sub_all = per_dataset[per_dataset["framework"] != "nofe"].copy()
        sub_all = sub_all.dropna(subset=["pct_improvement_mean"])
        if sub_all.empty:
            return None

        tasks = [t for t in ["classification", "regression"] if t in sub_all["task"].values]
        if not tasks:
            return None

        frameworks = sorted(sub_all["framework"].dropna().astype(str).unique())
        colors = _framework_color_map(frameworks)
        n_tasks = len(tasks)

        fig, axes = plt.subplots(1, n_tasks, figsize=(8 * n_tasks, max(6, top_n * 0.22)))
        if n_tasks == 1:
            axes = [axes]
        fig.patch.set_facecolor("#fbf8f3")

        for ax, task in zip(axes, tasks):
            ax.set_facecolor("#fffdfa")
            sub = sub_all[sub_all["task"] == task].copy()

            # Rank datasets by mean pct_improvement across frameworks
            ds_means = (
                sub.groupby("dataset_id")["pct_improvement_mean"].mean().sort_values(ascending=True)
            )
            if len(ds_means) > top_n:
                keep = ds_means.abs().nlargest(top_n).index
                ds_means = ds_means[keep].sort_values(ascending=True)

            ds_ids = ds_means.index.tolist()
            ds_y = {ds: i for i, ds in enumerate(ds_ids)}

            for fw in frameworks:
                fw_sub = sub[sub["framework"] == fw].set_index("dataset_id")
                xs, ys = [], []
                for ds in ds_ids:
                    if ds in fw_sub.index:
                        xs.append(float(fw_sub.loc[ds, "pct_improvement_mean"]))
                        ys.append(ds_y[ds])
                if xs:
                    ax.scatter(xs, ys, s=40, alpha=0.85, color=colors.get(fw, "#6b7280"),
                               edgecolors="#374151", linewidths=0.3, label=fw, zorder=3)

            ax.axvline(0, color="#374151", linewidth=1.2, linestyle="--", alpha=0.7)
            ax.grid(axis="x", linestyle=":", alpha=0.3)

            # Show dataset IDs on y-axis (truncated to keep compact)
            ax.set_yticks(range(len(ds_ids)))
            ax.set_yticklabels([str(d)[:22] for d in ds_ids], fontsize=7.5)
            ax.set_xlabel("Mean % improvement vs no-FE")
            ax.set_title(task.title(), fontsize=14, fontweight="bold")

            handles, lbls = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, lbls, loc="lower right", fontsize=9, frameon=True, framealpha=0.85)

        n_total = sub_all["dataset_id"].nunique()
        shown = len(ds_means)
        note = f"top {shown} by |improvement|" if n_total > top_n else f"all {n_total} datasets"
        fig.suptitle(f"Per-Dataset Improvement by Framework ({note})", fontsize=15, fontweight="bold", color="#111827")
        fig.tight_layout(rect=[0, 0, 1, 0.94])

        img = wandb.Image(fig)
        return img
    except Exception as e:
        print(f"[wandb] Error building per_dataset_improvement_figure: {e}")
        return None
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)


def _build_features_added_figure(task_summary: pd.DataFrame):
    """Bar chart: mean features added per framework, annotated with mean % improvement."""
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        if task_summary.empty or "n_added_mean" not in task_summary.columns:
            return None
        ts = task_summary.dropna(subset=["n_added_mean"])
        if ts[ts["framework"] != "nofe"].empty:
            return None

        tasks = ["classification", "regression"]
        frameworks = ts["framework"].dropna().astype(str).unique().tolist()
        colors = _framework_color_map(frameworks)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
        fig.patch.set_facecolor("#fbf8f3")

        for ax, task in zip(axes, tasks):
            ax.set_facecolor("#fffdfa")
            sub = (
                ts[(ts["task"] == task) & (ts["framework"] != "nofe")]
                .dropna(subset=["n_added_mean"])
                .sort_values("n_added_mean", ascending=False)
            )
            if sub.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, f"No {task} data", ha="center", va="center", fontsize=13, color="#6b7280")
                continue

            x = list(range(len(sub)))
            vals = sub["n_added_mean"].tolist()
            labels = sub["framework"].astype(str).tolist()
            bar_colors = [colors.get(lbl, "#6b7280") for lbl in labels]

            bars = ax.bar(x, vals, color=bar_colors, alpha=0.88, edgecolor="#374151", linewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.grid(axis="y", linestyle=":", alpha=0.35)
            ax.set_ylabel("Mean features added")
            ax.set_title(task.title(), fontsize=14, fontweight="bold")

            for bar, (_, row) in zip(bars, sub.iterrows()):
                h = bar.get_height()
                pct = row.get("pct_improvement_mean")
                label = f"{h:.1f}"
                if pd.notna(pct):
                    sign = "+" if pct >= 0 else ""
                    label += f"\n({sign}{pct:.3f}%)"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + max(h * 0.02, 0.3),
                    label, ha="center", va="bottom", fontsize=8.5, color="#111827",
                )

        fig.suptitle("Mean Features Added per Framework (with mean % improvement)", fontsize=15, fontweight="bold", color="#111827")
        _apply_figure_margins(fig, left=0.08, right=0.98, bottom=0.22, top=0.86, wspace=0.28)

        img = wandb.Image(fig)
        return img
    except Exception as e:
        print(f"[wandb] Error building features_added_figure: {e}")
        return None
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)


class OrchestratorRun:
    def __init__(
        self,
        *,
        project: str,
        entity: Optional[str],
        artifact_name: str,
        enabled: bool = True,
    ):
        self.project = project
        self.entity = entity
        self.artifact_name = artifact_name
        self.enabled = bool(enabled and _wandb_enabled())
        self._run = None
        self._last_push = 0.0
        self._report_step = 0
        self._pending_rows: List[Dict[str, Any]] = []

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            import wandb
            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                id=self.artifact_name,
                resume="allow",
                name="orchestrator",
                job_type="orchestrator",
                tags=["orchestrator", "benchmark"],
                reinit=True,
                settings=wandb.Settings(start_method="thread", init_timeout=300),
            )
        except Exception as e:
            print(f"[wandb] orchestrator init failed; artifact sync disabled: {e}")
            self.enabled = False
        return self

    def __exit__(self, *exc):
        if self._run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        return False

    def append_result(self, row: dict) -> None:
        self._pending_rows.append(dict(row))

    def _result_columns(self) -> List[str]:
        return _RESULTS_TABLE_COLS

    def _load_per_run_frame(self, master_csv: Optional[Path]) -> pd.DataFrame:
        return _load_master_frame(master_csv)

    def _build_extra_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        del snapshot
        return {}

    def _build_snapshot(self, master_csv: Optional[Path]) -> Dict[str, Any]:
        per_run_df = self._load_per_run_frame(master_csv)
        if not per_run_df.empty and "task" in per_run_df.columns:
            per_run_df = per_run_df.copy()
            per_run_df["task"] = per_run_df["task"].replace("multiclass", "classification")
        per_dataset_df = _build_per_dataset_frame(per_run_df)
        task_summary_df = _build_task_summary_frame(per_run_df, per_dataset_df)
        scorer_summary_df = _build_scorer_summary_frame(per_dataset_df)
        snapshot: Dict[str, Any] = {
            "per_run_df": per_run_df,
            "per_dataset_df": per_dataset_df,
            "task_summary_df": task_summary_df,
            "scorer_summary_df": scorer_summary_df,
        }
        snapshot.update(self._build_extra_snapshot(snapshot))
        return snapshot

    def _build_metrics(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        per_run_df = snapshot["per_run_df"]
        per_dataset_df = snapshot["per_dataset_df"]
        task_summary_df = snapshot["task_summary_df"]
        return {
            "n_rows_total": int(len(per_run_df)),
            "n_ok_rows": int((per_run_df["status"] == "ok").sum()) if not per_run_df.empty else 0,
            "n_framework_task_pairs": int(len(task_summary_df)),
            "n_dataset_framework_rows": int(len(per_dataset_df)),
        }

    def _build_table_payload(self, snapshot: Dict[str, Any], *, final: bool = False) -> Dict[str, Any]:
        payload = _build_native_plots(
            snapshot["per_dataset_df"],
            snapshot["task_summary_df"],
            snapshot["scorer_summary_df"],
        )
        if final:
            ordered = _ordered_table_frame(snapshot["per_run_df"], self._result_columns())
            payload["results_per_run"] = _to_wandb_table(ordered)
        return payload

    def _figure_specs(self, snapshot: Dict[str, Any]) -> List[tuple]:
        ts = snapshot["task_summary_df"]
        pd_ = snapshot["per_dataset_df"]
        _w = "Results will appear here as the benchmark progresses."
        return [
            ("figure_pct_improvement",       lambda: _build_pct_improvement_figure(ts),         "% Improvement",          _w),
            ("figure_win_rate",              lambda: _build_win_rate_figure(ts),                 "Win Rate",               _w),
            ("figure_runtime_vs_improvement",lambda: _build_pareto_figure(pd_),                  "Runtime vs Improvement", _w),
            ("figure_failure_rate",          lambda: _build_failure_rate_figure(ts),             "Failure Rate",           _w),
            ("figure_score_distribution",    lambda: _build_score_distribution_figure(pd_),      "Score Distribution",     _w),
            ("figure_per_dataset_improvement",lambda: _build_per_dataset_improvement_figure(pd_),"Per-Dataset Improvement",_w),
            ("figure_features_added",        lambda: _build_features_added_figure(ts),           "Features Added",         _w),
        ]

    def _build_figure_payload(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        # Every key is always logged — either the real chart or a "waiting for data"
        # placeholder. This prevents media panels from showing "no data" errors.
        payload: Dict[str, Any] = {}
        for key, builder, title, message in self._figure_specs(snapshot):
            fig = _safe_fig(builder, title, message)
            if fig is not None:
                payload[key] = fig
        return payload

    def push(self, paths: List[Path], *, force: bool = False, min_interval_s: float = 30.0) -> bool:
        if not self.enabled or self._run is None:
            return False
        now = time.time()
        if not force and (now - self._last_push) < min_interval_s:
            return False

        try:
            import wandb
            artifact = wandb.Artifact(name=self.artifact_name, type="benchmark_results")
            for p in paths:
                p = Path(p)
                if p.is_dir():
                    artifact.add_dir(str(p))
                elif p.exists():
                    artifact.add_file(str(p))
            wandb.log_artifact(artifact)

            master_csv = _find_master_csv(paths)
            snapshot = self._build_snapshot(master_csv)

            # Scalar metrics → logged with step so they appear as charts over time
            metrics = self._build_metrics(snapshot)
            tables = self._build_table_payload(snapshot, final=force)
            figures = self._build_figure_payload(snapshot)

            self._report_step += 1

            # One run.log() call per step: scalars → line charts, figures → media panels,
            # lightweight tables → workspace table panels.
            # Heavy per-run table is only logged to summary (too large for every sync).
            log_dict: Dict[str, Any] = {}
            log_dict.update(metrics)
            log_dict.update(figures)
            log_dict.update({k: v for k, v in tables.items() if k != "results_per_run"})
            self._run.log(log_dict, step=self._report_step)

            # Summary: all tables + metrics always reflect the latest state
            self._run.summary.update({**metrics, **tables})

            self._last_push = now
            self._pending_rows.clear()
            return True
        except Exception as e:
            print(f"[wandb] artifact push failed: {e}")
            return False
