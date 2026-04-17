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


def _is_lower_better_scorer(scorer_name: Any) -> bool:
    return str(scorer_name) in {"rmse", "categorical_crossentropy", "logloss", "mae", "mse"}


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
    """Context manager that yields a wandb.run or None.

    Always closes the run on exit even if the inner block raises.
    """
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
        settings=wandb.Settings(start_method="thread"),
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
    """Write all result fields to the run summary.

    Uses summary.update() (not wandb.log) so each run appears as a single
    bar/point in W&B's comparison view rather than a flat line chart.
    """
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
    """Attach file/dir paths as a wandb artifact on the active run."""
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
    """Pull `{entity}/{project}/{artifact_name}:{alias}` into `out_dir`.

    Copies every file in the artifact to out_dir (preserving basenames), so a
    prior master.csv / raw/*.csv snapshot rehydrates on top of an empty dir.

    Returns True if files were copied; False on first run or if wandb is
    disabled. Never raises - errors are printed and the benchmark runs anyway.
    """
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
    "dataset_id", "seed", "score_holdout", "score_nofe_same_seed", "pct_improvement",
    "n_train", "n_test",
    "n_features_before", "n_features_after", "n_added",
    "wall_time_fit", "wall_time_transform", "wall_time_total",
    "peak_rss_mb", "n_boost_rounds",
)


def _py_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _normalize_results_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy() if len(df) else pd.DataFrame(columns=_RESULTS_TABLE_COLS)
    for col in _RESULTS_TABLE_COLS:
        if col not in frame.columns:
            frame[col] = None
    for col in _NUMERIC_RESULT_COLS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in ("task", "framework", "scorer_name", "status", "error_msg"):
        frame[col] = frame[col].where(frame[col].notna(), None)
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
        summary = summary.merge(agg, on=["framework", "task"], how="left")
    else:
        for col in (
            "n_datasets", "pct_improvement_mean", "pct_improvement_median",
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
        raw = pd.read_csv(master_csv)
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

    if df.empty:
        return wandb.Table(columns=list(df.columns))
    data = [[_py_scalar(v) for v in row] for row in df.itertuples(index=False, name=None)]
    return wandb.Table(columns=list(df.columns), data=data)


def _apply_figure_margins(
    fig,
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    wspace: float = 0.2,
) -> None:
    """Use explicit subplot margins for crowded dashboard figures.

    The benchmark plots combine suptitles, legends, rotated tick labels, and
    dense annotations; ``tight_layout`` is prone to emitting warnings there.
    """
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace)


def log_media_placeholder(run, *, key: str, caption: str) -> None:
    """Log a tiny placeholder image so W&B media panels have a stable key."""
    if run is None:
        return
    try:
        import numpy as np
        import wandb

        pixel = np.full((2, 2, 3), 247, dtype=np.uint8)
        run.log({key: wandb.Image(pixel, caption=caption)})
    except Exception:
        pass


def _build_native_plots(per_dataset: pd.DataFrame, task_summary: pd.DataFrame, scorer_summary: pd.DataFrame) -> Dict[str, Any]:
    import wandb

    plots: Dict[str, Any] = {}

    if not task_summary.empty:
        combined_task = task_summary.copy()
        plots["results_aggregated"] = _to_wandb_table(combined_task)
        try:
            plots["chart_pct_improvement"] = wandb.plot.bar(
                plots["results_aggregated"],
                "framework_task",
                "pct_improvement_mean",
                title="Mean % Improvement vs No-FE by Framework / Task",
            )
        except Exception:
            pass
        try:
            plots["chart_fit_time"] = wandb.plot.bar(
                plots["results_aggregated"],
                "framework_task",
                "wall_time_total_mean",
                title="Mean Total Runtime (s) by Framework / Task",
            )
        except Exception:
            pass
        try:
            plots["chart_n_added"] = wandb.plot.bar(
                plots["results_aggregated"],
                "framework_task",
                "n_added_mean",
                title="Mean Features Added by Framework / Task",
            )
        except Exception:
            pass
        try:
            plots["chart_failure_rate"] = wandb.plot.bar(
                plots["results_aggregated"],
                "framework_task",
                "non_ok_rate",
                title="Non-OK Rate by Framework / Task",
            )
        except Exception:
            pass

        for task in sorted(task_summary["task"].dropna().unique(), key=_task_sort_key):
            sub = task_summary[task_summary["task"] == task].reset_index(drop=True)
            task_key = _slug(task)
            table_key = f"results_by_framework_{task_key}"
            plots[table_key] = _to_wandb_table(sub)
            try:
                plots[f"chart_pct_improvement_{task_key}"] = wandb.plot.bar(
                    plots[table_key],
                    "framework",
                    "pct_improvement_mean",
                    title=f"Mean % Improvement vs No-FE ({task})",
                )
            except Exception:
                pass
            try:
                plots[f"chart_fit_time_{task_key}"] = wandb.plot.bar(
                    plots[table_key],
                    "framework",
                    "wall_time_total_mean",
                    title=f"Mean Total Runtime (s) ({task})",
                )
            except Exception:
                pass
            try:
                plots[f"chart_n_added_{task_key}"] = wandb.plot.bar(
                    plots[table_key],
                    "framework",
                    "n_added_mean",
                    title=f"Mean Features Added ({task})",
                )
            except Exception:
                pass
            try:
                plots[f"chart_failure_rate_{task_key}"] = wandb.plot.bar(
                    plots[table_key],
                    "framework",
                    "non_ok_rate",
                    title=f"Non-OK Rate ({task})",
                )
            except Exception:
                pass

    if not scorer_summary.empty:
        combined_scorer = scorer_summary.copy()
        plots["results_by_scorer"] = _to_wandb_table(combined_scorer)
        try:
            plots["chart_score_holdout"] = wandb.plot.bar(
                plots["results_by_scorer"],
                "framework_metric",
                "score_holdout_mean",
                title="Mean Holdout Score by Framework / Scorer",
            )
        except Exception:
            pass
        for scorer_name in sorted(scorer_summary["scorer_name"].dropna().unique()):
            sub = scorer_summary[scorer_summary["scorer_name"] == scorer_name].reset_index(drop=True)
            scorer_key = _slug(scorer_name)
            table_key = f"results_by_scorer_{scorer_key}"
            plots[table_key] = _to_wandb_table(sub)
            try:
                plots[f"chart_score_holdout_{scorer_key}"] = wandb.plot.bar(
                    plots[table_key],
                    "framework",
                    "score_holdout_mean",
                    title=f"Mean Holdout Score ({scorer_name})",
                )
            except Exception:
                pass

    if not per_dataset.empty:
        plots["results_per_dataset"] = _to_wandb_table(per_dataset)
        try:
            plots["chart_time_vs_improvement"] = wandb.plot.scatter(
                plots["results_per_dataset"],
                "wall_time_total_mean",
                "pct_improvement_mean",
                title="Runtime vs % Improvement (per dataset mean)",
            )
        except Exception:
            pass
        for task in sorted(per_dataset["task"].dropna().unique(), key=_task_sort_key):
            sub = per_dataset[per_dataset["task"] == task].reset_index(drop=True)
            task_key = _slug(task)
            table_key = f"results_per_dataset_{task_key}"
            plots[table_key] = _to_wandb_table(sub)
            try:
                plots[f"chart_time_vs_improvement_{task_key}"] = wandb.plot.scatter(
                    plots[table_key],
                    "wall_time_total_mean",
                    "pct_improvement_mean",
                    title=f"Runtime vs % Improvement ({task})",
                )
            except Exception:
                pass

    return plots


def _build_message_figure(title: str, message: str):
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
    plt.close(fig)
    return img


def _framework_color_map(frameworks: List[str]) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    palette = list(plt.get_cmap("Set2").colors) + list(plt.get_cmap("tab10").colors)
    return {fw: palette[i % len(palette)] for i, fw in enumerate(sorted(frameworks))}


def _build_pct_improvement_figure(task_summary: pd.DataFrame):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        if task_summary.empty or task_summary["pct_improvement_mean"].dropna().empty:
            return _build_message_figure(
                "Feature Engineering Benchmark",
                "No completed baseline-comparable runs yet. "
                "The figure will appear once pct_improvement is available.",
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
                    x_text,
                    idx,
                    f"{val:.3f}  (n={int(row['n_datasets'])})",
                    va="center",
                    ha=align,
                    fontsize=9,
                    color="#111827",
                )

        fig.suptitle("Feature Engineering Improvement by Task", fontsize=17, fontweight="bold", color="#111827")
        _apply_figure_margins(fig, left=0.16, right=0.98, bottom=0.12, top=0.86, wspace=0.3)
        img = wandb.Image(fig)
        plt.close(fig)
        return img
    except Exception:
        return None


def _build_pareto_figure(per_dataset: pd.DataFrame):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        if per_dataset.empty or per_dataset["pct_improvement_mean"].dropna().empty:
            return _build_message_figure(
                "Runtime vs Improvement",
                "Pareto view will appear after the first completed dataset-level results.",
            )

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
                    grp["wall_time_total_mean"],
                    grp["pct_improvement_mean"],
                    s=54,
                    alpha=0.82,
                    color=colors[str(fw)],
                    edgecolors="#374151",
                    linewidths=0.4,
                    label=str(fw),
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
        plt.close(fig)
        return img
    except Exception:
        return None


def _build_failure_rate_figure(task_summary: pd.DataFrame):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        if task_summary.empty:
            return _build_message_figure(
                "Failure and Timeout Rates",
                "No benchmark attempts have been logged yet.",
            )

        tasks = ["classification", "regression"]
        frameworks = task_summary["framework"].dropna().astype(str).unique().tolist()
        colors = _framework_color_map(frameworks)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
        fig.patch.set_facecolor("#fbf8f3")

        for ax, task in zip(axes, tasks):
            ax.set_facecolor("#fffdfa")
            sub = task_summary[task_summary["task"] == task].copy()
            sub = sub.sort_values("non_ok_rate", ascending=False)
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
                    idx,
                    rate + 0.01,
                    f"{rate:.1%}\n{int(row['n_non_ok_runs'])}/{int(row['n_attempts'])}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color="#111827",
                )

        fig.suptitle("Crash / Timeout / Unsupported Rate by Task", fontsize=17, fontweight="bold", color="#111827")
        _apply_figure_margins(fig, left=0.08, right=0.98, bottom=0.24, top=0.85, wspace=0.18)
        img = wandb.Image(fig)
        plt.close(fig)
        return img
    except Exception:
        return None


class OrchestratorRun:
    """Long-lived wandb run owned by the benchmark orchestrator.

    Responsibilities:
    - Upload versioned `{artifact_name}` artifacts (master.csv + raw/*.csv)
      so results survive ephemeral environments (HF Spaces, Colab, Vast.ai).
    - Build all benchmark charts from the authoritative `master.csv`, not just
      rows seen in the current process, so resumed runs stay accurate.

    Per-(dataset, framework, seed) runs are independent and spawned by workers.
    """

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

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            import wandb

            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                id=self.artifact_name,  # Unified orchestrator ID per artifact
                resume="allow",
                name="orchestrator",
                job_type="orchestrator",
                tags=["orchestrator", "benchmark"],
                reinit=True,
                settings=wandb.Settings(start_method="thread"),
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
        """Compatibility no-op.

        Reporting is rebuilt from master.csv on every push so resumed sessions
        and fresh orchestrator processes show the same benchmark state.
        """
        del row
        return None

    def push(self, paths: List[Path], *, force: bool = False, min_interval_s: float = 30.0) -> bool:
        """Upload a versioned artifact and refresh all charts/tables.

        Debounced - no-ops if the last push was within `min_interval_s`
        unless `force=True`.
        """
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
            per_run_df = _load_master_frame(master_csv)
            per_dataset_df = _build_per_dataset_frame(per_run_df)
            task_summary_df = _build_task_summary_frame(per_run_df, per_dataset_df)
            scorer_summary_df = _build_scorer_summary_frame(per_dataset_df)

            log_dict: Dict[str, Any] = {
                "results": _to_wandb_table(per_run_df),
                "results_per_run": _to_wandb_table(per_run_df),
                **_build_native_plots(per_dataset_df, task_summary_df, scorer_summary_df),
            }

            pct_fig = _build_pct_improvement_figure(task_summary_df)
            if pct_fig is not None:
                log_dict["figure_pct_improvement"] = pct_fig

            pareto_fig = _build_pareto_figure(per_dataset_df)
            if pareto_fig is not None:
                log_dict["figure_runtime_vs_improvement"] = pareto_fig

            failure_fig = _build_failure_rate_figure(task_summary_df)
            if failure_fig is not None:
                log_dict["figure_failure_rate"] = failure_fig

            self._run.log(log_dict)
            self._run.summary.update({
                "n_rows_total": int(len(per_run_df)),
                "n_ok_rows": int((per_run_df["status"] == "ok").sum()) if not per_run_df.empty else 0,
                "n_framework_task_pairs": int(len(task_summary_df)),
                "n_dataset_framework_rows": int(len(per_dataset_df)),
            })

            self._last_push = now
            return True
        except Exception as e:
            print(f"[wandb] artifact push failed: {e}")
            return False
