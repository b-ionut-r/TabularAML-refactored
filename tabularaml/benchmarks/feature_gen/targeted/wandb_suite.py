"""Suite-level W&B logging for targeted benchmarks.

Adds cross-suite comparison tables and charts on top of the per-suite charts
already built by wandb_logger.py helpers. All media is written via
`run.summary.update()` so panels always display the latest value (avoids the
"plots disappear after ~100 pushes" bug caused by repeated `run.log()` calls).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from tabularaml.benchmarks.feature_gen.wandb_logger import (
    _build_per_dataset_frame,
    _build_task_summary_frame,
    _build_scorer_summary_frame,
    _build_pct_improvement_figure,
    _build_pareto_figure,
    _build_failure_rate_figure,
    _to_wandb_table,
    _load_master_frame,
    _wandb_enabled,
    OrchestratorRun,
)

_SUITE_SUMMARY_COLS = [
    "suite", "framework", "task",
    "n_datasets", "n_ok_runs", "non_ok_rate",
    "pct_improvement_mean", "pct_improvement_median",
    "win_rate",             # fraction of datasets where framework > nofe
    "wall_time_total_mean",
    "peak_rss_mb_mean",
]

_TARGETED_RESULT_COLUMNS = [
    "dataset_id", "dataset_name", "dataset_source", "suite",
    "task", "framework", "seed", "time_budget_s",
    "n_train", "n_test",
    "n_features_before", "n_features_after", "n_added",
    "score_holdout", "scorer_name", "scorer_greater_is_better",
    "score_nofe_same_seed", "pct_improvement",
    "wall_time_fit", "wall_time_transform", "wall_time_total",
    "peak_rss_mb", "n_boost_rounds",
    "status", "error_msg", "adapter_version", "internal_log_json",
]


def _build_suite_summary(df: pd.DataFrame, per_dataset: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=_SUITE_SUMMARY_COLS)

    rows = []
    for (suite, fw, task), grp in df.groupby(["suite", "framework", "task"], dropna=False):
        n_attempts = len(grp)
        ok = grp[grp["status"] == "ok"]
        n_ok = len(ok)
        non_ok_rate = 1.0 - (n_ok / n_attempts) if n_attempts else float("nan")

        pct_mean = ok["pct_improvement"].mean() if not ok.empty else float("nan")
        pct_med  = ok["pct_improvement"].median() if not ok.empty else float("nan")
        wt_mean  = ok["wall_time_total"].mean() if not ok.empty else float("nan")
        rss_mean = ok["peak_rss_mb"].mean() if not ok.empty else float("nan")

        win_rate = float("nan")
        if not ok.empty and "pct_improvement" in ok.columns:
            valid = ok["pct_improvement"].dropna()
            win_rate = (valid > 0).sum() / len(valid) if len(valid) else float("nan")

        n_datasets = int(ok["dataset_id"].nunique()) if not ok.empty else 0

        rows.append({
            "suite":                str(suite),
            "framework":            str(fw),
            "task":                 str(task),
            "n_datasets":           n_datasets,
            "n_ok_runs":            n_ok,
            "non_ok_rate":          non_ok_rate,
            "pct_improvement_mean": pct_mean,
            "pct_improvement_median": pct_med,
            "win_rate":             win_rate,
            "wall_time_total_mean": wt_mean,
            "peak_rss_mb_mean":     rss_mean,
        })

    return pd.DataFrame(rows)[_SUITE_SUMMARY_COLS].sort_values(
        ["suite", "task", "pct_improvement_mean"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)


def _build_suite_rank_figure(suite_summary: pd.DataFrame):
    """Bump chart: framework rank by mean pct_improvement across suites."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        if suite_summary.empty:
            return None

        # Average across tasks per (suite, framework)
        agg = suite_summary.groupby(["suite", "framework"], as_index=False).agg(
            pct_improvement_mean=("pct_improvement_mean", "mean")
        ).dropna(subset=["pct_improvement_mean"])

        suites = sorted(agg["suite"].unique())
        frameworks = sorted(agg["framework"].unique())
        if len(suites) < 2 or len(frameworks) < 2:
            return None

        palette = list(plt.get_cmap("Set2").colors)
        colors = {fw: palette[i % len(palette)] for i, fw in enumerate(frameworks)}

        fig, ax = plt.subplots(figsize=(max(8, len(suites) * 3), 5))
        fig.patch.set_facecolor("#fbf8f3")
        ax.set_facecolor("#fffdfa")

        for fw in frameworks:
            ranks = []
            for suite in suites:
                sub = agg[agg["suite"] == suite].sort_values(
                    "pct_improvement_mean", ascending=False
                ).reset_index(drop=True)
                if fw in sub["framework"].values:
                    rank = int(sub[sub["framework"] == fw].index[0]) + 1
                else:
                    rank = len(frameworks) + 1
                ranks.append(rank)
            ax.plot(range(len(suites)), ranks, marker="o", label=fw,
                    color=colors[fw], linewidth=2, markersize=8)
            ax.text(len(suites) - 1 + 0.05, ranks[-1], fw,
                    va="center", fontsize=9, color=colors[fw])

        ax.set_xticks(range(len(suites)))
        ax.set_xticklabels(suites, fontsize=11)
        ax.invert_yaxis()
        ax.set_ylabel("Rank (1 = best)")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.set_title("Framework Rank by Mean % Improvement Across Suites", fontsize=14, fontweight="bold")

        plt.tight_layout()
        img = wandb.Image(fig)
        plt.close(fig)
        return img
    except Exception:
        return None


class TargetedOrchestratorRun:
    """Long-lived orchestrator run for targeted benchmarks.

    Extends OrchestratorRun with suite-level comparison charts. All media
    objects are written via summary.update() to avoid the plot-disappearance
    bug that affects repeated wandb.run.log() calls on long runs.
    """

    def __init__(
        self,
        *,
        project: str,
        entity: Optional[str],
        artifact_name: str,
        suite: str,
        enabled: bool = True,
    ):
        self.project = project
        self.entity = entity
        self.artifact_name = artifact_name
        self.suite = suite
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
                id=self.artifact_name,
                resume="allow",
                name=f"orchestrator-{self.suite}",
                job_type="orchestrator",
                group=f"targeted_{self.suite}",
                tags=["orchestrator", "targeted", self.suite],
                reinit=True,
                settings=wandb.Settings(start_method="thread"),
            )
        except Exception as e:
            print(f"[wandb] targeted orchestrator init failed: {e}")
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
        del row

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
            df = _load_master_frame_targeted(master_csv)

            per_dataset  = _build_per_dataset_frame(df)
            task_summary = _build_task_summary_frame(df, per_dataset)
            scorer_summary = _build_scorer_summary_frame(per_dataset)
            suite_summary = _build_suite_summary(df, per_dataset)

            # All media → summary (never log) so panels stay stable
            media: Dict[str, Any] = {
                "results":           _to_wandb_table(df[list(set(_TARGETED_RESULT_COLUMNS) & set(df.columns))]),
                "results_per_dataset": _to_wandb_table(per_dataset),
                "results_task_summary": _to_wandb_table(task_summary),
                "results_suite_summary": _to_wandb_table(suite_summary),
            }

            # Per-suite bar charts
            for suite_name in df["suite"].dropna().unique() if "suite" in df.columns else []:
                sub_df = df[df["suite"] == suite_name]
                sub_pd = _build_per_dataset_frame(sub_df)
                sub_ts = _build_task_summary_frame(sub_df, sub_pd)
                try:
                    media[f"chart_{suite_name}_pct_improvement"] = wandb.plot.bar(
                        _to_wandb_table(sub_ts),
                        "framework_task",
                        "pct_improvement_mean",
                        title=f"[{suite_name}] Mean % Improvement vs No-FE",
                    )
                except Exception:
                    pass
                try:
                    media[f"chart_{suite_name}_runtime"] = wandb.plot.bar(
                        _to_wandb_table(sub_ts),
                        "framework_task",
                        "wall_time_total_mean",
                        title=f"[{suite_name}] Mean Total Runtime (s)",
                    )
                except Exception:
                    pass

            pct_fig = _build_pct_improvement_figure(task_summary)
            if pct_fig is not None:
                media["figure_pct_improvement"] = pct_fig

            pareto_fig = _build_pareto_figure(per_dataset)
            if pareto_fig is not None:
                media["figure_runtime_vs_improvement"] = pareto_fig

            failure_fig = _build_failure_rate_figure(task_summary)
            if failure_fig is not None:
                media["figure_failure_rate"] = failure_fig

            rank_fig = _build_suite_rank_figure(suite_summary)
            if rank_fig is not None:
                media["figure_suite_rank"] = rank_fig

            self._run.summary.update(media)
            self._run.summary.update({
                "n_rows_total":     int(len(df)),
                "n_ok_rows":        int((df["status"] == "ok").sum()) if not df.empty else 0,
                "n_datasets":       int(df["dataset_id"].nunique()) if not df.empty else 0,
                "n_suites_started": int(df["suite"].nunique()) if "suite" in df.columns else 0,
            })
            if not df.empty:
                self._run.log({"n_rows_total": int(len(df))})

            self._last_push = now
            return True
        except Exception as e:
            print(f"[wandb] targeted push failed: {e}")
            return False


def _find_master_csv(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        p = Path(p)
        if p.is_file() and p.name == "master.csv":
            return p
    return None


def _load_master_frame_targeted(master_csv: Optional[Path]) -> pd.DataFrame:
    if master_csv is None or not master_csv.exists():
        return pd.DataFrame(columns=_TARGETED_RESULT_COLUMNS)
    try:
        df = pd.read_csv(master_csv)
    except Exception as e:
        print(f"[wandb] failed to read targeted master.csv: {e}")
        return pd.DataFrame(columns=_TARGETED_RESULT_COLUMNS)
    for col in ("score_holdout", "score_nofe_same_seed", "pct_improvement",
                "wall_time_total", "wall_time_fit", "wall_time_transform",
                "peak_rss_mb", "n_train", "n_test", "n_features_before",
                "n_features_after", "n_added", "n_boost_rounds"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
