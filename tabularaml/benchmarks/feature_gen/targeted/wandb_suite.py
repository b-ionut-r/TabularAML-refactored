"""Suite-level W&B logging for targeted benchmarks.

Builds on the shared orchestrator reporter with:
- live incremental results for in-flight monitoring
- mutable suite/task summary tables during execution
- immutable final tables on the closing sync
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from tabularaml.benchmarks.feature_gen.wandb_logger import (
    _build_per_dataset_frame,
    _build_task_summary_frame,
    _build_scorer_summary_frame,
    _build_pct_improvement_figure,
    _build_pareto_figure,
    _build_failure_rate_figure,
    _to_wandb_table,
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

        if not ok.empty:
            ds_pct = ok.groupby("dataset_id")["pct_improvement"].mean().dropna()
            ds_wt  = ok.groupby("dataset_id")["wall_time_total"].mean().dropna()
            ds_rss = ok.groupby("dataset_id")["peak_rss_mb"].mean().dropna()

            pct_mean = ds_pct.mean() if not ds_pct.empty else float("nan")
            pct_med  = ds_pct.median() if not ds_pct.empty else float("nan")
            wt_mean  = ds_wt.mean() if not ds_wt.empty else float("nan")
            rss_mean = ds_rss.mean() if not ds_rss.empty else float("nan")

            win_rate = (ds_pct > 0).sum() / len(ds_pct) if len(ds_pct) > 0 else float("nan")
            n_datasets = int(ok["dataset_id"].nunique())
        else:
            pct_mean = pct_med = wt_mean = rss_mean = win_rate = float("nan")
            n_datasets = 0

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
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        if suite_summary.empty:
            return None

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
        return img
    except Exception:
        return None
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt
            plt.close(fig)


class TargetedOrchestratorRun(OrchestratorRun):
    """Long-lived orchestrator run for targeted benchmarks.

    Extends OrchestratorRun with suite-level comparison charts. All media
    is emitted through the shared push path so long runs keep updating while
    still ending with immutable full tables on the dashboard.
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
        super().__init__(
            project=project,
            entity=entity,
            artifact_name=artifact_name,
            enabled=enabled
        )
        self.suite = suite

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
                settings=wandb.Settings(start_method="thread", init_timeout=300),
            )
        except Exception as e:
            print(f"[wandb] targeted orchestrator init failed: {e}")
            self.enabled = False
        return self

    def _result_columns(self):
        return _TARGETED_RESULT_COLUMNS

    def _load_per_run_frame(self, master_csv):
        return _load_master_frame_targeted(master_csv)

    def _build_extra_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "suite_summary_df": _build_suite_summary(
                snapshot["per_run_df"],
                snapshot["per_dataset_df"],
            )
        }

    def _build_table_payload(self, snapshot: Dict[str, Any], *, final: bool = False) -> Dict[str, Any]:
        payload = super()._build_table_payload(snapshot, final=False)
        payload["results_task_summary"] = _to_wandb_table(snapshot["task_summary_df"])
        payload["results_suite_summary"] = _to_wandb_table(snapshot["suite_summary_df"])
        if final:
            ordered = snapshot["per_run_df"].reindex(columns=self._result_columns())
            payload["results_per_run"] = _to_wandb_table(ordered)
        return payload

    def _build_figure_payload(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        payload = super()._build_figure_payload(snapshot)
        rank_fig = _build_suite_rank_figure(snapshot["suite_summary_df"])
        if rank_fig is not None:
            payload["figure_suite_rank"] = rank_fig
        return payload

    def _build_metrics(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        metrics = super()._build_metrics(snapshot)
        per_run_df = snapshot["per_run_df"]
        metrics["n_datasets"] = int(per_run_df["dataset_id"].nunique()) if not per_run_df.empty else 0
        metrics["n_suites_started"] = int(per_run_df["suite"].nunique()) if "suite" in per_run_df.columns else 0
        return metrics


def _load_master_frame_targeted(master_csv: Optional[Path]) -> pd.DataFrame:
    if master_csv is None or not master_csv.exists():
        return pd.DataFrame(columns=_TARGETED_RESULT_COLUMNS)
    try:
        df = pd.read_csv(master_csv, dtype={"dataset_id": str})
    except Exception as e:
        print(f"[wandb] failed to read targeted master.csv: {e}")
        return pd.DataFrame(columns=_TARGETED_RESULT_COLUMNS)
        
    numeric_cols = ("score_holdout", "score_nofe_same_seed", "pct_improvement",
                "wall_time_total", "wall_time_fit", "wall_time_transform",
                "peak_rss_mb", "n_train", "n_test", "n_features_before",
                "n_features_after", "n_added", "n_boost_rounds")
                
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in _TARGETED_RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    for col in ("dataset_name", "dataset_source", "suite", "task", "framework", "scorer_name", "status", "error_msg"):
        if col in df.columns:
            df[col] = df[col].replace({pd.NA: None})
            
    if "dataset_id" in df.columns:
        df["dataset_id"] = df["dataset_id"].astype(str)

    return df[_TARGETED_RESULT_COLUMNS].sort_values(
        by=["suite", "task", "dataset_id", "framework", "seed"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
