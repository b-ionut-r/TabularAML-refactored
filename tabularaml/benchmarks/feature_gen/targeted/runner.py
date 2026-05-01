"""Orchestrator for targeted feature-engineering benchmark suites.

Each (dataset, framework, seed) triple runs in an isolated subprocess.
Supports three curated suites: amlb, pmlb, stress_test (or "all").

Usage::

    from tabularaml.benchmarks.feature_gen.targeted.runner import TargetedBenchmarkRunner
    runner = TargetedBenchmarkRunner(suite="stress_test", frameworks=["nofe", "tabularaml"],
                                     seeds=[0], results_dir=Path("./results/targeted"))
    runner.run()
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from tqdm.auto import tqdm

from .registry import get_suite, DatasetSpec
from .wandb_suite import TargetedOrchestratorRun, _TARGETED_RESULT_COLUMNS
from tabularaml.benchmarks.feature_gen.wandb_logger import download_results_artifact


RESULT_COLUMNS = _TARGETED_RESULT_COLUMNS


@dataclass
class TargetedRunSpec:
    dataset_id: str          # OpenML task ID (as string) or PMLB dataset name
    dataset_source: str      # "openml_task" | "pmlb"
    dataset_name: str        # Human-readable label
    task: str
    suite: str
    rationale: str
    framework: str
    seed: int
    time_budget_s: int = 1200
    n_jobs: int = -1
    mode: str = "medium"
    framework_kwargs: dict = field(default_factory=dict)
    task_filter: Optional[list[str]] = None
    wandb_enabled: bool = True
    wandb_project: str = "tabularaml-targeted-benchmark"
    wandb_entity: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def _acquire_lock(path: Path):
    try:
        from filelock import FileLock
        return FileLock(str(path) + ".lock")
    except ImportError:
        class _NoLock:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return _NoLock()


def _append_row(csv_path: Path, row: dict, columns: Sequence[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(csv_path)
    with lock:
        if not csv_path.exists():
            ordered = {c: row.get(c) for c in columns}
            pd.DataFrame([ordered], columns=list(columns)).to_csv(csv_path, index=False)
            return

        try:
            existing_cols = list(pd.read_csv(csv_path, nrows=0).columns)
        except Exception:
            existing_cols = []

        full_columns = list(existing_cols)
        for c in columns:
            if c not in full_columns:
                full_columns.append(c)

        if full_columns != existing_cols:
            existing = pd.read_csv(csv_path)
            for c in full_columns:
                if c not in existing.columns:
                    existing[c] = None
            existing = existing[full_columns]
            existing.to_csv(csv_path, index=False)

        ordered = {c: row.get(c) for c in full_columns}
        pd.DataFrame([ordered], columns=full_columns).to_csv(
            csv_path, mode="a", header=False, index=False
        )


def _load_master(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.read_csv(csv_path)


def _finite_float(value) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _row_value(row, name: str, default=None):
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _result_key(row, *, framework: str | None = None) -> tuple:
    key = (
        str(_row_value(row, "dataset_source", "") or ""),
        str(_row_value(row, "dataset_id")),
        str(_row_value(row, "suite", "") or ""),
        int(_row_value(row, "seed")),
    )
    if framework is None:
        return key
    return (*key, str(framework))


def _parse_metrics_json(raw, *, scorer_name=None, score_holdout=None) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {}
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if value is None:
                    metrics[str(key)] = None
                else:
                    metrics[str(key)] = _finite_float(value)
    if scorer_name and score_holdout is not None and scorer_name not in metrics:
        score = _finite_float(score_holdout)
        if score is not None:
            metrics[str(scorer_name)] = score
    return metrics


def _dump_metrics_json(metrics: dict[str, float | None]) -> str:
    return json.dumps(metrics, sort_keys=True) if metrics else ""


def _done_key_set(master: pd.DataFrame, retry_crashes: bool = False) -> set:
    if len(master) == 0:
        return set()
    terminal = {
        "ok", "timeout", "oom", "contract_violation", "unsupported_task",
        "dataset_fetch_failed", "degenerate_dataset",
        "autofeat_internal_nan", "autofeat_upstream_bug", "featuretools_upstream_bug",
        "openfe_upstream_bug",
    }
    if not retry_crashes:
        terminal.add("crash")
    done = master[master["status"].isin(terminal)]
    return {
        _result_key(r, framework=str(r.framework))
        for r in done.itertuples(index=False)
    }


def _dispatch(args: tuple) -> tuple:
    runner, spec = args
    return spec, runner._run_one_subprocess(spec)


class TargetedBenchmarkRunner:
    def __init__(
        self,
        suite: str,
        frameworks: Sequence[str],
        seeds: Sequence[int],
        results_dir: Path,
        time_budget_s: int = 1200,
        n_workers: int = 4,
        tabularaml_mode: str = "medium",
        tasks: Optional[Sequence[str]] = None,
        wandb_project: str = "tabularaml-targeted-benchmark",
        wandb_entity: Optional[str] = None,
        wandb_enabled: bool = True,
        skip_existing: bool = True,
        retry_crashes: bool = False,
        nofe_first: bool = True,
        artifact_name: Optional[str] = None,
        artifact_sync: bool = True,
        sync_every_rows: int = 5,
        sync_min_interval_s: float = 30.0,
    ):
        self.suite = suite
        self.frameworks = list(frameworks)
        self.seeds = [int(s) for s in seeds]
        self.results_dir = Path(results_dir)
        self.master_csv = self.results_dir / "master.csv"
        self.raw_dir = self.results_dir / "raw"
        self.time_budget_s = int(time_budget_s)
        self.n_workers = max(1, int(n_workers))
        self.tabularaml_mode = tabularaml_mode
        self.tasks = set(tasks) if tasks else None
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_enabled = bool(wandb_enabled)
        self.skip_existing = skip_existing
        self.retry_crashes = retry_crashes
        self.nofe_first = nofe_first
        self.artifact_name = artifact_name or f"targeted-{suite}-results"
        self.artifact_sync = artifact_sync
        self.sync_every_rows = max(1, int(sync_every_rows))
        self.sync_min_interval_s = float(sync_min_interval_s)

    def sync_from_wandb(self) -> bool:
        if not self.artifact_sync or not self.wandb_enabled:
            return False
        return download_results_artifact(
            project=self.wandb_project,
            entity=self.wandb_entity,
            artifact_name=self.artifact_name,
            out_dir=self.results_dir,
        )

    def build_run_plan(self) -> list[TargetedRunSpec]:
        specs_registry = get_suite(self.suite)
        if self.tasks is not None:
            specs_registry = [s for s in specs_registry if s.task in self.tasks]

        fws = list(self.frameworks)
        if self.nofe_first and "nofe" in fws:
            fws = ["nofe"] + [f for f in fws if f != "nofe"]

        master = _load_master(self.master_csv)
        done = _done_key_set(master, retry_crashes=self.retry_crashes) if self.skip_existing else set()

        n_cpus = multiprocessing.cpu_count()
        n_jobs_per_worker = max(1, n_cpus // self.n_workers)

        run_specs: list[TargetedRunSpec] = []
        for ds_spec in specs_registry:
            for seed in self.seeds:
                for fw in fws:
                    key = (
                        str(ds_spec.source),
                        str(ds_spec.id),
                        str(ds_spec.suite),
                        int(seed),
                        str(fw),
                    )
                    if key in done:
                        continue
                    run_specs.append(TargetedRunSpec(
                        dataset_id=str(ds_spec.id),
                        dataset_source=ds_spec.source,
                        dataset_name=ds_spec.name,
                        task=ds_spec.task,
                        suite=ds_spec.suite,
                        rationale=ds_spec.rationale,
                        framework=fw,
                        seed=int(seed),
                        time_budget_s=self.time_budget_s,
                        n_jobs=n_jobs_per_worker,
                        mode=self.tabularaml_mode if fw == "tabularaml" else "medium",
                        task_filter=sorted(self.tasks) if self.tasks else None,
                        wandb_enabled=self.wandb_enabled,
                        wandb_project=self.wandb_project,
                        wandb_entity=self.wandb_entity,
                    ))
        return run_specs

    def _run_one_subprocess(self, spec: TargetedRunSpec) -> dict:
        env = os.environ.copy()
        if self.n_workers > 1:
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("LOKY_MAX_CPU_COUNT", "1")

        with tempfile.NamedTemporaryFile(
            prefix="tgt_bench_", suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as fh:
            out_path = fh.name

        cmd = [
            sys.executable, "-m",
            "tabularaml.benchmarks.feature_gen.targeted._worker",
            "--spec", spec.to_json(),
            "--out", out_path,
        ]
        grace_s = 180
        hard_cap = spec.time_budget_s + grace_s
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, env=env, timeout=hard_cap,
                capture_output=True, text=True,
            )
            row = None
            try:
                text = Path(out_path).read_text(encoding="utf-8").strip()
                if text:
                    row = json.loads(text)
            except Exception:
                pass
            if row is None:
                tail = (proc.stdout or "").strip().splitlines()
                if tail:
                    try:
                        row = json.loads(tail[-1])
                    except json.JSONDecodeError:
                        row = None
            if row is None:
                stderr = proc.stderr or ""
                tb_start = stderr.find("Traceback (most recent call last)")
                error_snippet = stderr[tb_start:][:1200] if tb_start != -1 else stderr[-800:]
                row = {
                    "dataset_id":     spec.dataset_id,
                    "dataset_name":   spec.dataset_name,
                    "dataset_source": spec.dataset_source,
                    "suite":          spec.suite,
                    "task":           spec.task,
                    "framework":      spec.framework,
                    "seed":           spec.seed,
                    "time_budget_s":  spec.time_budget_s,
                    "status":         "crash",
                    "error_msg":      error_snippet,
                    "wall_time_total": time.time() - t0,
                }
        except subprocess.TimeoutExpired:
            row = {
                "dataset_id":     spec.dataset_id,
                "dataset_name":   spec.dataset_name,
                "dataset_source": spec.dataset_source,
                "suite":          spec.suite,
                "task":           spec.task,
                "framework":      spec.framework,
                "seed":           spec.seed,
                "time_budget_s":  spec.time_budget_s,
                "status":         "timeout",
                "error_msg":      f"wall-clock exceeded {hard_cap}s",
                "wall_time_total": hard_cap,
            }
        finally:
            try:
                os.unlink(out_path)
            except Exception:
                pass
        return row

    def _attach_pct_improvement(self, row: dict, nofe_lookup: dict) -> dict:
        from tabularaml.benchmarks.feature_gen.evaluator import (
            compute_metric_gains,
            pct_improvement,
        )
        from tabularaml.eval.scorers import PREDEFINED_SCORERS

        key = _result_key(row)
        nofe_score = nofe_lookup.get(key)
        row_metrics = _parse_metrics_json(
            row.get("all_metrics_json"),
            scorer_name=row.get("scorer_name"),
            score_holdout=row.get("score_holdout"),
        )
        if nofe_score is not None:
            baseline_metrics = nofe_score.get("all_metrics", {})
            scorer_name = row.get("scorer_name")
            row_score = row_metrics.get(scorer_name) if scorer_name else None
            if row_score is None:
                row_score = _finite_float(row.get("score_holdout"))
            baseline_score = baseline_metrics.get(scorer_name) if scorer_name else None
            if baseline_score is None and scorer_name == nofe_score.get("scorer_name"):
                baseline_score = _finite_float(nofe_score.get("score_holdout"))

            row["score_nofe_same_seed"] = baseline_score
            scorer = PREDEFINED_SCORERS.get(str(scorer_name)) if scorer_name else None
            if row_score is not None and baseline_score is not None and scorer is not None:
                row["pct_improvement"] = float(pct_improvement(row_score, baseline_score, scorer))
            else:
                row["pct_improvement"] = None
            row["metric_gains_json"] = _dump_metrics_json(
                compute_metric_gains(row_metrics, baseline_metrics)
            )
        else:
            row["score_nofe_same_seed"] = None
            row["pct_improvement"] = None
            row["metric_gains_json"] = _dump_metrics_json(
                {metric_name: None for metric_name in row_metrics}
            )
        return row

    def _sync_paths(self) -> list[Path]:
        paths = []
        if self.master_csv.exists():
            paths.append(self.master_csv)
        if self.raw_dir.exists():
            paths.append(self.raw_dir)
        return paths

    def _finalize_row(self, row: dict, nofe_lookup: dict, orch=None) -> None:
        if row.get("framework") == "nofe" and row.get("status") == "ok":
            nofe_lookup[_result_key(row)] = {
                "score_holdout":            _finite_float(row.get("score_holdout")),
                "scorer_name":              row.get("scorer_name"),
                "scorer_greater_is_better": bool(row.get("scorer_greater_is_better", True)),
                "all_metrics": _parse_metrics_json(
                    row.get("all_metrics_json"),
                    scorer_name=row.get("scorer_name"),
                    score_holdout=row.get("score_holdout"),
                ),
            }
        row = self._attach_pct_improvement(row, nofe_lookup)
        _append_row(self.master_csv, row, RESULT_COLUMNS)
        fw_csv = self.raw_dir / f"{row['framework']}.csv"
        _append_row(fw_csv, row, RESULT_COLUMNS)
        if orch is not None:
            orch.append_result(row)

    def run(self) -> None:
        self.sync_from_wandb()
        specs = self.build_run_plan()

        with TargetedOrchestratorRun(
            project=self.wandb_project,
            entity=self.wandb_entity,
            artifact_name=self.artifact_name,
            suite=self.suite,
            enabled=self.wandb_enabled and self.artifact_sync,
        ) as orch:
            if not specs:
                print("No runs scheduled (all rows already complete).")
                if self.master_csv.exists():
                    orch.push(self._sync_paths(), force=True)
                return

            n_unique_ds = len({s.dataset_id for s in specs})
            print(f"[targeted:{self.suite}] Scheduled {len(specs)} runs "
                  f"({n_unique_ds} datasets × {len(self.seeds)} seed(s) × {len(self.frameworks)} frameworks)")

            master = _load_master(self.master_csv)
            nofe_lookup: dict = {}
            if len(master):
                nofe_rows = master[(master["framework"] == "nofe") & (master["status"] == "ok")]
                for r in nofe_rows.itertuples(index=False):
                    nofe_lookup[_result_key(r)] = {
                        "score_holdout":            _finite_float(getattr(r, "score_holdout", None)),
                        "scorer_name":              getattr(r, "scorer_name", None),
                        "scorer_greater_is_better": bool(getattr(r, "scorer_greater_is_better", True)),
                        "all_metrics": _parse_metrics_json(
                            getattr(r, "all_metrics_json", ""),
                            scorer_name=getattr(r, "scorer_name", None),
                            score_holdout=getattr(r, "score_holdout", None),
                        ),
                    }

            n_done = 0
            try:
                with tqdm(total=len(specs), desc=f"targeted:{self.suite}") as pbar:
                    if self.n_workers == 1:
                        for spec in specs:
                            _, row = _dispatch((self, spec))
                            self._finalize_row(row, nofe_lookup, orch)
                            n_done += 1
                            pbar.update(1)
                            if n_done % self.sync_every_rows == 0:
                                orch.push(self._sync_paths(),
                                          min_interval_s=self.sync_min_interval_s)
                    else:
                        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                            # Split into two phases: nofe first, then the rest
                            nofe_specs = [s for s in specs if s.framework == "nofe"]
                            other_specs = [s for s in specs if s.framework != "nofe"]
                            
                            # Phase 1: nofe
                            if nofe_specs:
                                futures = [pool.submit(_dispatch, (self, s)) for s in nofe_specs]
                                for fut in as_completed(futures):
                                    _, row = fut.result()
                                    self._finalize_row(row, nofe_lookup, orch)
                                    n_done += 1
                                    pbar.update(1)
                                    if n_done % self.sync_every_rows == 0:
                                        orch.push(self._sync_paths(),
                                                  min_interval_s=self.sync_min_interval_s)
                                                  
                            # Phase 2: others
                            if other_specs:
                                futures = [pool.submit(_dispatch, (self, s)) for s in other_specs]
                                for fut in as_completed(futures):
                                    _, row = fut.result()
                                    self._finalize_row(row, nofe_lookup, orch)
                                    n_done += 1
                                    pbar.update(1)
                                    if n_done % self.sync_every_rows == 0:
                                        orch.push(self._sync_paths(),
                                                  min_interval_s=self.sync_min_interval_s)
            finally:
                if self.master_csv.exists():
                    orch.push(self._sync_paths(), force=True)
