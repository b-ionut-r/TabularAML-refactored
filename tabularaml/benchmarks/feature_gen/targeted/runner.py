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
    time_budget_s: int = 600
    n_jobs: int = -1
    mode: str = "medium"
    framework_kwargs: dict = field(default_factory=dict)
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
    ordered = {c: row.get(c) for c in columns}
    df = pd.DataFrame([ordered])
    lock = _acquire_lock(csv_path)
    with lock:
        header = not csv_path.exists()
        df.to_csv(csv_path, mode="a", header=header, index=False)


def _load_master(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.read_csv(csv_path)


def _done_key_set(master: pd.DataFrame, retry_crashes: bool = False) -> set:
    if len(master) == 0:
        return set()
    terminal = {
        "ok", "timeout", "oom", "contract_violation", "unsupported_task",
        "dataset_fetch_failed", "degenerate_dataset",
        "autofeat_internal_nan", "autofeat_upstream_bug", "featuretools_upstream_bug",
    }
    if not retry_crashes:
        terminal.add("crash")
    done = master[master["status"].isin(terminal)]
    return {
        (str(r.dataset_id), str(r.framework), int(r.seed))
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
        time_budget_s: int = 600,
        n_workers: int = 4,
        tabularaml_mode: str = "medium",
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
                    key = (str(ds_spec.id), fw, int(seed))
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
        key = (str(row["dataset_id"]), int(row["seed"]))
        nofe_score = nofe_lookup.get(key)
        if nofe_score is not None and row.get("score_holdout") is not None:
            row["score_nofe_same_seed"] = float(nofe_score["score_holdout"])
            denom = abs(row["score_nofe_same_seed"])
            if denom > 0:
                raw = (float(row["score_holdout"]) - row["score_nofe_same_seed"]) / denom
                gib = bool(nofe_score.get("scorer_greater_is_better", True))
                row["pct_improvement"] = float(raw if gib else -raw)
            else:
                row["pct_improvement"] = 0.0
        else:
            row["score_nofe_same_seed"] = None
            row["pct_improvement"] = None
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
            nofe_lookup[(str(row["dataset_id"]), int(row["seed"]))] = {
                "score_holdout":            float(row["score_holdout"]),
                "scorer_greater_is_better": bool(row.get("scorer_greater_is_better", True)),
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
                    nofe_lookup[(str(r.dataset_id), int(r.seed))] = {
                        "score_holdout":            float(r.score_holdout),
                        "scorer_greater_is_better": bool(getattr(r, "scorer_greater_is_better", True)),
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
                            futures = [pool.submit(_dispatch, (self, s)) for s in specs]
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
