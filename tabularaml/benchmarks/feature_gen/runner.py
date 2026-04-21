"""Orchestrator for the cross-framework FE benchmark.

Each (dataset_id, framework, seed) triple runs in an isolated Python
subprocess. Timeouts, crashes, and OOMs are caught per-run and recorded as a
row in master.csv + results/raw/{framework}.csv — the orchestrator never
dies because a worker did.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Sequence
import pandas as pd
from tqdm.auto import tqdm

from .manifest import load_manifest, subset_manifest
from .wandb_logger import OrchestratorRun, download_results_artifact


RESULT_COLUMNS = [
    "dataset_id", "task", "framework", "seed", "time_budget_s",
    "n_train", "n_test",
    "n_features_before", "n_features_after", "n_added",
    "score_holdout", "scorer_name", "scorer_greater_is_better",
    "score_nofe_same_seed", "pct_improvement",
    "wall_time_fit", "wall_time_transform", "wall_time_total",
    "peak_rss_mb", "n_boost_rounds",
    "status", "error_msg", "adapter_version", "internal_log_json",
]


@dataclass
class RunSpec:
    dataset_id: int
    task: str
    framework: str
    seed: int
    time_budget_s: int = 1200
    n_jobs: int = -1
    mode: str = "medium"
    framework_kwargs: dict = field(default_factory=dict)
    wandb_enabled: bool = True
    wandb_project: str = "tabularaml-fe-benchmark"
    wandb_entity: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def _acquire_lock(path: Path):
    """Best-effort file lock using filelock if available, no-op otherwise."""
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


def _done_key_set(master: pd.DataFrame, retry_crashes: bool = True) -> set:
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
        (int(r.dataset_id), str(r.framework), int(r.seed))
        for r in done.itertuples(index=False)
    }


def _dispatch(args: tuple) -> tuple:
    runner, spec = args
    return spec, runner._run_one_subprocess(spec)


class BenchmarkRunner:
    def __init__(
        self,
        manifest_path: Path,
        frameworks: Sequence[str],
        seeds: Sequence[int],
        results_dir: Path,
        time_budget_s: int = 1200,
        n_workers: int = 1,
        subset: str = "full",
        tabularaml_mode: str = "medium",
        wandb_project: str = "tabularaml-fe-benchmark",
        wandb_entity: Optional[str] = None,
        wandb_enabled: bool = True,
        skip_existing: bool = True,
        retry_crashes: bool = False,
        nofe_first: bool = True,
        artifact_name: str = "benchmark_results",
        artifact_sync: bool = True,
        sync_every_rows: int = 5,
        sync_min_interval_s: float = 30.0,
    ):
        self.manifest_path = Path(manifest_path)
        self.frameworks = list(frameworks)
        self.seeds = [int(s) for s in seeds]
        self.results_dir = Path(results_dir)
        self.master_csv = self.results_dir / "master.csv"
        self.raw_dir = self.results_dir / "raw"
        self.time_budget_s = int(time_budget_s)
        self.n_workers = max(1, int(n_workers))
        self.subset = subset
        self.tabularaml_mode = tabularaml_mode
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_enabled = bool(wandb_enabled)
        self.skip_existing = skip_existing
        self.retry_crashes = retry_crashes
        self.nofe_first = nofe_first
        self.artifact_name = artifact_name
        self.artifact_sync = artifact_sync
        self.sync_every_rows = max(1, int(sync_every_rows))
        self.sync_min_interval_s = float(sync_min_interval_s)

    def sync_from_wandb(self) -> bool:
        """Pull the latest benchmark_results artifact into results_dir.

        Called at startup by run(). Enables restart-safe execution on
        ephemeral environments (HF Spaces, Colab, Modal) where the local disk
        is wiped between sessions. After this returns True, master.csv holds
        the completed rows from all prior sessions.
        """
        if not self.artifact_sync or not self.wandb_enabled:
            return False
        return download_results_artifact(
            project=self.wandb_project,
            entity=self.wandb_entity,
            artifact_name=self.artifact_name,
            out_dir=self.results_dir,
        )

    def build_run_plan(self) -> list[RunSpec]:
        manifest = subset_manifest(load_manifest(self.manifest_path), self.subset)
        # Order frameworks so 'nofe' runs first per (dataset, seed); this lets
        # analysis fill pct_improvement in a single pass.
        fws = list(self.frameworks)
        if self.nofe_first and "nofe" in fws:
            fws = ["nofe"] + [f for f in fws if f != "nofe"]

        master = _load_master(self.master_csv)
        done = _done_key_set(master, retry_crashes=self.retry_crashes) if self.skip_existing else set()

        n_cpus = multiprocessing.cpu_count()
        n_jobs_per_worker = max(1, n_cpus // self.n_workers)

        specs: list[RunSpec] = []
        for _, row in manifest.iterrows():
            for seed in self.seeds:
                for fw in fws:
                    key = (int(row["tid"]), fw, int(seed))
                    if key in done:
                        continue
                    specs.append(RunSpec(
                        dataset_id=int(row["tid"]),
                        task=row["task"],
                        framework=fw,
                        seed=int(seed),
                        time_budget_s=self.time_budget_s,
                        n_jobs=n_jobs_per_worker,
                        mode=self.tabularaml_mode if fw == "tabularaml" else "medium",
                        wandb_enabled=self.wandb_enabled,
                        wandb_project=self.wandb_project,
                        wandb_entity=self.wandb_entity,
                    ))
        return specs

    def _run_one_subprocess(self, spec: RunSpec) -> dict:
        env = os.environ.copy()
        # Keep numerical libraries from oversubscribing cores when n_workers > 1.
        if self.n_workers > 1:
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("LOKY_MAX_CPU_COUNT", "1")

        with tempfile.NamedTemporaryFile(
            prefix="bench_", suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as fh:
            out_path = fh.name

        cmd = [
            sys.executable, "-m", "tabularaml.benchmarks.feature_gen._worker",
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
            # Prefer the JSON written to --out; fall back to parsing the last stdout line.
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
                # Prefer the Python traceback (before wandb/tqdm noise floods stderr).
                tb_start = stderr.find("Traceback (most recent call last)")
                if tb_start != -1:
                    error_snippet = stderr[tb_start:][:1200]
                else:
                    error_snippet = stderr[-800:]
                row = {
                    "dataset_id": spec.dataset_id, "task": spec.task,
                    "framework": spec.framework, "seed": spec.seed,
                    "time_budget_s": spec.time_budget_s,
                    "status": "crash",
                    "error_msg": error_snippet,
                    "wall_time_total": time.time() - t0,
                }
        except subprocess.TimeoutExpired:
            row = {
                "dataset_id": spec.dataset_id, "task": spec.task,
                "framework": spec.framework, "seed": spec.seed,
                "time_budget_s": spec.time_budget_s,
                "status": "timeout",
                "error_msg": f"wall-clock exceeded {hard_cap}s",
                "wall_time_total": hard_cap,
            }
        finally:
            try:
                os.unlink(out_path)
            except Exception:
                pass
        return row

    def _attach_pct_improvement(self, row: dict, nofe_lookup: dict) -> dict:
        key = (int(row["dataset_id"]), int(row["seed"]))
        nofe_score = nofe_lookup.get(key)
        if nofe_score is not None and row.get("score_holdout") is not None:
            row["score_nofe_same_seed"] = float(nofe_score["score_holdout"])
            denom = abs(row["score_nofe_same_seed"])
            if denom > 0:
                raw = (row["score_holdout"] - row["score_nofe_same_seed"]) / denom
                gib = bool(nofe_score.get("scorer_greater_is_better", True))
                row["pct_improvement"] = float(raw if gib else -raw)
            else:
                row["pct_improvement"] = 0.0
        else:
            row["score_nofe_same_seed"] = None
            row["pct_improvement"] = None
        return row

    def run(self) -> None:
        # Pull prior state from W&B (no-op if artifact doesn't exist yet or
        # wandb is disabled). Must happen BEFORE build_run_plan so resume logic
        # sees the rehydrated master.csv.
        self.sync_from_wandb()

        specs = self.build_run_plan()
        if not specs:
            print("No runs scheduled (all rows already complete).")
            # Still push a final artifact so the latest master.csv is on W&B.
            with OrchestratorRun(
                project=self.wandb_project, entity=self.wandb_entity,
                artifact_name=self.artifact_name,
                enabled=self.wandb_enabled and self.artifact_sync,
            ) as orch:
                if self.master_csv.exists():
                    orch.push(self._sync_paths(), force=True)
            return

        print(f"Scheduled {len(specs)} runs "
              f"({len({(s.dataset_id, s.seed) for s in specs})} unique dataset×seed)")

        # Build nofe_lookup incrementally from master.csv as runs complete.
        master = _load_master(self.master_csv)
        nofe_lookup: dict = {}
        if len(master):
            nofe_rows = master[(master["framework"] == "nofe") & (master["status"] == "ok")]
            for r in nofe_rows.itertuples(index=False):
                nofe_lookup[(int(r.dataset_id), int(r.seed))] = {
                    "score_holdout": float(r.score_holdout),
                    "scorer_greater_is_better": bool(getattr(r, "scorer_greater_is_better", True)),
                }

        with OrchestratorRun(
            project=self.wandb_project, entity=self.wandb_entity,
            artifact_name=self.artifact_name,
            enabled=self.wandb_enabled and self.artifact_sync,
        ) as orch:
            n_done = 0
            try:
                with tqdm(total=len(specs), desc="benchmark") as pbar:
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
                # Always push one final artifact version so the last rows make it
                # to W&B even if the loop exited via KeyboardInterrupt / crash.
                if self.master_csv.exists():
                    orch.push(self._sync_paths(), force=True)

    def _sync_paths(self) -> list[Path]:
        """Files to include in every artifact push."""
        paths = []
        if self.master_csv.exists():
            paths.append(self.master_csv)
        if self.raw_dir.exists():
            paths.append(self.raw_dir)
        return paths

    def _finalize_row(self, row: dict, nofe_lookup: dict, orch=None) -> None:
        if row.get("framework") == "nofe" and row.get("status") == "ok":
            nofe_lookup[(int(row["dataset_id"]), int(row["seed"]))] = {
                "score_holdout": float(row["score_holdout"]),
                "scorer_greater_is_better": bool(row.get("scorer_greater_is_better", True)),
            }
        row = self._attach_pct_improvement(row, nofe_lookup)
        _append_row(self.master_csv, row, RESULT_COLUMNS)
        fw_csv = self.raw_dir / f"{row['framework']}.csv"
        _append_row(fw_csv, row, RESULT_COLUMNS)
        if orch is not None:
            orch.append_result(row)
