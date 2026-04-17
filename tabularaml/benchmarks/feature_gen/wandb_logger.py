"""Thin wandb helper. Designed to be safely no-op when wandb isn't installed or
no API key is present — the runner stays functional either way, and all real
analysis is driven off master.csv.

Artifact-based state sync:
    download_results_artifact() pulls the latest `{project}/{name}:latest`
    artifact into a target directory (used by the orchestrator at startup so
    a fresh HF Space clone can resume a partial benchmark).

    OrchestratorRun is a long-lived wandb run that owns artifact uploads —
    distinct from the per-(dataset, framework, seed) runs spawned by workers.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List
import os
import shutil
import time


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


@contextmanager
def wandb_run(
    *,
    project: str,
    entity: Optional[str],
    run_name: str,
    group: str,
    tags: list,
    config: Dict[str, Any],
    enabled: bool = True,
):
    """Context manager that yields a wandb.run or None.

    Handles offline mode when no API key is present; always closes the run on
    exit even if the inner block raises.
    """
    if not enabled or not _wandb_enabled():
        yield None
        return

    import wandb
    if not os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "offline")
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=group,
        tags=tags,
        config=config,
        reinit=True,
        settings=wandb.Settings(start_method="thread"),
    )
    try:
        yield run
    finally:
        try:
            wandb.finish()
        except Exception:
            pass


def log_row(run, row: Dict[str, Any]) -> None:
    """Log a completed result row as wandb metrics (single step)."""
    if run is None:
        return
    import wandb
    # Filter non-primitive values from metrics; store as config/summary instead.
    metrics = {}
    summary = {}
    for k, v in row.items():
        if isinstance(v, (int, float, bool)) or v is None:
            metrics[k] = v
        else:
            summary[k] = str(v)
    wandb.log(metrics)
    for k, v in summary.items():
        try:
            wandb.run.summary[k] = v
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
# Artifact-based state sync: pull on startup, push after every N rows / at end.
# ---------------------------------------------------------------------------

def download_results_artifact(
    *, project: str, entity: Optional[str], artifact_name: str,
    out_dir: Path, alias: str = "latest",
) -> bool:
    """Pull `{entity}/{project}/{artifact_name}:{alias}` into `out_dir`.

    Copies every file in the artifact to out_dir (preserving basenames), so a
    prior master.csv / raw/*.csv snapshot rehydrates on top of an empty dir.

    Returns True if an artifact was downloaded, False if it does not exist yet
    (first run) or wandb is disabled. Never raises on transient errors —
    errors are printed and we fall through so the benchmark can still run.
    """
    if not _wandb_enabled() or not os.environ.get("WANDB_API_KEY"):
        return False
    try:
        import wandb
        api = wandb.Api()
        qualified = (
            f"{entity}/{project}/{artifact_name}:{alias}" if entity
            else f"{project}/{artifact_name}:{alias}"
        )
        artifact = api.artifact(qualified)
    except Exception as e:
        # Most common: artifact doesn't exist yet. That's a clean first-run signal.
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
            rel = src.relative_to(staging)
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    print(f"[wandb] pulled {copied} file(s) from {qualified} into {out_dir}")
    return copied > 0


class OrchestratorRun:
    """Long-lived wandb run owned by the benchmark orchestrator.

    Used only to upload versioned `{artifact_name}` artifacts so the master
    CSV + raw/*.csv survive a fresh HF Space clone. Per-(dataset, framework,
    seed) runs are independent and spawned by workers.
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
        self.enabled = bool(enabled and _wandb_enabled() and os.environ.get("WANDB_API_KEY"))
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
                name=f"orchestrator_{int(time.time())}",
                job_type="orchestrator",
                reinit=True,
                settings=wandb.Settings(start_method="thread"),
            )
        except Exception as e:
            print(f"[wandb] orchestrator init failed; artifact sync disabled: {e}")
            self.enabled = False
        return self

    def push(self, paths: List[Path], *, force: bool = False, min_interval_s: float = 30.0) -> bool:
        """Upload a new versioned artifact. Debounced — no-ops if the last push
        happened within `min_interval_s` unless `force=True`.
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
            self._last_push = now
            return True
        except Exception as e:
            print(f"[wandb] artifact push failed: {e}")
            return False

    def __exit__(self, *exc):
        if self._run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        return False
