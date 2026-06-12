"""Pre-fetch all PMLB datasets used by the targeted `pmlb` suite into the local cache.

Run once before benchmark arms so no run pays (or fails on) download latency:

    PYTHONPATH=<worktree> python scripts/prewarm_pmlb.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabularaml.benchmarks.feature_gen.targeted.registry import get_suite  # noqa: E402


def main() -> int:
    import pmlb

    cache_dir = os.environ.get("PMLB_CACHE_DIR", str(Path.home() / ".cache" / "pmlb"))
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    specs = get_suite("pmlb")
    failures = []
    for spec in specs:
        try:
            df = pmlb.fetch_data(spec.id, local_cache_dir=cache_dir)
            print(f"OK   {spec.id:<22} rows={len(df):>6} cols={df.shape[1]:>4}")
        except Exception as e:  # noqa: BLE001
            failures.append((spec.id, str(e)))
            print(f"FAIL {spec.id:<22} {e}")
    print(f"\n{len(specs) - len(failures)}/{len(specs)} datasets cached at {cache_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
