"""Child tables too big for memory, aggregated a range of keys at a time.

Optiver's full order book is 160M rows; read whole, its aggregation (each column, same-unit differences and ratios,
five statistics) needs about 4 bytes x 3 copies x (columns + 2 x pairs) per row, far beyond a 15 GB machine. Every
per-key feature of a keyed child table (RelatedTables, price paths, latest-state history) only reads the rows of
one key, so the table can be read from parquet one key range at a time, each range aggregated and dropped: the
result is the same as aggregating it whole. Ranges come from the key column alone, and parquet's row-group
statistics let a sorted file skip what is outside a range.
"""
from __future__ import annotations

import os
from itertools import combinations
from typing import Callable, Iterator, List, Optional

import numpy as np
import pandas as pd


def _dataset(path: str):
    import pyarrow.dataset as ds
    return ds.dataset(path, format="parquet", partitioning="hive" if os.path.isdir(path) else None)


def memory_bytes() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 16 << 30


def _cgroup_dirs() -> List[tuple]:
    """This process's memory cgroup and its parents, innermost first, as (directory, limit file, usage
    file) for cgroup v2 or v1. A container's limit is often set on the process's own cgroup, not on the
    root that /sys/fs/cgroup/memory.max names (here: a 14.3 GB limit two levels down)."""
    out = []
    try:
        lines = open("/proc/self/cgroup").read().splitlines()
    except OSError:
        return out
    for line in lines:
        _, ctrl, path = line.split(":", 2)
        if ctrl == "" and os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
            root, files = "/sys/fs/cgroup", ("memory.max", "memory.current")
        elif "memory" in ctrl.split(","):
            root, files = "/sys/fs/cgroup/memory", ("memory.limit_in_bytes", "memory.usage_in_bytes")
        else:
            continue
        parts = [q for q in path.split("/") if q]
        for k in range(len(parts), -1, -1):
            d = os.path.join(root, *parts[:k])
            if os.path.isdir(d):
                out.append((d,) + files)
    return out


def _read_int(path: str) -> Optional[int]:
    try:
        v = open(path).read().strip()
    except OSError:
        return None
    return int(v) if v.isdigit() and int(v) < 1 << 60 else None


def cgroup_headroom() -> Optional[int]:
    """What this process's memory cgroups (it and every parent with a limit) can still give: the least of
    limit minus usage, usage not counting the file cache, which the kernel reclaims before it kills
    (recently read parquet files otherwise counted as gigabytes in use). None without a limit."""
    best = None
    for d, lim_f, use_f in _cgroup_dirs():
        lim, use = _read_int(os.path.join(d, lim_f)), _read_int(os.path.join(d, use_f))
        if lim is None or use is None:
            continue
        try:
            stat = dict(l.split() for l in open(os.path.join(d, "memory.stat")) if len(l.split()) == 2)
            v1 = "total_cache" in stat
            cache = int(stat["total_cache"]) - int(stat.get("total_shmem", 0)) if v1 else \
                int(stat.get("file", 0)) - int(stat.get("shmem", 0))
            use -= max(cache, 0)
        except (OSError, ValueError, KeyError):
            pass
        best = lim - max(use, 0) if best is None else min(best, lim - max(use, 0))
    return best


def available_bytes() -> int:
    """Memory free for new allocations now: the kernel's MemAvailable, or less when this process's
    memory cgroup has less left (else half the machine)."""
    free = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    free = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    if free is None:
        free = memory_bytes() // 2
    room = cgroup_headroom()
    return free if room is None else max(0, min(free, room))


def aggregation_bytes(path: str, key: str) -> Optional[int]:
    """What aggregating the parquet table whole would take: rows x (numeric columns + 2 x same-unit pairs) x 4 bytes
    x 3 copies. None for files that are not parquet."""
    if not (os.path.isdir(path) or path.endswith(".parquet")):
        return None
    import pyarrow as pa
    d = _dataset(path)
    rows = d.count_rows()
    num = [f.name for f in d.schema if f.name != key and (pa.types.is_integer(f.type) or pa.types.is_floating(f.type))]
    stems = {}
    for c in num:
        stems.setdefault(c.split("_")[0].upper(), []).append(c)
    pairs = sum(len(list(combinations(v[:6], 2))) for k, v in stems.items() if len(v) >= 2 and len(k) >= 2)
    return rows * (len(num) + 2 * min(pairs, 12)) * 4 * 3


def key_ranges(path: str, key: str, n: int) -> List[tuple]:
    """``n`` ranges [lo, hi) of the key's sorted distinct values, about equal in rows."""
    k = _dataset(path).to_table(columns=[key]).column(key).to_numpy(zero_copy_only=False)
    v, cnt = np.unique(k, return_counts=True)
    cum = np.cumsum(cnt)
    cuts = np.searchsorted(cum, np.linspace(0, cum[-1], n + 1)[1:-1])
    bounds = [v[0]] + [v[i] for i in np.unique(cuts) if 0 < i < len(v)]
    return [(bounds[i], bounds[i + 1] if i + 1 < len(bounds) else None) for i in range(len(bounds))]


def read_range(path: str, key: str, lo, hi, columns=None) -> pd.DataFrame:
    import pyarrow.dataset as ds
    f = ds.field(key) >= lo if hi is None else (ds.field(key) >= lo) & (ds.field(key) < hi)
    return _dataset(path).to_table(filter=f, columns=columns).to_pandas()


def chunks(path: str, key: str, budget: int, normalise: Callable[[pd.DataFrame], pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """The table a key range at a time, each range's aggregation within ``budget`` bytes."""
    n = max(2, int(np.ceil(aggregation_bytes(path, key) / budget)))
    for lo, hi in key_ranges(path, key, n):
        yield normalise(read_range(path, key, lo, hi))
