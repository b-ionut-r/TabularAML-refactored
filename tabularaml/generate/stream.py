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
