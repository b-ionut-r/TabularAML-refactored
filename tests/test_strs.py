import tracemalloc

import numpy as np
import pandas as pd

from tabularaml.generate.forecast import _is_date
from tabularaml.generate.relational import Child, RelatedTables
from tabularaml.generate.strs import as_text


def _long_text_category(n_levels=3000, width=60_000, n=6000):
    # Every level is long text: astype(str) on this categorical asks numpy for
    # n_levels * width * 4 bytes (~0.7 GB here; MCTS's rules column needed 118 GiB).
    rng = np.random.default_rng(0)
    base = "x" * width
    levels = [base + str(i) for i in range(n_levels)]
    return pd.Series(pd.Categorical(np.array(levels, dtype=object)[rng.integers(0, n_levels, n)], categories=levels))


def test_as_text_matches_astype_str_without_fixed_width_copies():
    s = pd.Series(["a", None, "b"]).astype("category")
    assert list(as_text(s)) == ["a", "nan", "b"]
    s = _long_text_category()
    out = as_text(s)
    assert out.dtype == object and out.iloc[0] == str(s.iloc[0])
    assert list(as_text(pd.Series([1.5, np.nan]))) == ["1.5", "nan"]
    assert list(as_text(pd.Series(["x", None], dtype=object))) == ["x", "nan"]
    assert list(as_text(pd.Series([1, 2]).astype("category"))) == ["1", "2"]


def test_long_text_columns_go_through_date_check_and_child_aggregates():
    s = _long_text_category()
    tracemalloc.start()
    assert not _is_date(s)
    key = np.arange(len(s)) % 500
    child = pd.DataFrame({"id": key, "text": s, "v": np.arange(len(s), dtype=float)})
    F = RelatedTables([Child("c", child, key="id")]).features()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert len(F) == 500
    assert peak < 1.5e9  # fixed-width copies of the levels took about 4.5 GB
