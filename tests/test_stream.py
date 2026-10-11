import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from tabularaml.generate.relational import Child, RelatedTables
from tabularaml.generate.returns import return_features
from tabularaml.generate.stream import key_ranges, read_range

_spec = importlib.util.spec_from_file_location("cf", Path(__file__).resolve().parents[1] / "scripts" / "contest_features.py")
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)


def _book(n_keys=200, rows=60, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for k in range(n_keys):
        mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005 * (1 + k % 4), rows)))
        parts.append(pd.DataFrame({"row_id": k, "seconds_in_bucket": np.arange(rows) * 5,
                                   "bid_price1": (mid * 0.9998).astype(np.float32), "ask_price1": (mid * 1.0002).astype(np.float32),
                                   "bid_size1": rng.integers(1, 500, rows), "ask_size1": rng.integers(1, 500, rows),
                                   "side": rng.choice(["a", "b", "c"], rows)}))
    return pd.concat(parts, ignore_index=True)


def test_key_ranges_cover_every_key_once(tmp_path):
    df = _book()
    p = tmp_path / "book.parquet"
    df.to_parquet(p)
    r = key_ranges(str(p), "row_id", 7)
    got = pd.concat([read_range(str(p), "row_id", lo, hi) for lo, hi in r])
    assert len(got) == len(df) and got.row_id.nunique() == 200


def test_streamed_equals_whole(tmp_path, monkeypatch):
    df = _book()
    p = tmp_path / "book.parquet"
    df.to_parquet(p)
    # About seven key ranges of ~1,700 rows.
    monkeypatch.setattr(cf, "STREAM_BUDGET_SHARE", cf.aggregation_bytes(str(p), "row_id") / 7 / cf.memory_bytes())
    ch = Child("book", cf.normalise(df.head(1000)), key="row_id", source=str(p))
    S = cf.streamed_features(ch, history=True)
    whole = cf.normalise(df.copy())
    F = RelatedTables([Child("book", whole, key="row_id")]).features()
    R = return_features(whole, "row_id", "book")
    assert len(S) == 2
    assert set(S[0].columns) == set(F.columns)
    pd.testing.assert_frame_equal(S[0].loc[F.index, F.columns], F, check_exact=False, rtol=1e-5)
    pd.testing.assert_frame_equal(S[1].loc[R.index, R.columns], R, check_exact=False, rtol=1e-5)
