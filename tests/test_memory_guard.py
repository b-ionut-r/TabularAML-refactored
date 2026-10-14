import numpy as np
import pandas as pd
import pytest

import tabularaml.generate.relational as R
from tabularaml.generate.returns import return_features


def _log(n=150_000, seed=0, categorical_key=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"sid": rng.integers(0, 3000, n), "t": rng.permutation(n).astype(float),
                       "ev": rng.choice(list("abcdefghij"), n), "room": rng.choice(["r1", "r2", "r3"], n),
                       "lvl": rng.integers(0, 4, n), "x": rng.normal(size=n), "x_b": rng.normal(size=n)})
    if categorical_key:
        # every parent a category, some with no rows (the main table's loans without history)
        df["sid"] = pd.Categorical(df["sid"], categories=np.arange(3100))
    return df


@pytest.mark.parametrize("categorical_key", [False, True])
def test_key_range_aggregation_equals_one_pass(monkeypatch, categorical_key):
    ch = R.Child("log", _log(categorical_key=categorical_key), key="sid", time="t")
    whole = R.RelatedTables([ch]).features()
    monkeypatch.setattr(R, "available_bytes", lambda: 20_000_000)   # the guard splits the keys into ranges
    part = R.RelatedTables([ch]).features()
    assert list(part.columns) == list(whole.columns)
    assert list(part.index) == list(whole.index)
    pd.testing.assert_frame_equal(part, whole)


def test_price_paths_ignore_equals_drop():
    rng = np.random.default_rng(0)
    k = np.repeat(rng.permutation(100), 50)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(k))))
    df = pd.DataFrame({"id": k, "seq": np.tile(np.arange(50), 100), "bid_price": p, "ask_price": p * 1.0005,
                       "bid_size": rng.integers(1, 99, len(k)), "ask_size": rng.integers(1, 99, len(k)),
                       "parent_id": rng.integers(0, 9, len(k))}).sample(frac=1, random_state=0)
    for time in ("seq", None):
        a = return_features(df.drop(columns=["parent_id"]), "id", "book", time)
        b = return_features(df, "id", "book", time, ignore=["parent_id"])
        pd.testing.assert_frame_equal(a, b)
        assert a.shape[1] > 0 or time is None


def test_key_ranges_that_cannot_fit_raise_a_clear_error(monkeypatch):
    ch = R.Child("log", _log(), key="sid", time="t")
    monkeypatch.setattr(R, "available_bytes", lambda: 200_000)   # not even a small range fits
    with pytest.raises(MemoryError, match="related table log"):
        R.RelatedTables([ch]).features()
