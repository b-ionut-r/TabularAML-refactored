"""Tests for BaselineFeatureExpander (datetime decomposition + row stats)."""
import numpy as np
import pandas as pd
import pytest

from tabularaml.generate.expanders import BaselineFeatureExpander


def _frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-03-01", periods=n, freq="7h")
    X = pd.DataFrame({
        "dt_native": dates,
        "dt_string": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "num_a": rng.normal(size=n),
        "num_b": rng.normal(size=n),
        "num_c": rng.normal(size=n),
        "cat_a": pd.Categorical(rng.choice(["x", "y", "z"], size=n)),
        "word": rng.choice(["foo", "bar", "baz"], size=n),  # not parseable
    })
    X.loc[X.index[:7], "num_a"] = np.nan
    return X


def test_detects_native_and_string_datetimes_only():
    X = _frame()
    exp = BaselineFeatureExpander().fit(X)
    assert set(exp.datetime_cols_) == {"dt_native", "dt_string"}
    out = exp.transform(X)
    # raw datetime columns dropped, no datetime64 left for the model
    assert "dt_native" not in out.columns and "dt_string" not in out.columns
    assert not any(pd.api.types.is_datetime64_any_dtype(out[c]) for c in out.columns)
    assert "dt_native_month" in out.columns and "dt_string_dayofweek" in out.columns


def test_numeric_columns_never_parsed():
    X = pd.DataFrame({"maybe_date": [20210101, 20210102, 20210103] * 40,
                      "num_a": np.arange(120.0), "num_b": np.arange(120.0),
                      "num_c": np.arange(120.0)})
    exp = BaselineFeatureExpander().fit(X)
    assert exp.datetime_cols_ == []
    assert "maybe_date" in exp.transform(X).columns


def test_constant_outputs_pruned():
    n = 60
    dates = pd.date_range("2021-01-04", periods=n, freq="D")  # hour constant
    X = pd.DataFrame({"d": dates, "num_a": np.arange(n, dtype=float),
                      "num_b": np.arange(n, dtype=float), "num_c": np.arange(n, dtype=float)})
    exp = BaselineFeatureExpander().fit(X)
    assert "d_hour" not in exp.dt_outputs_["d"]      # constant 0
    assert "d_month" in exp.dt_outputs_["d"]


def test_transform_deterministic_and_batch_independent():
    X = _frame()
    exp = BaselineFeatureExpander().fit(X.iloc[:80])
    full = exp.transform(X)
    head = exp.transform(X.iloc[:30])
    pd.testing.assert_frame_equal(full.iloc[:30], head)


def test_row_stats_exact():
    X = _frame()
    exp = BaselineFeatureExpander(datetime_features=False).fit(X)
    out = exp.transform(X)
    num_block = X[["num_a", "num_b", "num_c"]]
    np.testing.assert_allclose(out["row_mean"], num_block.mean(axis=1))
    np.testing.assert_allclose(out["row_nan_count"], X.isna().sum(axis=1).astype(float))


def test_exclude_cols_respected():
    X = _frame()
    exp = BaselineFeatureExpander(exclude_cols=("dt_native",)).fit(X)
    assert exp.datetime_cols_ == ["dt_string"]
    assert "dt_native" in exp.transform(X).columns


def test_generator_roundtrip_with_datetime():
    rng = np.random.default_rng(1)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="13h")
    X = pd.DataFrame({"when": dates,
                      "a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    # target depends on day-of-week: only reachable through dt decomposition
    y = pd.Series((dates.dayofweek >= 5).astype(float) * 2 + X["a"] + rng.normal(size=n) * 0.2)

    from tabularaml.generate.features import FeatureGenerator
    gen = FeatureGenerator(n_generations=2, n_parents=6, n_children=24, cv=3,
                           log_file=None, use_proxy_evaluation=False,
                           final_selection=False, meta_validation_frac=0.0,
                           early_stopping_iter=10, time_budget=120)
    X_out, _, _, _ = gen.search(X, y)
    assert gen.base_expander is not None
    assert "when_dayofweek" in X_out.columns and "when" not in X_out.columns
    # transform on a fresh frame reproduces the expanded schema
    X_new = gen.transform(X.iloc[:50].copy())
    assert "when_dayofweek" in X_new.columns and "when" not in X_new.columns
