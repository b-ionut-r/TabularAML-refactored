import numpy as np
import pandas as pd
import pytest

from tabularaml.preprocessing.encoders import CategoricalEncoder


def _data(n=400, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "cat": rng.choice(["a", "b", "c", "d"], n),
        "num": rng.randn(n),
    })
    # Target correlated with category so encodings carry signal.
    y = pd.Series((X["cat"].isin(["a", "b"])).astype(int).values)
    return X, y


@pytest.mark.parametrize("strategy", ["mean", "smoothed", "catboost"])
def test_strategy_produces_stable_output_columns(strategy):
    X, y = _data()
    enc = CategoricalEncoder(
        target_enc_cols=["cat"], count_enc_cols=["cat"], freq_enc_cols=["cat"],
        target_encoding_strategy=strategy,
    )
    out = enc.fit(X, y).transform(X)
    # Output column names are identical across strategies.
    for col in ("cat_target", "cat_count", "cat_freq"):
        assert col in out.columns
    assert out["cat_target"].notna().all()


def test_smoothing_shrinks_toward_prior():
    X, y = _data()
    plain = CategoricalEncoder(target_enc_cols=["cat"], target_encoding_strategy="mean")
    smooth = CategoricalEncoder(target_enc_cols=["cat"], target_encoding_strategy="smoothed",
                                te_smoothing=50.0)
    plain_out = plain.fit(X, y).transform(X)["cat_target"]
    smooth_out = smooth.fit(X, y).transform(X)["cat_target"]
    prior = y.mean()
    # The overall mean of mean-target-encoding equals the prior by construction, so
    # the meaningful effect of smoothing is per-category shrinkage toward the prior:
    # encoded values spread less and sit nearer the prior at the extremes.
    assert smooth_out.std() < plain_out.std()
    assert (smooth_out - prior).abs().max() < (plain_out - prior).abs().max()


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        CategoricalEncoder(target_enc_cols=["cat"], target_encoding_strategy="bogus")


def test_catboost_multiclass_emits_k_minus_one_columns():
    rng = np.random.RandomState(1)
    n = 400
    X = pd.DataFrame({"cat": rng.choice(["a", "b", "c"], n)})
    y = pd.Series(rng.choice([0, 1, 2], n))
    enc = CategoricalEncoder(target_enc_cols=["cat"], target_encoding_strategy="catboost")
    out = enc.fit(X, y).transform(X)
    target_cols = [c for c in out.columns if c.startswith("cat_target")]
    assert len(target_cols) == 2  # K-1 columns for 3 classes
