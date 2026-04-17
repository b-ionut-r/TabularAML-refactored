import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_cls():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "num_a": rng.normal(size=n),
        "num_b": rng.normal(size=n),
        "num_c": rng.normal(size=n),
        "num_d": rng.normal(size=n),
        "num_e": rng.normal(size=n),
        "cat_a": rng.choice(["x", "y", "z"], size=n),
        "cat_b": rng.choice(["p", "q"], size=n),
        "cat_c": rng.choice(["u", "v", "w", "t"], size=n),
    })
    X["cat_a"] = X["cat_a"].astype("category")
    X["cat_b"] = X["cat_b"].astype("category")
    X["cat_c"] = X["cat_c"].astype("category")
    logits = X["num_a"] * 1.3 + X["num_b"] * -0.9 + rng.normal(size=n) * 0.5
    y = (logits > 0).astype(int)
    return X, pd.Series(y)


@pytest.fixture
def toy_reg():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "num_a": rng.normal(size=n),
        "num_b": rng.normal(size=n),
        "num_c": rng.normal(size=n),
        "num_d": rng.normal(size=n),
        "num_e": rng.normal(size=n),
        "cat_a": pd.Categorical(rng.choice(["x", "y", "z"], size=n)),
        "cat_b": pd.Categorical(rng.choice(["p", "q"], size=n)),
    })
    y = X["num_a"] + 0.5 * X["num_b"] - 0.3 * X["num_c"] + rng.normal(size=n) * 0.2
    return X, pd.Series(y)
