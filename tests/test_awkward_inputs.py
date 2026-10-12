"""Inputs a new contest can bring that once crashed or overran the budget."""
import numpy as np
import pandas as pd

from tabularaml.generate.forge import FeatureForge


def _frame(n=400, seed=0):
    r = np.random.default_rng(seed)
    X = pd.DataFrame({"a": r.normal(size=n), "b": r.integers(0, 20, n), "c": r.choice(list("pqrs"), n)})
    y = (X.a + (X.c == "p") + r.normal(0, 0.5, n) > 0.5).astype(int)
    return X, y


def _fit(X, y):
    U = X.iloc[:50].reset_index(drop=True)
    f = FeatureForge(task="binary", time_budget=5, random_state=0, verbose=False).fit(X, y, X_unlabeled=U)
    A, B = f.transform_train(X), f.transform(U)
    assert len(A) == len(X) and len(B) == len(U)
    assert A.columns.is_unique and all(isinstance(c, str) for c in A.columns)
    return A


def test_integer_column_names():
    X, y = _frame()
    X.columns = [0, 1, "c"]
    A = _fit(X, y)
    assert {"0", "1", "c"} <= set(A.columns)


def test_duplicated_column_names():
    X, y = _frame()
    X["a2"] = X["a"] * 2
    X = X.rename(columns={"a2": "a"})
    A = _fit(X, y)
    assert {"a", "a.1"} <= set(A.columns)

