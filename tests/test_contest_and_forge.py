import numpy as np
import pandas as pd
import pytest

from tabularaml.contest import ContestSolver, get_metric, hill_climb
from tabularaml.generate.forge import FeatureForge, column_families


def _frame(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.uniform(1, 10, n),
        "b": rng.uniform(1, 10, n),
        "c": rng.normal(size=n),
        "city": rng.choice(list("ABCDEFGH"), n),
        "kind": rng.choice(["x", "y", None], n),
    })
    return X, rng


def test_hill_climb_weights_sum_to_one_and_beat_singles():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    oof = {"m1": y + rng.normal(scale=0.5, size=500), "m2": y + rng.normal(scale=0.5, size=500)}
    m = get_metric("rmse")
    w = hill_climb(oof, y, m)
    assert abs(sum(w.values()) - 1) < 1e-9
    blend = sum(w[k] * oof[k] for k in w)
    assert m(y, blend) <= min(m(y, p) for p in oof.values())


@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_solver_shapes(task):
    X, rng = _frame(600)
    if task == "regression":
        y = X.a * X.b + rng.normal(size=len(X))
    elif task == "binary":
        y = (X.a > 5).astype(int)
    else:
        y = pd.Series(np.digitize(X.a, [4, 7]))
    s = ContestSolver(task=task, models=["lgbm"], n_folds=3, n_jobs=1, verbose=False)
    s.fit(X.iloc[:500], y.iloc[:500], X.iloc[500:])
    p = s.predict_proba(X.iloc[500:])
    assert np.allclose(p, s.test_ensemble_)
    if task == "multiclass":
        assert p.shape == (100, 3)
    else:
        assert p.shape == (100,)
    assert len(s.predict(X.iloc[500:])) == 100


def test_column_families():
    fam = column_families(["Soil_1", "Soil_2", "Soil_3", "x", "px10", "px11"])
    assert fam == {"Soil": ["Soil_1", "Soil_2", "Soil_3"]}


def test_forge_finds_ratio_and_transforms_consistently():
    X, rng = _frame(3000)
    y = np.log(X.a / X.b) * 3 + rng.normal(scale=0.1, size=len(X))
    f = FeatureForge(task="regression", time_budget=60, n_rounds=1, n_jobs=1, verbose=False).fit(X, y)
    assert f.gate_passed_
    assert any(("a" in c and "b" in c) for c in f.new_columns_)
    A, B = f.transform_train(X), f.transform(X)
    assert list(A.columns) == list(B.columns) == list(X.columns) + f.new_columns_
    assert len(f.transform(X.iloc[:10])) == 10


def test_forge_target_encoding_is_out_of_fold():
    X, rng = _frame(3000)
    effect = X.city.map({k: i for i, k in enumerate("ABCDEFGH")}).astype(float)
    y = (effect + rng.normal(size=len(X)) > 3.5).astype(int)
    f = FeatureForge(task="binary", time_budget=60, n_jobs=1, verbose=False, gate_frac=0).fit(X, y)
    te_cols = [c for c in f.new_columns_ if c.startswith("te__")]
    for c in te_cols:
        # Training-row encodings are out-of-fold, so they differ from full-data encodings.
        assert not np.allclose(f.transform_train(X)[c], f.transform(X)[c])


def test_forge_adds_nothing_on_pure_noise_target():
    X, rng = _frame(3000)
    y = rng.normal(size=len(X))
    f = FeatureForge(task="regression", time_budget=60, n_jobs=1, verbose=False).fit(X, y)
    assert f.new_columns_ == []
