import numpy as np
import pandas as pd
import pytest

from tabularaml.contest import ContestSolver, get_metric, hill_climb
from tabularaml.generate.forge import (CrossLinearOOF, FeatureForge, GeoPair, column_families,
                                       geo_points)


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


def test_column_families_sorted_by_index():
    fam = column_families(["PAY_0", "PAY_10", "PAY_2", "PAY_3"])
    assert fam == {"PAY": ["PAY_0", "PAY_2", "PAY_3", "PAY_10"]}


def test_geo_points_and_haversine():
    cols = ["Restaurant_latitude", "Restaurant_longitude", "drop_lat", "drop_lng", "x"]
    pts = geo_points(cols)
    assert pts == [("Restaurant_latitude", "Restaurant_longitude"), ("drop_lat", "drop_lng")]
    df = pd.DataFrame({"Restaurant_latitude": [0.0], "Restaurant_longitude": [0.0],
                       "drop_lat": [0.0], "drop_lng": [1.0]})
    km = GeoPair(pts[0], pts[1], "hav").transform(df, None)
    assert abs(km[0] - 111.19) < 0.1


def test_cross_linear_is_out_of_fold_and_learns_pairs():
    from sklearn.model_selection import StratifiedKFold
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({"u": rng.choice(list("abcdefghij"), n), "v": rng.choice(list("klmnopqrst"), n)})
    # The label depends only on the (u, v) pair, not on u or v alone.
    good = {(a, b) for a in "abcdefghij" for b in "klmnopqrst" if rng.random() < 0.5}
    y = np.array([int((a, b) in good) for a, b in zip(X.u, X.v)])
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    pair = CrossLinearOOF(["u", "v"], 2, pairs=True).fit_transform_oof(X, y, None, folds)
    single = CrossLinearOOF(["u", "v"], 2, pairs=False).fit_transform_oof(X, y, None, folds)
    acc = lambda m: ((m > 0) == y).mean()
    assert acc(pair) > 0.95 > 0.7 > acc(single)
    spec = CrossLinearOOF(["u", "v"], 2).fit(X, y, None)
    assert spec.transform(X.iloc[:7], None).shape == (7,)


def test_forge_high_cardinality_recode_is_consistent():
    rng = np.random.default_rng(0)
    n = 6000
    ids = np.array([f"id{i}" for i in range(400)])
    X = pd.DataFrame({"user": rng.choice(ids, n), "x": rng.normal(size=n)})
    eff = dict(zip(ids, rng.normal(size=len(ids))))
    y = (X.user.map(eff) + X.x + rng.normal(scale=0.5, size=n) > 0).astype(int)
    f = FeatureForge(task="binary", time_budget=60, n_rounds=1, n_jobs=1, verbose=False).fit(X, y)
    A, B = f.transform_train(X), f.transform(X)
    assert list(A.columns) == list(B.columns)
    if f.recode_:
        assert A["user"].dtype.kind == "f" and np.array_equal(A["user"], B["user"])
    new = f.transform(pd.DataFrame({"user": ["never_seen"], "x": [0.0]}))
    assert len(new) == 1


def test_linresid_recovers_unexplained_part():
    from tabularaml.generate.forge import LinResid
    rng = np.random.default_rng(0)
    n = 3000
    a, b, c = rng.uniform(1, 5, (3, n))
    extra = rng.normal(0, 1, n)
    X = pd.DataFrame({"a": a, "b": b, "c": c, "w": a + b + c + extra})
    r = LinResid(["a", "b", "c", "w"], "w").fit(X, None, None).transform(X, None)
    assert np.corrcoef(r, extra)[0, 1] > 0.95
    assert np.isfinite(LinResid(["a", "b", "c", "w"], "w", log=True).fit(X, None, None).transform(X, None)).all()


def test_expr_spec_is_canonical_and_evaluates():
    from tabularaml.generate.forge import Expr
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 0.0, 1.0], "c": [1.0, 1.0, 2.0]})
    e1, e2 = Expr(("div", ("add", "a", "b"), "c")), Expr(("div", ("add", "b", "a"), "c"))
    assert e1.name == e2.name and e1.parents == ["a", "b", "c"]
    assert np.allclose(e1.transform(X, None), [3.0, 2.0, 2.0])
    assert np.isnan(Expr(("div", "a", "b")).transform(X, None)[1])


def test_genetic_search_finds_compound_ratio():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame(rng.uniform(1, 5, (n, 6)), columns=list("abcdef"))
    y = X.a * X.b / X.c - X.d * X.e / X.f + rng.normal(0, 0.3, n)
    kw = dict(task="regression", n_rounds=1, max_new_features=10, n_jobs=1, verbose=False, random_state=0)
    base = FeatureForge(**kw).fit(X, y)
    ga = FeatureForge(evolve_time=8, **kw).fit(X, y)
    assert any(c.startswith("gp__") for c in ga.new_columns_)
    assert ga.gate_fe_loss_ < base.gate_fe_loss_


def test_entity_anchor_detection_finds_opening_day():
    rng = np.random.default_rng(0)
    n_ent, n = 3000, 30000
    card = rng.integers(1000, 1300, n_ent)          # many entities share a card number
    opened = rng.integers(0, 400, n_ent)            # hidden account-opening day
    ent = rng.integers(0, n_ent, n)
    day = rng.uniform(400, 580, n)
    X = pd.DataFrame({"T": day * 86400, "card": card[ent], "D1": np.floor(day) - opened[ent],
                      "D3": rng.integers(0, 300, n), "amt": rng.gamma(2, 50, n)})
    f = FeatureForge(task="binary", verbose=False)
    f.cat_cols_ = []
    f.id_cols_ = f._id_columns(X)
    assert "card" in f.id_cols_
    anchors = f._find_anchors(X)
    assert [(t, s, d) for t, s, d, _ in anchors] == [("T", 86400, "D1")]
    f.anchors_ = anchors
    A = f._add_anchors(X.copy())
    assert (A.groupby(ent)[anchors[0][3]].nunique() == 1).all()
