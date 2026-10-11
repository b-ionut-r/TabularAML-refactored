import numpy as np
import pandas as pd

from tabularaml.generate.rowstats import column_kinds, row_stat_features


def test_kinds_and_sparse_row_statistics():
    rng = np.random.default_rng(0)
    n = 300
    amounts = pd.DataFrame(np.where(rng.random((n, 30)) < 0.9, 0.0, rng.lognormal(10, 2, (n, 30))),
                           columns=[f"{i:06x}a1" for i in range(30)])
    X = pd.concat([amounts, pd.DataFrame({"age": rng.integers(18, 90, n), "x": rng.normal(size=n)})], axis=1)
    K = column_kinds(X)
    assert set(K["anon_sparse_pos"]) == set(amounts.columns) and not any("age" in v for v in K.values())
    F, G, groups = row_stat_features(X.iloc[:200], X.iloc[200:])
    assert groups == {"anon_sparse_pos": 30} and len(G) == 100
    v = amounts.iloc[3].to_numpy()
    assert F.loc[3, "rows_anon_sparse_pos__nonzero"] == (v != 0).sum()
    if (v != 0).any():
        assert np.isclose(F.loc[3, "rows_anon_sparse_pos__max"], np.log1p(v.max()), rtol=1e-5)


def test_off_on_narrow_tables():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(100, 6)))
    F, G, groups = row_stat_features(X, X)
    assert groups == {} and F.shape[1] == 0


def test_name_families_and_named_columns():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 12)), columns=[f"var_{i}" for i in range(12)])
    X = pd.concat([X, pd.DataFrame(rng.lognormal(size=(200, 12)), columns=[f"price_{w}" for w in "abcdefghijkl"])], axis=1)
    assert column_kinds(X) == {"var_dense_signed": [f"var_{i}" for i in range(12)]}
