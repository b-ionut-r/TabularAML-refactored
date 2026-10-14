import numpy as np
import pandas as pd

from tabularaml.generate.forge import FamilyCount, _map_counts, _value_table


def _cases():
    rng = np.random.default_rng(0)
    f = rng.integers(0, 50, 2000).astype(np.float32) / 4
    f[::7] = np.nan
    f[:3] = [0.0, -0.0, np.inf]
    yield pd.Series(f), pd.Series(np.r_[f[:500], np.float32([1e9, -3.25, np.nan, -0.0])])
    i = rng.integers(-5, 400, 3000)
    yield pd.Series(i), pd.Series(np.r_[i[:100], [10_000, -7]])
    yield pd.Series(i), pd.Series(np.r_[i[:100], [10_000, -7]].astype(float))
    yield pd.Series(rng.normal(size=1000)), pd.Series(rng.normal(size=50))   # all seen once
    yield pd.Series(list("aabbbc") * 5), pd.Series(list("abcz"))


def test_value_table_maps_as_value_counts():
    for fit, new in _cases():
        want = new.map(fit.value_counts()).to_numpy(dtype=float)
        got = _map_counts(new, _value_table(fit))
        np.testing.assert_array_equal(got, want)
        rep = _map_counts(new, _value_table(fit, ones=False))
        np.testing.assert_array_equal(rep > 1, want > 1)


def test_family_count_unchanged():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({f"v{j}": np.round(rng.normal(size=3000), 1 + j % 3).astype(np.float32)
                       for j in range(9)})
    df.iloc[::11, 2] = np.nan
    new = df.sample(700, random_state=0) + np.float32(0.05) * (np.arange(700) % 2)[:, None]
    for kind in ("count", "mask"):
        sp = FamilyCount(list(df.columns), kind, "v").fit(df, None, None)
        old = np.empty((len(new), sp.n_out), dtype=np.float32)
        for j, c in enumerate(df.columns):
            cnt = new[c].map(df[c].value_counts()).to_numpy(dtype=float)
            x = new[c].to_numpy(dtype=float)
            old[:, j] = np.nan_to_num(cnt, nan=0.0) if kind == "count" else np.where(cnt > 1, x, np.nan)
        np.testing.assert_array_equal(sp.transform(new, None), old)
