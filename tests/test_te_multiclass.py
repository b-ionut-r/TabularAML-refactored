import numpy as np
import pandas as pd

from tabularaml.generate.forge import TE_SMOOTHING, BinnedPairTE, TargetEnc


def _old_stats(k, Y):
    frame = pd.DataFrame(Y)
    frame["_k"] = k
    g = frame.groupby("_k")
    return g.sum(), g.size(), Y.mean(axis=0)


def _old_apply(k, sums, counts, prior, n_out):
    s = sums.reindex(k).to_numpy()
    c = counts.reindex(k).to_numpy(dtype=float)[:, None]
    s = np.where(np.isnan(s), 0.0, s)
    c = np.where(np.isnan(c), 0.0, c)
    out = (s + TE_SMOOTHING * prior) / (c + TE_SMOOTHING)
    return out[:, 0] if n_out == 1 else out


def _old_oof(sp, df, y, folds):
    k = sp._fit_codes(df)
    y = np.asarray(y, dtype=float)
    Y = np.eye(sp.n_classes)[y.astype(int)] if sp.n_out > 1 else y[:, None]
    out = np.zeros((len(df), sp.n_out))
    for tr, va in folds:
        r = _old_apply(k[va], *_old_stats(k[tr], Y[tr]), sp.n_out)
        out[va] = r if r.ndim == 2 else r[:, None]
    full = _old_stats(k, Y)
    return (out[:, 0] if sp.n_out == 1 else out), full


def test_target_encodings_unchanged():
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame({"a": rng.integers(0, 300, n), "b": rng.choice(list("pqrstu"), n),
                       "x": rng.normal(size=n), "z": np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n))})
    new = pd.DataFrame({"a": rng.integers(0, 400, 900), "b": rng.choice(list("pqrstuv"), 900),
                        "x": rng.normal(size=900), "z": rng.normal(size=900)})
    folds = [(np.setdiff1d(np.arange(n), va), va) for va in np.array_split(rng.permutation(n), 5)]
    for n_classes, y in ((0, rng.normal(size=n)), (2, rng.integers(0, 2, n)), (7, rng.integers(0, 7, n)),
                         (100, rng.integers(0, 100, n))):
        for sp in (TargetEnc(["a", "b"], n_classes), TargetEnc(["b"], n_classes), BinnedPairTE("x", "z", n_classes)):
            want, (sums, counts, prior) = _old_oof(sp, df, y, folds)
            got = sp.fit_transform_oof(df, y, None, folds)
            np.testing.assert_array_equal(got, want)
            np.testing.assert_array_equal(sp.sums_.to_numpy(), sums.to_numpy())
            np.testing.assert_array_equal(sp.sums_.index.to_numpy(), sums.index.to_numpy())
            np.testing.assert_array_equal(sp.counts_.to_numpy(), counts.to_numpy())
            np.testing.assert_array_equal(sp.prior_, prior)
            np.testing.assert_array_equal(sp.transform(new, None),
                                          _old_apply(sp._codes(new), sums, counts, prior, sp.n_out))
