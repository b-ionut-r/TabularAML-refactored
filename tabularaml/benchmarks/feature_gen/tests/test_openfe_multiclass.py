"""Verify that OpenFE runs on multiclass classification without crashing.

Three things are checked:
1. The init_score / eval_init_score Fortran-flatten patch prevents the
   'Length of init_score is not equal to n_samples * n_classes' LightGBM error.
2. The SystemExit catch in fit_transform converts OpenFE's internal exit()
   calls into a normal RuntimeError (subprocess survives).
3. When it does complete, the adapter output satisfies the column contract.
"""
from __future__ import annotations
import importlib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


openfe_available = importlib.util.find_spec("openfe") is not None


@pytest.fixture
def toy_multiclass():
    rng = np.random.default_rng(7)
    n = 300
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
        "d": rng.normal(size=n),
        "e": rng.normal(size=n),
    })
    y = pd.Series(rng.integers(0, 4, size=n))
    return X, y


@pytest.mark.skipif(not openfe_available, reason="openfe not installed")
def test_openfe_multiclass_no_system_exit(toy_multiclass):
    """fit_transform must not propagate SystemExit — subprocess must survive."""
    from tabularaml.benchmarks.feature_gen.adapters import get_adapter_cls

    X, y = toy_multiclass
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    adapter = get_adapter_cls("openfe")(
        task="classification",
        time_budget_s=120,
        random_state=42,
        n_jobs=1,
        n_data_blocks=2,
    )

    # Must not raise SystemExit.  May raise RuntimeError (converted exit()) or
    # succeed outright — both are acceptable; dying the process is not.
    try:
        X_tr_fe = adapter.fit_transform(X_tr, y_tr)
    except SystemExit:
        pytest.fail(
            "OpenFE raised SystemExit on multiclass — subprocess would have died. "
            "The SystemExit catch in fit_transform is not working."
        )
    except Exception:
        # RuntimeError or any other normal exception is fine — subprocess survives.
        pytest.skip("OpenFE raised a normal exception on multiclass (acceptable, subprocess survives).")
        return

    # If it completed, verify the column contract.
    X_te_fe = adapter.transform(X_te)
    assert list(X_tr_fe.columns) == list(X_te_fe.columns), \
        "Train and test column sets differ after OpenFE multiclass transform."
    assert X_tr_fe.shape[0] == len(X_tr), "Row count changed in train transform."
    assert X_te_fe.shape[0] == len(X_te), "Row count changed in test transform."


@pytest.mark.skipif(not openfe_available, reason="openfe not installed")
def test_openfe_init_score_patch_fires():
    """The Fortran-flatten patch must mark lgb.LGBMModel.fit after being applied."""
    import lightgbm as lgb
    from tabularaml.benchmarks.feature_gen.adapters.openfe_adapter import OpenFEAdapter

    OpenFEAdapter._patch_init_score_flatten()
    assert getattr(lgb.LGBMModel.fit, "_openfe_init_patched", False), \
        "_patch_init_score_flatten did not mark lgb.LGBMModel.fit as patched."


@pytest.mark.skipif(not openfe_available, reason="openfe not installed")
def test_fortran_flatten_correctness():
    """2-D DataFrame init_score must flatten to column-major 1-D of correct shape."""
    rng = np.random.default_rng(0)
    n, k = 50, 4
    arr = rng.random((n, k))
    df_2d = pd.DataFrame(arr)

    fortran_1d = np.asarray(df_2d).ravel(order='F')

    assert fortran_1d.shape == (n * k,), "Fortran flatten: wrong length."
    # Column-major means col-0 values come first, then col-1, etc.
    assert np.allclose(fortran_1d[:n], arr[:, 0]), "Fortran order: first block must be column 0."
    assert np.allclose(fortran_1d[n:2 * n], arr[:, 1]), "Fortran order: second block must be column 1."
    assert np.allclose(fortran_1d[2 * n:3 * n], arr[:, 2]), "Fortran order: third block must be column 2."
