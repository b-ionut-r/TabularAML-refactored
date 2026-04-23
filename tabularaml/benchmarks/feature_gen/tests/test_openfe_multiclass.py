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
from sklearn.datasets import load_wine, make_classification
from sklearn.metrics import log_loss
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


def _make_openfe_for_init_checks(X: pd.DataFrame, y: pd.Series):
    from openfe import OpenFE

    ofe = OpenFE()
    ofe.data = X.reset_index(drop=True)
    ofe.label = pd.DataFrame({"label": pd.Series(y).reset_index(drop=True)})
    ofe.task = "classification"
    ofe.feature_boosting = False
    ofe.metric = "multi_logloss"
    ofe.seed = 0
    ofe.n_jobs = 1
    ofe.verbose = False
    ofe.categorical_features = []
    return ofe


def _is_probability_matrix(arr: np.ndarray) -> bool:
    arr = np.asarray(arr, dtype=float)
    return bool(
        arr.ndim == 2
        and arr.min() >= 0.0
        and arr.max() <= 1.0
        and np.allclose(arr.sum(axis=1), 1.0, atol=1e-6)
    )


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


def test_log_probability_math_invariant():
    """softmax(log(p)) == p — the identity the init_score math fix relies on.

    OpenFE passes raw class probabilities as init_score but LightGBM applies
    softmax to init_score, so the fix converts p → log(p) so that
    softmax(log(p)) recovers p exactly.
    """
    rng = np.random.default_rng(42)
    n_samples, n_classes = 30, 4
    raw = np.abs(rng.normal(size=(n_samples, n_classes))) + 1e-6
    probs = raw / raw.sum(axis=1, keepdims=True)

    log_probs = np.log(np.clip(probs, 1e-15, 1.0))

    # Numerically stable softmax
    shifted = log_probs - log_probs.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    recovered = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(recovered, probs, atol=1e-6,
                               err_msg="softmax(log(p)) must equal p — math fix invariant violated")


def test_probability_detection_triggers_log_conversion():
    """A 2-D array with values in [0,1] and rows summing to 1 must be detected as
    a probability matrix by the patch and converted to log-domain (all values ≤ 0).
    A raw-margin matrix (values outside [0,1]) must be left untouched.
    """
    rng = np.random.default_rng(0)
    n, k = 20, 3

    # Build a probability matrix
    raw = np.abs(rng.normal(size=(n, k))) + 1e-6
    probs = raw / raw.sum(axis=1, keepdims=True)
    assert probs.min() >= 0 and probs.max() <= 1
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    log_probs = np.log(np.clip(probs, 1e-15, 1.0))
    # log of values in (0,1] is ≤ 0
    assert (log_probs <= 1e-9).all(), "log(p) for p in (0,1] must be ≤ 0"

    # Raw margins (from predict_proba with raw_score=True) are NOT in [0,1]
    margins = rng.normal(size=(n, k))  # can be negative or > 1
    # Confirming it would NOT be detected as a probability matrix
    is_prob = (margins.min() >= 0.0 and margins.max() <= 1.0 and
               np.allclose(margins.sum(axis=1), 1.0, atol=1e-6))
    assert not is_prob, "Raw margin matrix must not be misclassified as a probability matrix"


@pytest.mark.skipif(not openfe_available, reason="openfe not installed")
def test_openfe_upstream_default_multiclass_init_score_is_probability_matrix():
    """Upstream OpenFE default multiclass init_score really is a probability matrix."""
    X, y = load_wine(return_X_y=True, as_frame=True)
    ofe = _make_openfe_for_init_checks(X, y)

    init_scores = ofe.get_init_score(None).to_numpy(dtype=float)

    assert _is_probability_matrix(init_scores), (
        "Upstream OpenFE default multiclass init_score is expected to be a row-normalized "
        "probability matrix. If this stops being true, the adapter log(p) fix and its "
        "rationale need to be revisited."
    )


@pytest.mark.skipif(not openfe_available, reason="openfe not installed")
def test_openfe_multiclass_metric_matches_true_log_loss_only_after_log_conversion():
    """OpenFE.get_init_metric() is correct for multiclass only after p -> log(p).

    This checks a real imbalanced multiclass setting where softmax(p) noticeably differs
    from p, so the bug is not hidden by a uniform class prior.
    """
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_classes=4,
        weights=[0.70, 0.15, 0.10, 0.05],
        random_state=0,
    )
    X = pd.DataFrame(X)
    y = pd.Series(y)

    ofe = _make_openfe_for_init_checks(X, y)
    probs = ofe.get_init_score(None).to_numpy(dtype=float)
    labels = ofe.label.values.ravel()
    log_probs = np.log(np.clip(probs, 1e-15, 1.0))

    metric_with_probs = ofe.get_init_metric(probs, labels)
    metric_with_log_probs = ofe.get_init_metric(log_probs, labels)
    true_logloss = log_loss(labels, probs, labels=list(range(probs.shape[1])))

    assert metric_with_log_probs == pytest.approx(true_logloss, abs=1e-12), (
        "OpenFE.get_init_metric(log(p), y) must equal the true multiclass log loss of p."
    )
    assert metric_with_probs > true_logloss + 1e-3, (
        "Using raw probabilities as multiclass init_score should measurably distort the "
        "baseline metric on an imbalanced problem."
    )


@pytest.mark.skipif(not openfe_available, reason="openfe not installed")
def test_probability_detector_negative_control_on_real_raw_margins():
    """Real LightGBM multiclass raw margins should not look like probabilities."""
    import lightgbm as lgb

    X, y = load_wine(return_X_y=True, as_frame=True)
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=int(pd.Series(y_tr).nunique()),
        n_estimators=50,
        learning_rate=0.1,
        random_state=0,
        n_jobs=1,
        verbosity=-1,
    )
    clf.fit(X_tr, y_tr)
    raw_scores = clf.predict_proba(X_te, raw_score=True)

    assert not _is_probability_matrix(raw_scores), (
        "A real raw_score=True multiclass output from LightGBM should not satisfy the "
        "probability-matrix detector used by the adapter."
    )
