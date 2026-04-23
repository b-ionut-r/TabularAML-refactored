"""Adapter wrapping OpenFE (ICML 2023).

Note on known leakage: upstream OpenFE's `transform()` concatenates train and
test before computing aggregates. We deliberately use the upstream package as
distributed because the benchmark's purpose is to measure what a grant reader
would reproduce with `pip install openfe`. A companion unit test
(test_openfe_leakage_probe.py) flags the leakage so the final report can
annotate it honestly.
"""
from __future__ import annotations
import os
import tempfile
import warnings
from typing import Literal, Optional
import numpy as np
import pandas as pd

from .base import FEFrameworkAdapter


class _OpenFEUpstreamBugError(RuntimeError):
    """Raised for known upstream OpenFE failures that should not kill the worker."""


def _openfe_worker_mse_patch() -> None:
    """ProcessPoolExecutor initializer: re-apply the sklearn MSE compat patch.

    On Windows, each spawned worker process re-imports everything fresh.
    This function runs inside each worker before any task executes, patching
    both sklearn.metrics and openfe.openfe's own module-level binding.
    Must be a module-level function so it is picklable by the spawn pickler.
    """
    import numpy as _np
    import sklearn.metrics as _sm

    if getattr(_sm.mean_squared_error, "_openfe_patched", False):
        return

    _orig = _sm.mean_squared_error

    def _w_mse(y_true, y_pred, *, sample_weight=None,
               multioutput="uniform_average", squared=True):
        r = _orig(y_true, y_pred, sample_weight=sample_weight,
                  multioutput=multioutput)
        return r if squared else _np.sqrt(r)

    _w_mse._openfe_patched = True
    _sm.mean_squared_error = _w_mse
    try:
        import openfe.openfe as _o
        _o.mean_squared_error = _w_mse
    except (ImportError, AttributeError):
        pass


class OpenFEAdapter(FEFrameworkAdapter):
    name = "openfe"
    version = "upstream-0.0.12"
    # supports_multiclass is True (default). OpenFE multiclass has two bugs fixed
    # by _patch_init_score_flatten:
    #   (1) Shape crash — init_score is a DataFrame; LightGBM 4+ requires 1-D Fortran array.
    #       This is crash prevention only; without it OpenFE dies, producing no result.
    #   (2) Math fix — upstream passes raw class probabilities as init_score, but LightGBM
    #       applies softmax again, destroying the class prior. Fix: log(p) so softmax(log(p))=p.
    #       NOTE: this is a performance enhancement, not crash prevention. It gives OpenFE
    #       better multiclass scores than `pip install openfe` would produce out-of-the-box.
    #       Rationale: we want to measure the algorithm's quality, not a known implementation
    #       defect. The companion test (test_openfe_leakage_probe.py) documents transform
    #       leakage; this class-level comment documents the math fix.
    # OpenFE also calls exit() on internal LightGBM errors; the SystemExit catch in
    # fit_transform converts those to RuntimeError so the worker subprocess survives.

    def __init__(
        self,
        task: Literal["regression", "classification"],
        time_budget_s: int,
        random_state: int,
        n_jobs: int = -1,
        n_data_blocks: int = 8,
        feature_boosting: bool = False,
        **framework_kwargs,
    ):
        super().__init__(task, time_budget_s, random_state, n_jobs, **framework_kwargs)
        self.n_data_blocks = n_data_blocks
        self.feature_boosting = feature_boosting
        self._ofe = None
        self._features = None
        self._x_train_cache: Optional[pd.DataFrame] = None
        self._train_columns_fe: Optional[list] = None
        self._col_mapping: dict = {}

    def _task_for_openfe(self, y: pd.Series) -> str:
        if self.task == "regression":
            return "regression"
        # OpenFE handles binary + multiclass via label dtype
        return "classification"

    @staticmethod
    def _patch_init_score_flatten() -> None:
        """Patch LightGBM.fit to fix two init_score issues from OpenFE multiclass.

        Issue 1 — shape: OpenFE passes init_score as a pandas DataFrame of shape
        (n_samples, n_classes).  LightGBM 4.0+ requires a 1-D Fortran-order array
        of length n_samples * n_classes.  len(DataFrame) == n_samples, not n*k,
        so LightGBM raises 'Length of init_score != n_samples * n_classes'.

        Issue 2 — math: OpenFE's default multiclass path passes raw class
        probabilities [p0, p1, …] as init_score.  LightGBM applies softmax to
        init_score, so softmax([0.3, 0.5, 0.2]) ≈ uniform — the prior is lost.
        The fix (log(p)) restores the correct prior: softmax(log(p)) == p exactly.
        OpenFE's own check_init_scores() warns about this very problem.
        The feature_boosting path uses predict_proba(raw_score=True) which already
        returns raw margins, so it is left untouched.
        """
        import numpy as np
        import lightgbm as lgb

        if getattr(lgb.LGBMModel.fit, "_openfe_init_patched", False):
            return

        _orig_fit = lgb.LGBMModel.fit

        def _to_fortran_1d(score):
            """Convert a 2-D score array/DataFrame to Fortran-order 1-D; leave 1-D alone.

            Also fixes OpenFE's multiclass math flaw: its default (no feature_boosting)
            path passes raw class probabilities as init_score, but LightGBM expects raw
            margins.  OpenFE's own check_init_scores() warns about this yet its default
            code violates it.  Detection: 2-D array with all values in [0,1] and rows
            summing to 1.  Fix: log(p), so softmax(log(p)) == p — same prior, correct
            representation.  The feature_boosting path already uses predict_proba with
            raw_score=True, so its scores are outside [0,1] and are left untouched.
            """
            if isinstance(score, (list, tuple)):
                try:
                    score = np.column_stack(score)
                except ValueError:
                    score = np.concatenate(
                        [np.asarray(arr).reshape(-1, 1) for arr in score], axis=1
                    )
            else:
                score = np.asarray(score, dtype=float)
            if score.ndim == 2:
                # Detect probability matrix: values in [0,1], rows sum to ~1
                if (score.min() >= 0.0 and score.max() <= 1.0 and
                        np.allclose(score.sum(axis=1), 1.0, atol=1e-6)):
                    score = np.log(np.clip(score, 1e-15, 1.0))
                return score.ravel(order='F')
            if score.ndim == 1:
                return score
            return None  # unexpected shape — drop

        def _fit_with_correct_init_score(self_lgb, X, y, **kwargs):
            if kwargs.get('init_score') is not None:
                result = _to_fortran_1d(kwargs['init_score'])
                if result is None:
                    kwargs.pop('init_score')
                else:
                    kwargs['init_score'] = result

            # eval_init_score is a list (one entry per eval set)
            if kwargs.get('eval_init_score') is not None:
                fixed = []
                for s in kwargs['eval_init_score']:
                    r = _to_fortran_1d(s)
                    fixed.append(r if r is not None else np.array([]))
                kwargs['eval_init_score'] = fixed

            return _orig_fit(self_lgb, X, y, **kwargs)

        _fit_with_correct_init_score._openfe_init_patched = True
        lgb.LGBMModel.fit = _fit_with_correct_init_score

    @staticmethod
    def _patch_sklearn_mse() -> None:
        """Patch sklearn mean_squared_error for squared=False removal (sklearn>=1.4).

        On Windows, ProcessPoolExecutor uses the 'spawn' start method: each worker
        process re-imports everything from scratch, so patching sklearn.metrics in the
        parent is invisible to workers.  The fix injects _openfe_worker_mse_patch (a
        module-level function, therefore picklable) as an initializer into OpenFE's
        ProcessPoolExecutor subclass so every spawned worker re-applies the patch
        before executing tasks.  This preserves true CPU parallelism (no GIL penalty
        from a ThreadPoolExecutor swap) and avoids any timeout bias.
        """
        import sklearn.metrics as sm

        if getattr(sm.mean_squared_error, "_openfe_patched", False):
            return

        _orig_mse = sm.mean_squared_error

        def _compat_mse(
            y_true,
            y_pred,
            *,
            sample_weight=None,
            multioutput="uniform_average",
            squared=True,
        ):
            result = _orig_mse(
                y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
            )
            return result if squared else np.sqrt(result)

        _compat_mse._openfe_patched = True
        sm.mean_squared_error = _compat_mse

        # Patch openfe's own module-level MSE binding (bound at import time via
        # `from sklearn.metrics import mean_squared_error`).
        # Then subclass ProcessPoolExecutor to inject _openfe_worker_mse_patch as an
        # initializer so every spawned worker process also gets the patch applied.
        try:
            import openfe.openfe as _ofe
            from concurrent.futures import ProcessPoolExecutor as _BasePPE

            _ofe.mean_squared_error = _compat_mse

            if not getattr(_ofe, "_executor_patched", False):

                class _PatchedPPE(_BasePPE):
                    def __init__(self, max_workers=None, **kwargs):
                        if not kwargs.get("initializer"):
                            kwargs["initializer"] = _openfe_worker_mse_patch
                        super().__init__(max_workers, **kwargs)

                _ofe.ProcessPoolExecutor = _PatchedPPE
                _ofe._executor_patched = True

        except (ImportError, AttributeError):
            pass

    def _safe_column_names(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Replace column names with safe alphanumeric identifiers (col_0, col_1, ...)."""
        safe_names = [f"col_{i}" for i in range(df.shape[1])]
        mapping = dict(zip(df.columns, safe_names))
        return df.rename(columns=mapping), mapping

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        # Apply patches (init_score flattening and sklearn MSE)
        self._patch_init_score_flatten()
        self._patch_sklearn_mse()

        try:
            return self._fit_transform_inner(X_train, y_train)
        except SystemExit as exc:
            # OpenFE's _evaluate() calls exit() on any internal LightGBM error.
            # Catch it so the worker subprocess survives and can classify the
            # failure as an upstream framework bug instead of a generic crash.
            raise _OpenFEUpstreamBugError(
                f"OpenFE called exit() internally (multiclass LightGBM error): {exc}"
            ) from exc

    def _fit_transform_inner(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from openfe import OpenFE, transform

            self._n_features_before = X_train.shape[1]

            # Reset to contiguous 0-based index (required by OpenFE)
            X_train = X_train.reset_index(drop=True)
            y_train = pd.Series(y_train).reset_index(drop=True)
            self._x_train_cache = X_train.copy()

            # Replace column names with safe identifiers to avoid Feather/LightGBM crashes
            X_train_safe, self._col_mapping = self._safe_column_names(X_train)
            reverse_mapping = {v: k for k, v in self._col_mapping.items()}

            y_df = pd.DataFrame({"_label": y_train.values}, index=X_train_safe.index)

            effective_blocks = min(self.n_data_blocks, max(2, len(X_train_safe) // 100))

            self._ofe = OpenFE()
            # Change to a fresh temp directory so parallel workers don't collide
            os.chdir(tempfile.mkdtemp())
            self._features = self._ofe.fit(
                data=X_train_safe,
                label=y_df,
                task=self._task_for_openfe(y_train),
                n_jobs=self.n_jobs,
                n_data_blocks=effective_blocks,
                feature_boosting=self.feature_boosting,
                seed=self.random_state,
                verbose=False,
            )

            X_train_fe_safe, _ = transform(
                X_train_safe,
                X_train_safe,
                self._features,
                n_jobs=self.n_jobs,
            )

            # Restore original column names where possible (base features)
            # Generated features keep their safe names (they never had special chars anyway)
            new_cols = []
            for col in X_train_fe_safe.columns:
                if col in reverse_mapping:
                    new_cols.append(reverse_mapping[col])
                else:
                    new_cols.append(col)
            X_train_fe_safe.columns = new_cols

            self._train_columns_fe = list(X_train_fe_safe.columns)
            self._n_features_after = X_train_fe_safe.shape[1]
            return X_train_fe_safe

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        if self._features is None or self._x_train_cache is None:
            raise RuntimeError("OpenFEAdapter.transform called before fit_transform")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from openfe import transform

            X_test = X_test.reset_index(drop=True)
            # Apply the same column renaming to test data
            X_test_safe = X_test.rename(columns=self._col_mapping)

            try:
                _, X_test_fe_safe = transform(
                    self._x_train_cache.rename(columns=self._col_mapping),
                    X_test_safe,
                    self._features,
                    n_jobs=self.n_jobs,
                )
            except SystemExit as exc:
                raise _OpenFEUpstreamBugError(
                    f"OpenFE called exit() internally (multiclass LightGBM error): {exc}"
                ) from exc

            reverse_mapping = {v: k for k, v in self._col_mapping.items()}
            new_cols = []
            for col in X_test_fe_safe.columns:
                if col in reverse_mapping:
                    new_cols.append(reverse_mapping[col])
                else:
                    new_cols.append(col)
            X_test_fe_safe.columns = new_cols

        # Enforce same column order as train
        return X_test_fe_safe[self._train_columns_fe]
