"""Adversarial validation for train/test distribution-shift detection.

Adversarial validation trains a classifier to distinguish training rows from
(unlabeled) test rows. If the classifier cannot tell them apart (ROC-AUC ~= 0.5)
the two sets are drawn from the same distribution. A high AUC signals covariate
shift, and the per-feature importances of that classifier identify *which*
features drift between train and test.

This is a standard Kaggle technique used to (a) prune engineered features that do
not transfer to the test distribution and (b) reweight cross-validation so that
validation folds resemble the test set. Only feature values are used -- the target
is never touched, so this introduces no label leakage.
"""

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


class AdversarialValidator:
    """Detect and quantify train/test distribution shift on a per-feature basis.

    Parameters
    ----------
    estimator : sklearn-style classifier, optional
        Pre-built classifier. If ``None`` a gradient-boosted tree is selected
        automatically (LightGBM > XGBoost > HistGradientBoosting).
    cv : int, default 5
        Number of stratified folds used to produce out-of-fold P(test) estimates.
    random_state : int, default 42
    use_gpu : bool, default False
        Request GPU training for the auto-selected GBM where supported.
    n_jobs : int, default -1
    """

    def __init__(self, estimator=None, cv=5, random_state=42, use_gpu=False, n_jobs=-1):
        self.estimator = estimator
        self.cv = cv
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.n_jobs = n_jobs

        self.auc_ = None
        self.feature_importances_ = {}
        self._drift_scores = {}
        self._oof_test_proba = None
        self.feature_names_ = []

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_estimator(self):
        if self.estimator is not None:
            from sklearn.base import clone
            return clone(self.estimator)

        if _LGBM_AVAILABLE:
            params = dict(n_estimators=200, num_leaves=31, learning_rate=0.05,
                          random_state=self.random_state, n_jobs=self.n_jobs, verbose=-1)
            if self.use_gpu:
                params.update(device="gpu")
            try:
                return LGBMClassifier(**params)
            except Exception:
                return LGBMClassifier(n_estimators=200, random_state=self.random_state,
                                      n_jobs=self.n_jobs, verbose=-1)
        if _XGB_AVAILABLE:
            params = dict(n_estimators=200, max_depth=6, learning_rate=0.05,
                          random_state=self.random_state, n_jobs=self.n_jobs, verbosity=0)
            if self.use_gpu:
                params.update(tree_method="gpu_hist")
            return XGBClassifier(**params)

        return HistGradientBoostingClassifier(max_iter=200, random_state=self.random_state)

    @staticmethod
    def _encode(df):
        """Numeric-encode a frame so any GBM/sklearn model can consume it.

        Categorical / object columns are mapped to integer codes; non-finite
        numeric values are left as NaN (tree models handle them natively).
        """
        out = pd.DataFrame(index=df.index)
        for col in df.columns:
            s = df[col]
            if s.dtype.kind in "biufc":
                out[col] = pd.to_numeric(s, errors="coerce")
            else:
                out[col] = s.astype("category").cat.codes.replace(-1, np.nan)
        return out

    @staticmethod
    def _align(X_train, X_test):
        """Restrict both frames to their shared columns, preserving train order."""
        common = [c for c in X_train.columns if c in set(X_test.columns)]
        return X_train[common], X_test[common], common

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> "AdversarialValidator":
        """Train the train-vs-test classifier and compute drift diagnostics.

        Returns ``self``; ``self.auc_`` holds the out-of-fold ROC-AUC.
        """
        X_train, X_test, common = self._align(X_train, X_test)
        self.feature_names_ = list(common)
        if not common:
            self.auc_ = 0.5
            return self

        Xtr = self._encode(X_train)
        Xte = self._encode(X_test)

        X_all = pd.concat([Xtr, Xte], axis=0, ignore_index=True)
        y_all = np.concatenate([np.zeros(len(Xtr)), np.ones(len(Xte))]).astype(int)

        n_splits = max(2, min(self.cv, int(min(y_all.sum(), len(y_all) - y_all.sum()))))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        oof_proba = np.full(len(X_all), np.nan)
        importances = np.zeros(len(common))
        n_models = 0

        for train_idx, val_idx in skf.split(X_all, y_all):
            model = self._build_estimator()
            model.fit(X_all.iloc[train_idx], y_all[train_idx])
            proba = model.predict_proba(X_all.iloc[val_idx])[:, 1]
            oof_proba[val_idx] = proba
            if hasattr(model, "feature_importances_"):
                imp = np.asarray(model.feature_importances_, dtype=float)
                if imp.shape[0] == len(common):
                    importances += imp
                    n_models += 1

        try:
            self.auc_ = float(roc_auc_score(y_all, oof_proba))
        except ValueError:
            self.auc_ = 0.5

        if n_models:
            importances /= n_models
        imp_sum = importances.sum()
        if imp_sum > 0:
            norm_imp = importances / imp_sum
        else:
            norm_imp = np.zeros_like(importances)

        self.feature_importances_ = dict(zip(common, importances.tolist()))

        # Drift score = normalized importance scaled by the strength of the shift
        # signal (0 when AUC ~= 0.5 -> no penalty, ramps up as AUC -> 1).
        signal = max(0.0, min(1.0, 2.0 * (self.auc_ - 0.5)))
        max_norm = norm_imp.max() if norm_imp.size and norm_imp.max() > 0 else 1.0
        self._drift_scores = {
            col: float(min(1.0, (norm_imp[i] / max_norm) * signal))
            for i, col in enumerate(common)
        }

        # Out-of-fold P(test) for the training rows only (first len(Xtr) entries).
        self._oof_test_proba = oof_proba[:len(Xtr)]
        return self

    def feature_drift_scores(self) -> dict:
        """Per-feature drift score in [0, 1]; higher means more train/test shift."""
        return dict(self._drift_scores)

    def oof_test_likeness(self) -> np.ndarray:
        """Out-of-fold P(row is test-like) for each training row, in [0, 1]."""
        if self._oof_test_proba is None:
            return np.array([])
        return np.nan_to_num(self._oof_test_proba, nan=0.5)

    def cv_sample_weights(self, floor: float = 0.1) -> np.ndarray:
        """Training-row weights proportional to test-likeness (floored to avoid zeros)."""
        likeness = self.oof_test_likeness()
        if likeness.size == 0:
            return likeness
        return np.clip(likeness, floor, None)
