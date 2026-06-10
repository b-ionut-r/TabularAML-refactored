"""End-to-end micro smoke runs across task types and acceptance modes."""
import numpy as np
import pandas as pd
import pytest

from tabularaml.generate.features import FeatureGenerator
from tabularaml.eval.scorers import PREDEFINED_REG_SCORERS


def _micro_kwargs(**over):
    kw = dict(n_generations=3, n_parents=6, n_children=24, cv=3, log_file=None,
              early_stopping_iter=10, meta_validation_frac=0.0,
              final_selection=False, time_budget=90)
    kw.update(over)
    return kw


@pytest.mark.parametrize("acceptance", ["statistical", "mean"])
def test_smoke_classification(toy_cls, acceptance):
    X, y = toy_cls
    gen = FeatureGenerator(**_micro_kwargs(acceptance=acceptance,
                                           use_proxy_evaluation=False))
    X_out, pipeline, generation, interactions = gen.search(X, y)
    assert len(X_out) == len(X)
    fresh = gen.transform(X.iloc[:30].copy())
    assert len(fresh) == 30


def test_smoke_regression_with_proxy(toy_reg):
    X, y = toy_reg
    gen = FeatureGenerator(**_micro_kwargs(use_proxy_evaluation=True,
                                           proxy_mode="batched"))
    X_out, _, _, interactions = gen.search(X, y)
    assert len(X_out) == len(X)
    # every accepted non-pipeline interaction is materialized in X_out
    for inter in interactions:
        if not inter.require_pipeline:
            assert inter.name in X_out.columns


def test_smoke_era_dataset():
    rng = np.random.default_rng(11)
    n_eras, per_era = 40, 30
    n = n_eras * per_era
    eras = np.repeat(np.arange(n_eras), per_era)
    X = pd.DataFrame({"era": eras,
                      "f1": rng.normal(size=n), "f2": rng.normal(size=n),
                      "f3": rng.normal(size=n)})
    y = pd.Series(X["f1"] * 0.6 + rng.normal(size=n) * 0.6)
    gen = FeatureGenerator(**_micro_kwargs(era_col="era",
                                           scorer=PREDEFINED_REG_SCORERS["era_spearman"],
                                           task="regression",
                                           use_proxy_evaluation=False,
                                           expand_row_stats=False))
    X_out, _, _, _ = gen.search(X, y)
    assert "era" not in X_out.columns
    # search ran and produced a sane summary metric
    assert np.isfinite(gen.final_metric)
