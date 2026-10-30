"""Regressions found while running the genetic search on a small regression table under pandas 3."""
import numpy as np
import pandas as pd

from tabularaml.generate.features import FeatureGenerator
from tabularaml.preprocessing.pipeline import PipelineWrapper


def test_bare_log_file_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    FeatureGenerator(log_file="run.log", use_gpu=False)


def test_dtypes_recategorise_categorical_and_str_columns():
    X = pd.DataFrame({"pos": pd.Categorical(["WR", "TE", None] * 20),
                      "team": pd.Series(["KC", "BUF", "KC"] * 20, dtype="str"),
                      "w": np.arange(60.0)})
    pw = PipelineWrapper()
    pw._get_dtypes(X)   # raised TypeError under pandas 3 (categorical written into with other categories)
    assert set(pw.categorical_columns) == {"pos", "team"} and pw.numerical_columns == ["w"]


def test_inferred_regression_task_survives_search(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({"pos": pd.Categorical(rng.choice(["WR", "TE", "CB", "DE"], n)),
                      "a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(X["a"] * 2 + X["pos"].cat.codes + rng.normal(scale=0.1, size=n), name="y")
    fg = FeatureGenerator(mode="lite", time_budget=20, cv=3, use_gpu=False, random_state=0, n_jobs=1,
                          log_file=str(tmp_path / "gen.log"))
    fg.search(X.copy(), y.copy())
    fg.fit(X.copy(), y.copy())
    out = fg.transform(X.copy())
    # A continuous target is never factorized into one class per value (one target column per label).
    assert sum(c.startswith("pos_target_") for c in out.columns) == 0
