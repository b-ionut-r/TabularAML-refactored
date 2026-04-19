import numpy as np
import pandas as pd

from tabularaml.preprocessing.encoders import CategoricalEncoder
from tabularaml.preprocessing.pipeline import PipelineWrapper


def test_categorical_encoder_multiclass_expands_target_columns():
    X = pd.DataFrame(
        {
            "cat": pd.Categorical(["a", "a", "b", "b", "c", "c"]),
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    y = pd.Series([0, 1, 2, 0, 1, 2])

    enc = CategoricalEncoder(target_enc_cols=["cat"])
    Xt = enc.fit_transform(X, y)

    target_cols = [col for col in Xt.columns if col.startswith("cat_target")]
    assert len(target_cols) == 2
    assert all(col.startswith("cat_target_") for col in target_cols)
    assert enc.n_new_feats >= 2
    assert np.isfinite(Xt[target_cols].to_numpy(dtype=float)).all()


def test_categorical_encoder_binary_keeps_single_target_column():
    X = pd.DataFrame(
        {
            "cat": pd.Categorical(["a", "a", "b", "b", "c", "c"]),
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    y = pd.Series([0, 0, 1, 1, 0, 1])

    enc = CategoricalEncoder(target_enc_cols=["cat"])
    Xt = enc.fit_transform(X, y)

    target_cols = [col for col in Xt.columns if col.startswith("cat_target")]
    assert target_cols == ["cat_target"]


def test_pipeline_wrapper_handles_multiclass_target_encoder_outputs():
    X = pd.DataFrame(
        {
            "cat": pd.Categorical(["a", "a", "b", "b", "c", "c"]),
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    y = pd.Series([0, 1, 2, 0, 1, 2])

    wrapper = PipelineWrapper(
        imputer=None,
        scaler=None,
        encoder=CategoricalEncoder(target_enc_cols=["cat"]),
    )
    pipeline = wrapper.get_pipeline(X, y)

    Xt = pipeline.fit_transform(X, y)
    multiclass_target_cols = [col for col in Xt.columns if col.startswith("cat_target_")]
    assert len(multiclass_target_cols) == 2
