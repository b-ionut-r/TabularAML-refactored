import pandas as pd
import pytest

from tabularaml.preprocessing.encoders import GroupByEncoder, TemporalEncoder


def test_groupby_rank_is_transform_batch_independent():
    train = pd.DataFrame({
        "cat": ["A", "A", "B", "B"],
        "num": [1.0, 2.0, 10.0, 20.0],
    })
    val_a = pd.DataFrame({"cat": ["A", "A"], "num": [100.0, 200.0]})
    val_b = pd.DataFrame({"cat": ["A", "A"], "num": [200.0, 100.0]})

    enc = GroupByEncoder(cat_col="cat", num_col="num", agg_func="rank", output_col="gb_rank")
    enc.fit(train)

    out_a = enc.transform(val_a)["gb_rank"].tolist()
    out_b = enc.transform(val_b)["gb_rank"].tolist()

    assert out_a == pytest.approx([1.0, 1.0])
    assert out_b == pytest.approx([1.0, 1.0])


def test_temporal_lag_uses_fit_history_only():
    train = pd.DataFrame({
        "id": [1, 1],
        "t": [1, 2],
        "x": [10.0, 20.0],
    })
    val_pair = pd.DataFrame({
        "id": [1, 1],
        "t": [3, 4],
        "x": [30.0, 40.0],
    })
    val_single = pd.DataFrame({
        "id": [1],
        "t": [4],
        "x": [40.0],
    })

    enc = TemporalEncoder(col="x", id_col="id", time_col="t", op_name="lag_1", output_col="lag1")
    enc.fit(train)

    out_pair = enc.transform(val_pair)["lag1"].tolist()
    out_single = enc.transform(val_single)["lag1"].tolist()

    assert out_pair == pytest.approx([20.0, 20.0])
    assert out_single == pytest.approx([20.0])


def test_groupby_mean_handles_categorical_unseen_without_setitem_error():
    train = pd.DataFrame({
        "cat": pd.Categorical(["a", "b"]),
        "num": [1.0, 2.0],
    })
    val = pd.DataFrame({
        "cat": pd.Categorical(["a", "c"]),
        "num": [10.0, 20.0],
    })

    enc = GroupByEncoder(cat_col="cat", num_col="num", agg_func="mean", output_col="gb_mean")
    enc.fit(train)

    out = enc.transform(val)["gb_mean"].tolist()

    # Known category maps to train mean; unseen category gets global fallback.
    assert out == pytest.approx([1.0, 1.5])
