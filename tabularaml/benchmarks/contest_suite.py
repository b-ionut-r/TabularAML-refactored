"""Public tabular datasets used to measure feature-engineering lift.

Mix of contest-style tables with real categoricals (Ames housing, Titanic,
Adult, Covertype, FARS, ...) and numeric PMLB regression/classification
tables. Files are downloaded on first use into ``$TABULARAML_DATA``
(default ``~/data``) and returned as ``(X, y, task, metric)``.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

AG = "https://autogluon.s3.amazonaws.com/datasets/"
PMLB = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{0}/{0}.tsv.gz"


@dataclass
class SuiteDataset:
    name: str
    url: str
    target: str
    task: str            # regression | binary | multiclass
    metric: str
    drop: List[str] = field(default_factory=list)
    cat_cols: List[str] = field(default_factory=list)
    sep: str = ","
    max_rows: Optional[int] = 20_000


SUITE = {d.name: d for d in [
    # --- real categoricals / contest classics
    SuiteDataset("ames", AG + "AmesHousingPriceRegression/train_data.csv", "SalePrice",
                 "regression", "rmsle", drop=["Order", "PID"], cat_cols=["MS.SubClass"]),
    SuiteDataset("titanic", AG + "titanic/train.csv", "Survived", "binary", "logloss",
                 drop=["PassengerId", "Name"]),
    SuiteDataset("adult", AG + "AdultIncomeBinaryClassification/train_data.csv", "class",
                 "binary", "logloss"),
    SuiteDataset("covertype", AG + "CoverTypeMulticlassClassification/train_data.csv",
                 "Cover_Type", "multiclass", "logloss"),
    SuiteDataset("fars", PMLB.format("fars"), "target", "multiclass", "logloss", sep="\t"),
    SuiteDataset("churn", PMLB.format("churn"), "target", "binary", "logloss", sep="\t"),
    SuiteDataset("coil2000", PMLB.format("coil2000"), "target", "binary", "logloss", sep="\t"),
    SuiteDataset("sleep", PMLB.format("sleep"), "target", "multiclass", "logloss", sep="\t"),
    # --- numeric
    SuiteDataset("houses", PMLB.format("537_houses"), "target", "regression", "rmse", sep="\t"),
    SuiteDataset("fried", PMLB.format("564_fried"), "target", "regression", "rmse", sep="\t"),
    SuiteDataset("house_16H", PMLB.format("574_house_16H"), "target", "regression", "rmse", sep="\t"),
    SuiteDataset("pol", PMLB.format("201_pol"), "target", "regression", "rmse", sep="\t"),
    SuiteDataset("wind", PMLB.format("503_wind"), "target", "regression", "rmse", sep="\t"),
    SuiteDataset("puma8NH", PMLB.format("225_puma8NH"), "target", "regression", "rmse", sep="\t"),
    SuiteDataset("cpu_act", PMLB.format("197_cpu_act"), "target", "regression", "rmse", sep="\t"),
    SuiteDataset("magic", PMLB.format("magic"), "target", "binary", "logloss", sep="\t"),
    SuiteDataset("phoneme", PMLB.format("phoneme"), "target", "binary", "logloss", sep="\t"),
    SuiteDataset("spambase", PMLB.format("spambase"), "target", "binary", "logloss", sep="\t"),
    SuiteDataset("wine_white", PMLB.format("wine_quality_white"), "target", "multiclass", "logloss", sep="\t"),
    SuiteDataset("page_blocks", PMLB.format("page_blocks"), "target", "multiclass", "logloss", sep="\t"),
]}


def data_dir() -> Path:
    d = Path(os.environ.get("TABULARAML_DATA", Path.home() / "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_suite_dataset(name: str, seed: int = 0):
    spec = SUITE[name]
    fname = data_dir() / (spec.url.split("/datasets/")[-1].replace("/", "_"))
    if not fname.exists():
        urllib.request.urlretrieve(spec.url, fname)
    df = pd.read_csv(fname, sep=spec.sep)
    df = df.drop(columns=[c for c in spec.drop if c in df.columns])
    if spec.max_rows and len(df) > spec.max_rows:
        df = df.sample(spec.max_rows, random_state=seed).reset_index(drop=True)
    y = df.pop(spec.target)
    for c in spec.cat_cols:
        df[c] = df[c].astype(str)
    if spec.task != "regression":
        y = pd.Series(pd.factorize(y, sort=True)[0], name=spec.target)
    return df, y, spec.task, spec.metric
