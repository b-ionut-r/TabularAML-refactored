"""Public tabular datasets used to measure feature-engineering lift.

Mix of contest-style tables with real categoricals (Ames housing, Titanic,
Adult, Covertype, FARS, ...) and numeric PMLB regression/classification
tables. Files are downloaded on first use into ``$TABULARAML_DATA``
(default ``~/data``) and returned as ``(X, y, task, metric)``.
"""
from __future__ import annotations

import json
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


# Real contest tables from OpenML: Kaggle competitions (Amazon, Porto Seguro,
# Allstate, Give Me Some Credit, Kick, Bike Sharing, House Prices, Mercedes),
# KDD Cup 2009, and the original datasets behind Kaggle Playground episodes
# (bank marketing -> S5E8, bank churn -> S4E1, abalone -> S4E4, steel plates ->
# S4E3, diamonds -> S3E8, credit default, ...). ``url`` is ``openml:<data id>``;
# target and ignored columns come from the OpenML description.
OML = "openml:{}"
CONTEST = {d.name: d for d in [
    SuiteDataset("amazon", OML.format(4135), "", "binary", "logloss", max_rows=None),
    SuiteDataset("porto_seguro", OML.format(42742), "", "binary", "logloss", drop=["id"], max_rows=100_000),
    SuiteDataset("allstate", OML.format(42571), "", "regression", "rmsle", drop=["id"], max_rows=60_000),
    SuiteDataset("kdd_appetency", OML.format(1111), "", "binary", "logloss", max_rows=None),
    SuiteDataset("kdd_upselling", OML.format(1114), "", "binary", "logloss", max_rows=None),
    SuiteDataset("bank_marketing", OML.format(1461), "", "binary", "logloss", max_rows=None),
    SuiteDataset("bank_churn", OML.format(46911), "", "binary", "logloss", max_rows=None),
    SuiteDataset("give_me_credit", OML.format(46929), "", "binary", "logloss", max_rows=60_000),
    SuiteDataset("kick", OML.format(41162), "", "binary", "logloss", max_rows=None),
    SuiteDataset("bike_sharing", OML.format(42712), "", "regression", "rmsle", max_rows=None),
    SuiteDataset("credit_default", OML.format(46919), "", "binary", "logloss", max_rows=None),
    SuiteDataset("hr_analytics", OML.format(46935), "", "binary", "logloss", max_rows=None),
    SuiteDataset("coupon", OML.format(46937), "", "binary", "logloss", max_rows=None),
    SuiteDataset("abalone", OML.format(46903), "", "regression", "rmsle", max_rows=None),
    SuiteDataset("diamonds", OML.format(42225), "", "regression", "rmsle", max_rows=None),
    SuiteDataset("airline_satisfaction", OML.format(46920), "", "binary", "logloss", max_rows=60_000),
    SuiteDataset("food_delivery", OML.format(46928), "", "regression", "rmse", max_rows=None),
    SuiteDataset("steel_plates", OML.format(46959), "", "multiclass", "logloss", max_rows=None),
    SuiteDataset("click", OML.format(42733), "", "binary", "logloss", max_rows=None),
    SuiteDataset("house_prices", OML.format(42563), "", "regression", "rmsle", drop=["Id"], max_rows=None),
    SuiteDataset("mercedes", OML.format(42570), "", "regression", "rmse", drop=["ID"], max_rows=None),
    SuiteDataset("miami_housing", OML.format(46942), "", "regression", "rmsle", max_rows=None),
]}


# Tables with entity / ID / time structure (flights, providers, locations, stops,
# wineries): the setting where contest feature engineering usually earns most.
# Columns that would make the target trivial are dropped (total_amount includes
# the tip; Medicare payments are most of the total payment).
STRUCTURED = {d.name: d for d in [
    SuiteDataset("airlines", OML.format(1169), "", "binary", "logloss", cat_cols=["Flight"], max_rows=100_000),
    SuiteDataset("medical_charges", OML.format(42130), "", "regression", "rmsle",
                 drop=["provider_name", "provider_street_address", "average_medicare_payments"],
                 cat_cols=["provider_id", "provider_zip_code"], max_rows=100_000),
    SuiteDataset("nyc_taxi_tip", OML.format(42729), "", "regression", "rmse", drop=["total_amount"],
                 max_rows=100_000),
    SuiteDataset("sf_crime", OML.format(42344), "", "binary", "logloss", max_rows=100_000),
    SuiteDataset("kc_house", OML.format(42731), "", "regression", "rmsle", drop=["id"], cat_cols=["zipcode"],
                 max_rows=None),
    SuiteDataset("wine_reviews", OML.format(41275), "", "regression", "rmse", max_rows=100_000),
    SuiteDataset("zurich_delays", OML.format(42495), "", "regression", "rmse", max_rows=None),
]}

def data_dir() -> Path:
    d = Path(os.environ.get("TABULARAML_DATA", Path.home() / "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_openml(did: int):
    """Parquet copy of an OpenML dataset plus its default target and ignored columns."""
    meta_f = data_dir() / f"openml_{did}.json"
    if not meta_f.exists():
        with urllib.request.urlopen(f"https://www.openml.org/api/v1/json/data/{did}", timeout=60) as r:
            meta_f.write_bytes(r.read())
    meta = json.loads(meta_f.read_text())["data_set_description"]
    pq = data_dir() / f"openml_{did}.pq"
    if not pq.exists():
        urllib.request.urlretrieve(meta["parquet_url"], pq)
    df = pd.read_parquet(pq)
    drop = []
    for key in ("ignore_attribute", "row_id_attribute"):
        v = meta.get(key)
        drop += v.split(",") if isinstance(v, str) else list(v or [])  # may be one comma-joined string
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == bool:
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df.drop(columns=[c for c in drop if c in df.columns]), meta["default_target_attribute"]


def load_suite_dataset(name: str, seed: int = 0):
    spec = SUITE.get(name) or CONTEST.get(name) or STRUCTURED[name]
    if spec.url.startswith("openml:"):
        df, target = _load_openml(int(spec.url.split(":")[1]))
        spec = SuiteDataset(**{**spec.__dict__, "target": spec.target or target})
    else:
        fname = data_dir() / (spec.url.split("/datasets/")[-1].replace("/", "_"))
        if not fname.exists():
            urllib.request.urlretrieve(spec.url, fname)
        df = pd.read_csv(fname, sep=spec.sep)
    df = df.drop(columns=[c for c in spec.drop if c in df.columns])
    if spec.max_rows and len(df) > spec.max_rows:
        df = df.sample(spec.max_rows, random_state=seed).reset_index(drop=True)
    y = df.pop(spec.target)
    if spec.task == "regression":
        y = y.astype(float)
    for c in spec.cat_cols:
        df[c] = df[c].astype(str)
    if spec.task != "regression":
        y = pd.Series(pd.factorize(y, sort=True)[0], name=spec.target)
    return df, y, spec.task, spec.metric
