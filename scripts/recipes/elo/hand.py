"""Public top-kernel Elo features: per card, over historical and new-merchant transactions separately, counts,
purchase amount / installments / month lag statistics, authorised share, distinct merchants, categories, cities
and subsectors, first / last purchase and their span, purchases per month, category_1 / category_3 shares; plus
first active month as elapsed days. Built only from feature columns, never the target.

    python3 -I hand.py <hold dir> <out dir>
"""
import os, sys
import numpy as np, pandas as pd

H, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
tr, te = pd.read_parquet(f"{H}/train.parquet"), pd.read_parquet(f"{H}/test.parquet")
ref = pd.Timestamp("2018-05-01")


def agg(name):
    t = pd.read_parquet(f"{H}/{name}.parquet")
    t["authorized"] = (t.authorized_flag == "Y").astype(np.float32)
    t["cat1"] = (t.category_1 == "Y").astype(np.float32)
    for v in "ABC":
        t[f"cat3_{v}"] = (t.category_3 == v).astype(np.float32)
    t["month"] = t.purchase_date.dt.to_period("M").astype(str)
    g = t.groupby("card_id")
    f = g.agg(n=("purchase_amount", "size"), amt_sum=("purchase_amount", "sum"), amt_mean=("purchase_amount", "mean"),
              amt_max=("purchase_amount", "max"), amt_min=("purchase_amount", "min"), amt_std=("purchase_amount", "std"),
              inst_sum=("installments", "sum"), inst_mean=("installments", "mean"), inst_max=("installments", "max"),
              lag_mean=("month_lag", "mean"), lag_min=("month_lag", "min"), lag_max=("month_lag", "max"),
              auth=("authorized", "mean"), cat1=("cat1", "mean"), cat3_A=("cat3_A", "mean"), cat3_B=("cat3_B", "mean"),
              cat3_C=("cat3_C", "mean"), cat2_mean=("category_2", "mean"),
              merchants=("merchant_id", "nunique"), mcats=("merchant_category_id", "nunique"),
              cities=("city_id", "nunique"), states=("state_id", "nunique"), subsectors=("subsector_id", "nunique"),
              months=("month", "nunique"), first=("purchase_date", "min"), last=("purchase_date", "max"))
    f["span_days"] = (f["last"] - f["first"]).dt.days
    f["since_first"] = (ref - f["first"]).dt.days
    f["since_last"] = (ref - f["last"]).dt.days
    f["per_month"] = f["n"] / f["months"]
    f = f.drop(columns=["first", "last"])
    f.columns = [f"{name}_{c}" for c in f.columns]
    return f


F = agg("hist").join(agg("new"), how="outer")
for d in (tr, te):
    d["fam_days"] = (ref - pd.to_datetime(d.first_active_month)).dt.days
tr = tr.join(F, on="card_id"); te = te.join(F, on="card_id")
tr["new_n"] = tr["new_n"].fillna(0); te["new_n"] = te["new_n"].fillna(0)
tr.to_parquet(f"{out}/train_features.parquet", index=False); te.to_parquet(f"{out}/test_features.parquet", index=False)
print("hand columns", te.shape[1])
