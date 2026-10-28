"""Elo Merchant Category Recommendation (2019): 20% of the training cards held out at random (seed 0), the way the
contest's test cards were drawn (same first-active months as train). Transactions of the training cards only, as
two child tables keyed by card_id with purchase_date as time. Writes hold/{train,test,hold_y,train_shuf,hist,new}.parquet.

    python3 -I prep.py <dir with the contest csv files> <out dir>
"""
import os, sys
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.csv as pc, pyarrow.compute as pcm, pyarrow.parquet as pq

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
tr = pd.read_csv(f"{src}/train.csv")
rng = np.random.default_rng(0)
hold = rng.random(len(tr)) < 0.2
a, b = tr[~hold].reset_index(drop=True), tr[hold].reset_index(drop=True)
a.to_parquet(f"{out}/train.parquet", index=False)
b[["target"]].to_parquet(f"{out}/hold_y.parquet", index=False)
b.drop(columns="target").to_parquet(f"{out}/test.parquet", index=False)
s = a.copy(); s["target"] = rng.permutation(s["target"].to_numpy()); s.to_parquet(f"{out}/train_shuf.parquet", index=False)
TYPES = {**{c: pa.string() for c in ("authorized_flag", "card_id", "category_1", "category_3", "merchant_id")},
         **{c: pa.int64() for c in ("city_id", "installments", "merchant_category_id", "month_lag", "state_id",
                                    "subsector_id")},
         "purchase_amount": pa.float64(), "category_2": pa.float64(), "purchase_date": pa.timestamp("s")}
cards = pa.array(tr["card_id"].to_numpy(dtype=str))
for name, f in (("hist", "historical_transactions"), ("new", "new_merchant_transactions")):
    r = pc.open_csv(f"{src}/{f}.csv", read_options=pc.ReadOptions(block_size=1 << 26),
                    convert_options=pc.ConvertOptions(timestamp_parsers=["%Y-%m-%d %H:%M:%S"],
                                                      column_types=TYPES))
    w = None
    for bt in r:
        bt = pa.Table.from_batches([bt]).filter(pcm.is_in(bt["card_id"], value_set=cards))
        if w is None:
            w = pq.ParquetWriter(f"{out}/{name}.parquet", bt.schema)
        w.write_table(bt)
    w.close()
    print(name, pq.ParquetFile(f"{out}/{name}.parquet").metadata.num_rows, flush=True)
print("train", a.shape, "test", b.shape, "outliers in test", int((b.target < -30).sum()))
