"""Google Analytics Customer Revenue (v2 data) as a session-level holdout: one row per session, target
log1p(transactionRevenue / 1e6) (0 when absent), the latest 20% of sessions by date held out. JSON columns
(device, geoNetwork, totals, trafficSource) stay JSON strings as the contest gave them; the revenue fields
(transactionRevenue, totalTransactionRevenue, transactions) are removed from totals. The hits column (most
of the file's 25 GB) and customDimensions are dropped."""
import sys, csv, io, json, zipfile, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
csv.field_size_limit(1 << 31)
z = zipfile.ZipFile(src)
f = io.TextIOWrapper(z.open("train_v2.csv"), encoding="utf-8", newline="")
r = csv.reader(f)
head = next(r)
keep = ["channelGrouping", "date", "device", "fullVisitorId", "geoNetwork", "socialEngagementType", "totals",
        "trafficSource", "visitId", "visitNumber", "visitStartTime"]
ix = [head.index(c) for c in keep]
rows, ys = [], []
for row in r:
    v = [row[i] for i in ix]
    t = json.loads(v[keep.index("totals")])
    rev = t.pop("transactionRevenue", None)
    t.pop("totalTransactionRevenue", None); t.pop("transactions", None)
    v[keep.index("totals")] = json.dumps(t)
    rows.append(v); ys.append(np.log1p(float(rev) / 1e6) if rev else 0.0)
d = pd.DataFrame(rows, columns=keep); del rows
d["target"] = np.array(ys, dtype=np.float32)
for c in ("date", "visitNumber", "visitStartTime"):
    d[c] = pd.to_numeric(d[c])
d["sessionId"] = d.fullVisitorId + "_" + d.visitId
d = d.sort_values(["date", "visitStartTime"], kind="stable").reset_index(drop=True)
cut = int(0.8 * len(d))
tr, te = d.iloc[:cut].reset_index(drop=True), d.iloc[cut:].reset_index(drop=True)
print(len(d), "train", len(tr), "hold", len(te), "dates", tr.date.max(), te.date.min(), "buy rate", (tr.target > 0).mean(), (te.target > 0).mean())
tr.to_parquet(f"{out}/train.parquet", index=False)
te[["sessionId", "target"]].to_parquet(f"{out}/hold_y.parquet", index=False)
te.drop(columns="target").to_parquet(f"{out}/test.parquet", index=False)
t2 = tr.copy(); t2["target"] = np.random.default_rng(0).permutation(t2.target.to_numpy())
t2.to_parquet(f"{out}/train_shuf.parquet", index=False)
