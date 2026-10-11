"""PLAsTiCC holdout on wide-fast-deep test objects (most of the real test set), from the unblinded release on
Kaggle (siddharthchaini/unblinded-data-for-plasticc-challenge, no rules acceptance): train = the contest's 7,848
training objects; held out = 30,000 random objects (seed 0) of plasticc_test_set_batch2.csv whose true class is
a training class. Metadata keeps the contest's columns only (no true_* columns). Writes hold/{train,test,hold_y,
train_shuf,lc}.parquet; lc holds every light-curve row of both."""
import sys, os, zipfile, numpy as np, pandas as pd
D = sys.argv[1]   # folder with the four downloaded .csv.zip files
os.makedirs(f"{D}/hold", exist_ok=True)
rd = lambda f, **k: pd.read_csv(zipfile.ZipFile(f"{D}/{f}.zip").open(f), skipinitialspace=True, **k)
keep = ["object_id", "ra", "decl", "ddf_bool", "hostgal_specz", "hostgal_photoz", "hostgal_photoz_err", "distmod", "mwebv"]
tm = rd("plasticc_train_metadata.csv"); classes = set(tm.target.unique()); tr = tm[keep + ["target"]]
te_meta = rd("plasticc_test_metadata.csv", usecols=keep + ["true_target"])
lc_te = rd("plasticc_test_set_batch2.csv", dtype={"passband": "int8", "detected_bool": "int8"})
ids = te_meta[te_meta.object_id.isin(lc_te.object_id.unique()) & te_meta.true_target.isin(classes)].object_id.to_numpy()
pick = np.random.default_rng(0).choice(ids, 30000, replace=False)
te = te_meta[te_meta.object_id.isin(pick)].reset_index(drop=True)
lc = pd.concat([rd("plasticc_train_lightcurves.csv", dtype={"passband": "int8", "detected_bool": "int8"}),
                lc_te[lc_te.object_id.isin(pick)]], ignore_index=True)
for c in ("mjd", "flux", "flux_err"):
    lc[c] = lc[c].astype("float32")
tr.to_parquet(f"{D}/hold/train.parquet", index=False)
te.drop(columns="true_target").to_parquet(f"{D}/hold/test.parquet", index=False)
te[["object_id", "true_target"]].rename(columns={"true_target": "target"}).to_parquet(f"{D}/hold/hold_y.parquet", index=False)
lc.to_parquet(f"{D}/hold/lc.parquet", index=False)
t2 = tr.copy(); t2["target"] = np.random.default_rng(0).permutation(t2.target.to_numpy())
t2.to_parquet(f"{D}/hold/train_shuf.parquet", index=False)
