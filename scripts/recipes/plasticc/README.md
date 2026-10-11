# PLAsTiCC 2018, wide-fast-deep holdout

- Data: Kaggle dataset `siddharthchaini/unblinded-data-for-plasticc-challenge` (no rules acceptance). Download
  `plasticc_train_metadata.csv`, `plasticc_train_lightcurves.csv`, `plasticc_test_metadata.csv` and
  `plasticc_test_set_batch2.csv` (each comes as a .csv.zip): `/api/v1/datasets/download/<owner>/<slug>/<file>`.
- Split: `python prep.py <dir>`. Train = the 7,848 training objects. Held out = 30,000 test objects (seed 0)
  from batch 2 (wide-fast-deep, as most of the real test), true class among the training classes.
  70% of training objects have a spectroscopic redshift, 3% of held-out ones (the contest's shift).
- FeatureForge: `python scripts/contest_features.py --train <dir>/hold/train.parquet --test <dir>/hold/test.parquet
  --target target --id object_id --task multiclass --table lc=<dir>/hold/lc.parquet:object_id:mjd --out-dir out/ff`
  (shuffled: `--train .../train_shuf.parquet`). Default budget.
- Judge: `judge.py <arm> <dir>/hold <out-parent>` (reads <out-parent>/pout/<arm>). Bagged multiclass LightGBM, multiclass logloss, plain and with
  the contest's class weights. Hand arm: `hand.py out/hand` (public kernel features).
- Results at 0b5560b (one split): logloss raw 2.065, kernel features 1.369, FeatureForge 0.940
  (4 min, 1.2 GB), shuffled 1.831 (class prior). Weighted: 3.181, 1.302, 1.757, 3.261.
