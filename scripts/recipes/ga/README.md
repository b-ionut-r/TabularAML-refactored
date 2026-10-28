# Google Analytics Customer Revenue (2018), session level, run blind

Data: Kaggle `ga-customer-revenue-prediction` v2 `train_v2.csv` (1.71M sessions). `prep.py` streams it, drops
`hits` / `customDimensions` and every revenue field from `totals`, sorts by visit time and holds out the latest
20% (342k sessions) as the test; target is log1p(revenue). `hand.py` rebuilds the public kernels' features
(JSON flattened, frequency counts, visit hour / weekday). `judge.py <arm>` scores with the shared bagged LightGBM.

| arm | held-out RMSE | build | peak |
|---|---|---|---|
| raw (JSON as categories) | 0.3643 | | |
| hand (public kernels) | 0.3165 | | |
| FeatureForge 8737c2f | out of memory (14 GB) | | |
| FeatureForge 5137abe (JSON fields) | 0.3180 | 17 min | 7.2 GB |
| shuffled labels, 5137abe | 0.3757 (0 kept) | 5 min | 6.3 GB |
