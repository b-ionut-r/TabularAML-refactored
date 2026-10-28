# Elo Merchant Category Recommendation (2019), run blind

Data: public copy `ershisuila/eio-recommend` (file sizes equal the contest's). `prep.py` holds out 20% of the 202k
training cards at random (seed 0), the way the contest's test cards were drawn, and keeps the 19M transactions of
the training cards as child tables `hist` / `new` (key card_id, time purchase_date). `hand.py` rebuilds the public
kernels' per-card aggregates; `judge.py <arm>` is the shared bagged LightGBM (RMSE, outliers kept).

FeatureForge: `contest_features.py --id card_id --table hist=hold/hist.parquet:card_id:purchase_date
--table new=hold/new.parquet:card_id:purchase_date`.

| arm | RMSE | build | peak |
|---|---|---|---|
| raw (main table) | 3.8614 | | |
| public-kernel features | 3.6877 | 35 s | 5.3 GB |
| FeatureForge 6c034b5 (related tables; search kept 0) | 3.7049 | 12 min | 8.3 GB |
| shuffled labels, 6c034b5 | 3.8684 | 7 min | 6.5 GB |
