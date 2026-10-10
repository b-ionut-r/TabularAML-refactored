# List columns and per-unit amounts (`tabularaml/generate/lists.py`)

Two label-free families in `scripts/contest_features.py`, on from the data's structure (`--lists off` turns both off):

- **List columns**: cells holding lists (a listing's amenities, its photo URLs, a product's tags), as Python lists or
  as short recurring items joined by " ; ", "|" or ",". Per column: the number of items, the number of distinct ones,
  and an indicator per item held by at least 0.5% of the rows (the 100 most common; at least 20 rows). List cells
  used to reach FeatureForge as numpy reprs (`read` turned them into strings); they are now joined with " ; ".
- **Per-unit amounts**: a skewed amount (99th percentile over 3x the median) over each small count it grows with
  (whole or half units from 0 or 1 up to 20, 3-25 levels, mostly non-zero, Spearman >= 0.2 with the amount) and over
  their total. On only for an amount with two such counts: rent per bedroom, per bathroom, per room.

## Two Sigma Connect: Rental Listing Inquiries (Kaggle 2017), the case they come from

Public copy of the contest's train.json (`logan1997/two-sigma-challenge`, 49,352 listings); a random 20% held out
like the contest's test file (same months, managers and buildings); multiclass log loss, same bagged LightGBM;
`scripts/twosigma_bench.py`. Samples 0 / 1:

| Arm | Log loss | Build |
|---|---|---|
| prior | 0.785 / 0.789 | |
| raw | 0.577 / 0.588 | |
| single variant (ae6b09c) | 0.560 / 0.566 | 20 min |
| winners' public features (counts, dates, price per room, manager / building counts and out-of-fold rates, amenity tokens, location) | 0.543 / 0.540 | 16 s |
| single variant + both families | **0.544 / 0.553** | 12-13 min |
| shuffled labels (variant + families) | 0.786 (prior 0.785) | |

On the variant's own features, each winners' group adds little alone (price per room 0.004 / 0.007, amenity
tokens 0.004 / 0.003, others 0-0.002); what the families add is mostly those two.

The families stay off on Rossmann, Walmart, Recruit, Elo, ASHRAE, Enefit, M5, DSB 2019 and Home Credit's tables
(checked on their files), so those results are unchanged.

## New York City Taxi Trip Duration (Kaggle 2017): coordinates, nothing missing

Public copy of train.csv (`yasserh/nyc-taxi-trip-duration`, 1,458,644 trips); a random 20% held out like the contest's
test (same months, `dropoff_datetime` dropped); RMSLE, same bagged LightGBM on log1p; `scripts/nyctaxi_bench.py`.
Samples 0 / 1: raw 0.434 / 0.432 (1 GB); winners' public features without outside data (distances, bearing, PCA
coordinates, k-means clusters, time parts, cluster-hour counts, out-of-fold speeds) 0.377 / 0.375 (7 min, 2.4 GB);
PR #2 head (ac319b5) blind 0.371 / 0.369 (37-40 min, peak 2.9 GB); shuffled 0.798 (constant 0.798). All the hand
features on top of the head: 0.3713 -> 0.3705 (no group above 0.0006). A map family (k-means neighbourhoods shared by
both points, cell / cluster / cluster-hour counts, cluster pairs) scored 0.372 on sample 0 and was dropped.

## Price paths in child tables (`tabularaml/generate/returns.py`): Optiver Realized Volatility Prediction (Kaggle 2021)

For each keyed child table, label-free and on from structure: a price column is positive, not whole numbers, and
moves by under 1% from one row of a parent to the next (median); rows are ordered by the table's time column or by a
column that never decreases within a parent. Per parent: realized volatility sqrt(sum of squared log returns) over
the window and its later half, the net log move, and the share of rows that moved, for each price, their row mean,
and each order-book level's size-weighted price (bid / ask or buy / sell prices with their size / qty / volume
columns: (bid * ask size + ask * bid size) / (bid size + ask size)). `--log-target` now searches on log(y) when the
target is positive with a median below 1 (log1p of a volatility of 0.003 is the volatility itself).

Public copy of the contest's files (`akshaymairal/optiver-realized-volatility-prediction`); 12 of the 112 stocks
(24 ran PR #2's head out of memory on 34M book rows); a random 20% of the time buckets held out with all their
stocks (bucket order is hidden in the files); RMSPE, bagged LightGBM on log(target); `scripts/optiver_bench.py`.
Samples 0 / 1:

| Arm | RMSPE | Build |
|---|---|---|
| raw (stock_id) | 0.748 / 0.817 | |
| winners' public features (WAP 1 / 2, realized volatility over 0 / 150 / 300 / 450 s, spreads, volumes, trade volatility and counts, per-bucket means over stocks) | 0.336 / 0.332 | 2 min |
| PR #2 head (ac319b5) | 0.337 / 0.359 | 21-25 min, 9 GB |
| head + winners' features | 0.340 / 0.337 | |
| price paths without the size-weighted price | 0.357 / 0.348 | 26-27 min |
| **price paths with it** | **0.316 / 0.341** | 23-25 min, 9.6 GB |
| shuffled targets (price paths with it) | 0.852 (constant 0.840) | |
