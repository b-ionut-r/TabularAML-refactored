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

### All 112 stocks: child tables read a key range at a time (`tabularaml/generate/stream.py`)

A keyed parquet child table of over 30M rows whose aggregation would take more than 40% of memory (rows x
(numeric columns + 2 x same-unit pairs) x 12 bytes) is read a key range at a time: RelatedTables (levels and columns
fixed by the first range), price paths and latest-state history are computed per range and stacked, equal to the
whole-table result (tests/test_stream.py); child-row models and same-code matches are skipped for it. Optiver's full
book (167M rows, 62 GB to aggregate whole) runs in 20 ranges. Same holdout (random 20% of buckets), 112 stocks,
samples 0 / 1:

| Arm | RMSPE | Build, peak memory |
|---|---|---|
| raw (stock_id) | 0.605 / 0.616 | |
| winners' public features | 0.242 / 0.245 | 12 min, 1.3 GB |
| **FeatureForge** | **0.230 / 0.233** | 36-44 min, 6.7 GB |
| shuffled targets | 0.816 (constant 0.816) | |

On 12 stocks the streamed run (no child-row models) scored 0.317 / 0.324 in 8-12 min at 3.8 GB, against 0.316 / 0.341
whole (25 min, 9.6 GB).

## Intraday panels (`tabularaml/generate/panel.py`): Optiver Trading at the Close (Kaggle 2023)

One row per stock, day and 10-second step of the closing auction (5.2M rows, 200 stocks). The family switches on when
an entity recurs in the test rows, a moment's rows sit together in the file with at least 10 entities each, and the
test rows' days are new. It adds same-unit imbalances (sizes against sizes, prices against prices, signed by a
-1 / 0 / 1 side flag that shares the amount's name), each column against the moment's mean and its rank in the moment,
and the entity's change over its last 1-3 steps that day (earlier steps only). Off on Rossmann, Walmart, Recruit,
Favorita, Enefit, GoDaddy, Riiid and every random split.

`scripts/tatc_bench.py`: last 45 days held out (window 0: days 436-480, 1: days 391-435), 120 training days, L1
bagged LightGBM; MAE, mean of 3 judge seeds (seed noise about 0.01-0.02).

| arm | window 0 | window 1 |
|---|---|---|
| raw | 5.904 | 6.029 |
| winners' hand features | 5.844 | 5.984 |
| PR #2 head 8d6cf0e | 5.886 | 6.016 |
| 8d6cf0e + panel family | 5.851 | 6.001 |

Build 17-19 min (panel part seconds), judge peak 5.8 GB. The winners' gap left is mostly in their imbalance and
cross-stock groups beyond these (window 1: imbalance group alone 5.979).
