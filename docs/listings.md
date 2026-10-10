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
