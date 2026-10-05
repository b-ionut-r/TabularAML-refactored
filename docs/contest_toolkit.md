# Contest toolkit: FeatureForge + ContestSolver

Two pieces aimed at tabular ML competitions:

* **`tabularaml.generate.forge.FeatureForge`** — fast, wide automated feature
  engineering that only keeps features that lower loss on rows the search
  never saw.
* **`tabularaml.contest.ContestSolver`** — K-fold LightGBM / XGBoost / CatBoost
  with out-of-fold (OOF) hill-climbed ensembling on the leaderboard metric.

```python
from tabularaml.generate.forge import FeatureForge
from tabularaml.contest import ContestSolver

forge = FeatureForge(time_budget=600).fit(X_train, y_train, X_unlabeled=X_test)
Xtr, Xte = forge.transform_train(X_train), forge.transform(X_test)

solver = ContestSolver(metric="auc").fit(Xtr, y_train, Xte)
pred = solver.test_ensemble_          # probabilities / values for the submission
print(solver.leaderboard_)
```

or from the command line:

```bash
python scripts/contest_run.py --train train.csv --test test.csv --target target \
    --id id --metric auc --budget 600 --out submission.csv
```

## How FeatureForge searches

Each round:

1. **Base model.** Repeated K-fold LightGBM on the current features gives OOF
   margins and split-gain importances.
2. **Candidates** (thousands per round), seeded from the most important columns:
   * pairwise arithmetic `add / sub / mul / div / rdiv`;
   * frequency counts of keys, key pairs and triples, and of numeric values;
   * out-of-fold smoothed target encodings of keys, key pairs and triples
     (per-class for multiclass);
   * group statistics of a numeric within a key: mean, std, min, max,
     deviation from the group mean, z-score;
   * out-of-fold **kNN target features** (mean target / class frequencies of the
     5–100 nearest rows) in subspaces of 2…32 top numerics, both standardised
     and **importance-weighted** (each axis scaled by the root of its split gain,
     so distance follows what the model uses);
   * a rank-normalised (rank-gauss) variant of the weighted kNN for skewed
     axes;
   * same-scale sums of 3–4 top numerics (total-area-style aggregates);
   * out-of-fold **per-class kNN distances** (mean distance to the 1, 2, 4
     nearest rows of each class);
   * **interaction cells**: pairs and triples mined from the base model's tree
     paths (features split on in sequence) and from a FAST-style residual grid
     probe, expanded into every applicable family: pairwise arithmetic, 2-D
     target maps, out-of-fold target encodings and counts of key ×
     quantile-binned-numeric cells, group statistics;
   * PCA and (out-of-fold) PLS projections of the top numerics;
   * row statistics over column families (`Soil_Type_1..40`, `px1..px784`):
     argmax, sum, mean, std, min, max, plus row-wise non-zero and missing counts.

   Round 2+ compose selected features with raw columns (`a*b - c`, group
   statistics of a selected ratio, key triples).
3. **Cross-fitted residual screening.** Each candidate is binned and receives a
   regularised Newton step on the current model's gradients, fitted on 4/5 of
   the rows and applied to the remaining 1/5 (a one-feature tree on the
   residuals). The gain is the held-out loss reduction. Thousands of candidates
   are scored per second in vectorised NumPy.
4. **Novelty filter.** A candidate's gain must exceed the gain of its own parent
   columns under the same probe. Without this, re-expressions of a column that
   the early-stopped base model under-uses dominate the ranking.
5. **Selection.** Survivors are ordered two ways, by split gain in a joint
   model and by novel residual gain; the best prefix (3, 6, 12, 25, ...) of
   either order is chosen by repeated-CV loss.

Finally the **gate**: 20% of the training rows are held out before any search.
Raw features and each cumulative round are compared on those rows with the
same model; a feature set is kept only if its winsorised paired per-row
improvement has z ≥ 1 (z ≥ 1.645 when the gate has fewer than 1000 rows).
Otherwise FeatureForge returns the raw features unchanged.

Target-dependent features (target encodings, kNN) are always out-of-fold on
training rows (`transform_train`) and fitted on all training rows for new data
(`transform`).

## Benchmarks

* `scripts/bench_fe.py` — FE lift on an untouched outer 20% holdout, same
  downstream model (5-fold bagged LightGBM) on raw vs. engineered features.
  Datasets: `tabularaml/benchmarks/contest_suite.py`.
* `scripts/summarize_fe.py` — per-dataset lift, win rate and FE runtime.
* `scripts/bench_contest.py` — ContestSolver vs. the repo's fixed XGBoost base
  learner on the same kind of holdout.

## Measured results (outer 20% holdout, 3 splits per dataset)

Raw CSVs: `docs/results/`. Lift = % reduction of holdout loss (logloss / RMSE /
RMSLE) of 5-fold bagged LightGBM on engineered vs. raw features; positive is better.

### FeatureForge, 20 datasets × 3 holdouts (default settings, 300 s budget)

| Dataset | Holdout lift, current (mean of 3) | First version (v7) | FE time | Features kept |
|---|---|---|---|---|
| pol | +18.26% | +0.00% | 52s | 16.0 |
| magic | +13.77% | +11.30% | 33s | 31.3 |
| churn | +7.97% | -0.36% | 28s | 24.7 |
| covertype | +6.90% | +0.00% | 90s | 118.0 |
| phoneme | +6.60% | +0.00% | 7s | 7.7 |
| wine_white | +5.56% | +0.00% | 52s | 37.3 |
| adult | +4.29% | +4.24% | 32s | 13.0 |
| houses | +3.58% | +3.62% | 13s | 5.7 |
| fried | +0.82% | +0.56% | 18s | 3.0 |
| wind | +0.68% | +0.95% | 11s | 3.7 |
| coil2000 | +0.29% | +0.15% | 26s | 25.0 |
| puma8NH | +0.12% | +0.63% | 7s | 11.0 |
| ames | +0.00% | +0.00% | 52s | 0.0 |
| page_blocks | +0.00% | +0.00% | 5s | 0.0 |
| cpu_act | +0.00% | +0.00% | 13s | 0.0 |
| house_16H | +0.00% | +0.00% | 14s | 0.0 |
| titanic | +0.00% | -1.30% | 6s | 0.0 |
| sleep | +0.00% | +0.00% | 22s | 0.0 |
| spambase | +0.00% | +0.00% | 41s | 0.0 |
| fars | -0.09% | +0.00% | 75s | 15.7 |

Mean +3.44% (first version +0.99%), 29/60 runs improved, 2/60 worse (by at most
0.28%), mean search time 30 s. The big gains come from
importance-weighted kNN target and per-class distance features (Pol, Magic,
Phoneme, Covertype), rotated coordinates (Houses) and categorical / binned
interactions (Adult, Covertype). Wine White's gain is concentrated in one
split (+16.5%; the other two returned raw) and comes from out-of-fold target
encodings of exact value triples, which exploit the many duplicate rows in
that dataset. On 8 datasets the gate found no held-out gain and returned the
raw features unchanged.

### Previous genetic `FeatureGenerator` (`mode="lite"`, same protocol, seed 0)

| Dataset | Lift | FE time | Features kept |
|---|---|---|---|
| magic | +0.00% | 349s | 0 |
| adult | +1.02% | 307s | 8 |
| houses | −0.54% | 808s | 1 |
| fried | +0.00% | 442s | 0 |
| churn | +0.00% | 997s | 0 |
| phoneme | +0.00% | 402s | 0 |

Its configured 300 s budget is not enforced. On the same seed-0 splits
FeatureForge scored Magic +9.75% (12 s), Adult +4.18% (23 s), Houses +2.81%
(8 s), Fried +0.60% (7 s), Churn −2.74% (23 s) and Phoneme +0.00% (3 s).

### ContestSolver vs. the repo's fixed XGBoost learner (8 classification datasets × 3 holdouts)

Logloss lift of the OOF hill-climbed LightGBM + XGBoost + CatBoost ensemble:
mean +6.7%, median +6.2%, better on 24/24 holdouts (Adult +0.7% … Ring +13.4%).

## Real contest tables (FeatureForge v9)

`python scripts/bench_fe.py --suite contest` runs the same protocol on 22 real
competition tables pulled from OpenML (`CONTEST` in
`tabularaml/benchmarks/contest_suite.py`): Kaggle competitions (Amazon access,
Porto Seguro, Allstate, Give Me Some Credit, Kick, Bike Sharing, House Prices,
Mercedes), KDD Cup 2009, and the original datasets behind Kaggle Playground
episodes (bank marketing → S5E8, bank churn → S4E1, abalone → S4E4, steel plates
→ S4E3, diamonds → S3E8, ...). Tables are used in full up to 60k rows (Porto
Seguro 100k); Kaggle's own Playground files need a Kaggle API token and are not
included yet.

### What v9 adds

* **Cross-linear stacking** (`CrossLinearOOF`): an L2 logistic / ridge model on
  one-hot keys and *all pairwise key crosses*, out of fold, handed to the trees
  as one column. Solved in the dual when the design is wider than tall.
* **High-cardinality recode**: categoricals with > 32 levels can be given to the
  model as frequency ranks instead of native categorical splits; kept only if
  CV *and* the gate prefer it (the gate baseline is always the raw native frame).
* **Key × numeric-band target maps** (`KeyBinTE`), **target encoding of numeric
  values**, **digit / fractional-part features**, **geo features** (haversine /
  Manhattan distance and bearing between lat/lon pairs found by name, rotated
  coordinates, map-only kNN target), **trend features** over ordered column
  families (`PAY_0..PAY_6`: slope, first − last, count positive).
* **Leak fix**: round 2+ no longer builds group statistics over out-of-fold
  target features (other rows' encodings carry this row's label). PR #1's gate
  rejected those rounds, so its reported holdout numbers were not inflated, but
  the leak blocked useful round-2 features.
* **Faster gate**: the gate reuses the search-time feature values instead of
  refitting every target feature again.

### Results, 22 tables × 3 outer 20% holdouts

Lift = % reduction of holdout loss of 5-fold bagged LightGBM on engineered vs.
raw features (positive = better). Each version was run from its own commit.
Raw CSV: `docs/results/fe_contest_bench.csv`.

| Version | Mean lift | Runs better / worse (of 66) | Mean FE time |
|---|---|---|---|
| PR #1 @ a8fbdb1 (branch point) | +1.02% | 24 / 2 | 67s |
| v9 (this branch, before merging PR #1) | +1.88% | 36 / 1 | 51s |
| v9 + PR #1 v11 (tree-path / FAST interactions) | +2.10% | 38 / 1 | 81s |
| v9 + PR #1 v13 (final; this branch) | +2.00% | 35 / 1 | 93s |

| Dataset | Train rows | Metric | This branch: lift, mean of 3 (min … max) | FE time | Features kept | PR #1 @ a8fbdb1 | v9 | v9 + PR #1 v11 |
|---|---|---|---|---|---|---|---|---|
| amazon | 26215 | logloss | **+13.11%** (+10.5 … +15.8) | 56s | 55.3 | +10.15% | +12.95% | +13.21% |
| kdd_appetency | 40000 | logloss | **+5.87%** (+5.1 … +6.4) | 247s | 16.0 | +0.31% | +5.74% | +5.56% |
| kdd_upselling | 40000 | logloss | **+4.71%** (+4.4 … +5.1) | 245s | 19.3 | +0.26% | +4.95% | +5.02% |
| kick | 58386 | logloss | **+4.44%** (+3.9 … +5.0) | 319s | 44.0 | +2.65% | +4.26% | +4.46% |
| food_delivery | 36360 | rmse | **+3.70%** (+3.4 … +4.0) | 56s | 17.0 | +2.64% | +3.67% | +3.70% |
| airline_satisfaction | 48000 | logloss | **+3.48%** (+0.0 … +5.3) | 103s | 13.3 | +1.87% | +1.86% | +4.86% |
| miami_housing | 11020 | rmsle | **+2.51%** (+0.0 … +3.8) | 30s | 7.3 | +0.01% | +2.50% | +2.51% |
| diamonds | 43152 | rmsle | **+1.91%** (+1.8 … +2.0) | 67s | 5.7 | +1.95% | +1.72% | +1.91% |
| abalone | 3340 | rmsle | **+1.83%** (+1.0 … +2.4) | 5s | 2.3 | +1.15% | +1.15% | +1.83% |
| bank_marketing | 36168 | logloss | **+0.91%** (+0.0 … +2.7) | 62s | 3.7 | +1.18% | +1.11% | +1.39% |
| allstate | 48000 | rmsle | **+0.83%** (+0.6 … +1.0) | 167s | 1.0 | +0.00% | +0.83% | +0.83% |
| bank_churn | 8000 | logloss | **+0.39%** (+0.0 … +0.6) | 19s | 6.7 | +0.00% | +0.00% | +0.39% |
| porto_seguro | 80000 | logloss | **+0.31%** (+0.0 … +0.7) | 318s | 0.0 | +0.00% | +0.31% | +0.31% |
| bike_sharing | 13903 | rmsle | **+0.10%** (+0.0 … +0.3) | 21s | 0.7 | +0.10% | +0.10% | +0.10% |
| mercedes | 3367 | rmse | **+0.03%** (+0.0 … +0.1) | 29s | 0.0 | -0.05% | -0.04% | +0.03% |
| coupon | 10147 | logloss | **+0.00%** (+0.0 … +0.0) | 9s | 0.0 | +0.00% | +0.00% | +0.00% |
| credit_default | 24000 | logloss | **+0.00%** (+0.0 … +0.0) | 81s | 0.0 | +0.00% | +0.00% | +0.00% |
| give_me_credit | 48000 | logloss | **+0.00%** (+0.0 … +0.0) | 86s | 0.0 | +0.00% | +0.00% | +0.00% |
| hr_analytics | 15326 | logloss | **+0.00%** (+0.0 … +0.0) | 13s | 0.0 | +0.00% | +0.00% | +0.00% |
| house_prices | 1168 | rmsle | **+0.00%** (+0.0 … +0.0) | 48s | 0.0 | +0.24% | +0.24% | +0.22% |
| steel_plates | 1552 | logloss | **+0.00%** (+0.0 … +0.0) | 32s | 0.0 | +0.00% | +0.00% | +0.00% |
| click | 31958 | logloss | **-0.04%** (-0.1 … +0.0) | 32s | 1.7 | +0.02% | +0.02% | -0.04% |

The big gains are where tables carry interaction structure a GBDT approximates
poorly: many-level categorical crosses (Amazon +13%, KDD Cup +5–6%, Kick +4.4%),
geography (food delivery +3.7%, Miami housing +2.5%) and three-way rating
crosses (airline satisfaction, from PR #1's mined triple counts and encodings).
Six tables (coupon, credit default, Give Me Some Credit, HR analytics, steel
plates, click) show no held-out gain from any candidate family yet.

PR #1's v12/v13 additions (rank-gauss weighted kNN, same-scale column sums) are
neutral on these tables (+2.00% vs +2.10% for v11, within split-to-split noise;
airline −1.4 points, KDD appetency +0.3) and add ~12 s per table; they are kept
because they carry PR #1's public-benchmark gains. A tree-path interaction
search written on this branch independently of PR #1's (+1.96%, 3 worse runs)
was dropped in favour of PR #1's.
