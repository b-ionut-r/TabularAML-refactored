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
     5–100 nearest rows in standardised subspaces of the top numerics);
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
5. **Selection.** Survivors are ranked by split gain in a joint model; the best
   prefix (3, 6, 12, 25, ...) is chosen by repeated-CV loss.

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

| Dataset | Holdout lift (mean of 3) | FE time | Features kept |
|---|---|---|---|
| magic | +11.30% | 12s | 15.3 |
| adult | +4.24% | 30s | 32.3 |
| houses | +3.62% | 10s | 3.7 |
| wind | +0.95% | 8s | 4.0 |
| puma8NH | +0.63% | 3s | 1.0 |
| fried | +0.56% | 8s | 2.0 |
| coil2000 | +0.15% | 18s | 18.0 |
| ames | +0.00% | 37s | 0.0 |
| page_blocks | +0.00% | 3s | 0.0 |
| cpu_act | +0.00% | 15s | 0.0 |
| fars | +0.00% | 48s | 0.0 |
| covertype | +0.00% | 69s | 0.0 |
| sleep | +0.00% | 12s | 0.0 |
| house_16H | +0.00% | 8s | 0.0 |
| pol | +0.00% | 33s | 0.0 |
| phoneme | +0.00% | 3s | 0.0 |
| wine_white | +0.00% | 9s | 0.0 |
| spambase | +0.00% | 20s | 0.0 |
| churn | -0.36% | 23s | 12.0 |
| titanic | -1.30% | 8s | 2.0 |

Mean +0.99%, 18/60 runs improved, 3/60 worse (Titanic −3.9% on a 712-row
split, Churn −2.7%, Coil2000 −0.3%), mean search time 19 s. Gains concentrate where
the data has structure trees approximate poorly: local neighbourhoods (Magic,
via kNN target features), rotated coordinates (Houses, `latitude ± longitude`),
categorical interactions (Adult). On the other 11 datasets the gate found no
held-out gain and returned the raw features unchanged.

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
raw features (positive = better). "Merged" is this branch: v9 plus PR #1's later
interaction mining (tree-path and FAST pairs/triples, key × binned cells,
PCA/PLS projections, importance-weighted and per-class kNN). "PR #1 @ branch
point" is FeatureForge at commit a8fbdb1. Raw CSV: `docs/results/fe_contest_bench.csv`.

| Dataset | Train rows | Metric | Merged lift, mean of 3 (min … max) | FE time | Features kept | PR #1 @ branch point | v9 |
|---|---|---|---|---|---|---|---|
| amazon | 26215 | logloss | **+13.21%** (+10.4 … +16.0) | 32s | 60.0 | +10.15% (59s) | +12.95% (39s) |
| kdd_appetency | 40000 | logloss | **+5.56%** (+5.0 … +6.6) | 193s | 0.0 | +0.31% (179s) | +5.74% (131s) |
| kdd_upselling | 40000 | logloss | **+5.02%** (+4.6 … +5.3) | 208s | 0.0 | +0.26% (232s) | +4.95% (148s) |
| airline_satisfaction | 48000 | logloss | **+4.86%** (+4.2 … +5.3) | 117s | 18.3 | +1.87% (166s) | +1.86% (95s) |
| kick | 58386 | logloss | **+4.46%** (+4.0 … +4.9) | 262s | 36.7 | +2.65% (264s) | +4.26% (155s) |
| food_delivery | 36360 | rmse | **+3.70%** (+3.4 … +4.0) | 57s | 17.0 | +2.64% (79s) | +3.67% (47s) |
| miami_housing | 11020 | rmsle | **+2.51%** (+0.0 … +3.8) | 34s | 7.3 | +0.01% (14s) | +2.50% (21s) |
| diamonds | 43152 | rmsle | **+1.91%** (+1.8 … +2.0) | 69s | 5.7 | +1.95% (51s) | +1.72% (43s) |
| abalone | 3340 | rmsle | **+1.83%** (+1.0 … +2.4) | 5s | 2.3 | +1.15% (3s) | +1.15% (4s) |
| bank_marketing | 36168 | logloss | **+1.39%** (+0.0 … +2.7) | 59s | 6.7 | +1.18% (35s) | +1.11% (39s) |
| allstate | 48000 | rmsle | **+0.83%** (+0.6 … +1.0) | 140s | 1.0 | +0.00% (70s) | +0.83% (70s) |
| bank_churn | 8000 | logloss | **+0.39%** (+0.0 … +0.6) | 19s | 6.7 | +0.00% (18s) | +0.00% (9s) |
| porto_seguro | 80000 | logloss | **+0.31%** (+0.0 … +0.7) | 257s | 0.0 | +0.00% (94s) | +0.31% (122s) |
| house_prices | 1168 | rmsle | **+0.22%** (+0.0 … +0.6) | 56s | 3.0 | +0.24% (38s) | +0.24% (28s) |
| bike_sharing | 13903 | rmsle | **+0.10%** (+0.0 … +0.3) | 16s | 0.7 | +0.10% (16s) | +0.10% (12s) |
| mercedes | 3367 | rmse | **+0.03%** (+0.0 … +0.1) | 34s | 0.0 | -0.05% (19s) | -0.04% (23s) |
| credit_default | 24000 | logloss | **+0.00%** (+0.0 … +0.0) | 71s | 0.0 | +0.00% (62s) | +0.00% (52s) |
| coupon | 10147 | logloss | **+0.00%** (+0.0 … +0.0) | 9s | 0.0 | +0.00% (8s) | +0.00% (7s) |
| hr_analytics | 15326 | logloss | **+0.00%** (+0.0 … +0.0) | 13s | 0.0 | +0.00% (17s) | +0.00% (11s) |
| give_me_credit | 48000 | logloss | **+0.00%** (+0.0 … +0.0) | 74s | 0.0 | +0.00% (29s) | +0.00% (30s) |
| steel_plates | 1552 | logloss | **+0.00%** (+0.0 … +0.0) | 28s | 0.0 | +0.00% (14s) | +0.00% (20s) |
| click | 31958 | logloss | **-0.04%** (-0.1 … +0.0) | 36s | 1.7 | +0.02% (14s) | +0.02% (17s) |

| Version | Mean lift | Runs better / worse (of 66) | Mean FE time |
|---|---|---|---|
| PR #1 @ branch point | +1.02% | 24 / 2 | 67s |
| v9 | +1.88% | 36 / 1 | 51s |
| Merged (this branch) | +2.10% | 38 / 1 | 81s |

The big gains are where tables carry interaction structure a GBDT approximates
poorly: many-level categorical crosses (Amazon +13%, KDD Cup +5–6%, Kick +4.5%),
geography (food delivery +3.7%, Miami housing +2.5%) and local neighbourhoods
(airline satisfaction +4.9% from PR #1's kNN additions). Six tables (coupon,
credit default, Give Me Some Credit, HR analytics, steel plates, click) show no
held-out gain from any candidate family yet. A tree-path interaction search
written on this branch independently of PR #1's (+1.96%, 3 worse runs) was
dropped in favour of PR #1's version.
