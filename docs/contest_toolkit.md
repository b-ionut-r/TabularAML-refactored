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
| pol | +18.45% | +0.00% | 63s | 22.7 |
| magic | +13.74% | +11.30% | 27s | 23.0 |
| covertype | +6.71% | +0.00% | 95s | 106.3 |
| wine_white | +5.51% | +0.00% | 50s | 54.3 |
| adult | +4.29% | +4.24% | 31s | 13.0 |
| phoneme | +3.79% | +0.00% | 7s | 8.3 |
| houses | +3.58% | +3.62% | 12s | 5.7 |
| churn | +1.87% | -0.36% | 27s | 25.0 |
| fried | +0.82% | +0.56% | 16s | 3.0 |
| wind | +0.68% | +0.95% | 10s | 3.7 |
| coil2000 | +0.29% | +0.15% | 25s | 25.0 |
| puma8NH | +0.27% | +0.63% | 5s | 3.3 |
| ames | +0.00% | +0.00% | 52s | 0.0 |
| page_blocks | +0.00% | +0.00% | 4s | 0.0 |
| cpu_act | +0.00% | +0.00% | 13s | 0.0 |
| house_16H | +0.00% | +0.00% | 11s | 0.0 |
| titanic | +0.00% | -1.30% | 6s | 0.0 |
| sleep | +0.00% | +0.00% | 20s | 0.0 |
| spambase | +0.00% | +0.00% | 36s | 0.0 |
| fars | -0.09% | +0.00% | 73s | 15.7 |

Mean +3.0% (first version +0.99%), 29/60 runs improved, 2/60 worse (Churn
−0.6%, Fars −0.3%), mean search time 29 s. The big gains come from
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
