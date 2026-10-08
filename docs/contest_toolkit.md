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

### v14: residual features, contest setting, faster search (2026-10-05)

| Version | Mean lift | Runs better / worse (of 66) | Mean FE time |
|---|---|---|---|
| v9 + PR #1 v13 | +2.00% | 35 / 1 | 93s |
| **v14** (`--transductive`) | **+2.18%** | 35 / 1 | **85s** |

What changed:

- **Linear-residual features** (`LinResid`): the part of a column the other
  top numerics do not explain (raw and log scale). Abalone +1.8% → +4.0%.
- **Contest setting** (`bench_fe.py --transductive`): FeatureForge gets the
  test rows' features (never labels), so counts and group statistics cover
  train + test, both while searching and in the final features. Amazon and
  KDD Cup gain about +0.3 to +1.0 points.
- **Speed:** key codes are cached per frame, which makes group statistics about
  5× faster. kNN switches to brute force past 5 dimensions. Porto Seguro
  320s → 140s, Kick 350s → 280s.

Measured and **not** adopted (each run on 9–15 tables × 3 holdouts, same
protocol):

| Idea | Result |
|---|---|
| Genetic interaction search (`evolve_time=20`): target-encoded column sets + depth-3 arithmetic trees evolved under the novel residual gain | +2.75% vs +2.86% on 12 tables, 26% slower (synthetic compound ratio: +27% → +36%) |
| OpenFE (ICML 2023) via the repo adapter | crashes on 13 of 22 tables; worse on 5 of 9 that ran |
| Wider search (2× seeds, 4 rounds, 80 features) | +2.36% vs +2.34%, 35% slower |
| Looser novelty filter, no gate | no gain; without the gate house prices −8% |
| Gate averaged over 3 models | +2.25% vs +2.30% |
| Composite-key group statistics (`n_composite`) | +2.02% vs +2.05% (kept, off by default) |
| Gate z ≥ 0 on large gates | +2.01% vs +1.96%, more losing runs |
| Denoising-autoencoder code, OOF MLP prediction, permutation-based column drop | no gain on the zero tables (MLP: steel plates −2 to −5%) |

**Downstream AutoGluon** (`--judge autogluon`, medium_quality, 120s per fit):
on 8 tables × 2 holdouts, PR #2's features gave +1.5% (13 of 16 runs better).
The LightGBM judge gave +4.3% on the same runs. With the contest setting, a
partial 22-table run (33 pairs) gave +0.95% mean, with 18 pairs better and 4
worse: Amazon +4.3%, diamonds +2.6%, Kick +2.4%, bank marketing +2.3%, food
delivery +2.1%.

## v15: hidden entities and a time-ordered gate (IEEE-CIS Fraud Detection)

Tested on a real paid competition: IEEE-CIS Fraud Detection (Kaggle 2019, $20k),
most recent 236k of 590k transactions, holdout = the latest 20% by time (the
competition's test set was later in time too). Judge: one LightGBM; reproduce
with `python scripts/ieee_fraud.py`.

| Columns | Holdout AUC | Log loss |
|---|---|---|
| Raw | 0.9338 | 0.0886 |
| FeatureForge v14 (random gate) | 0.9207 | 0.1022 |
| FeatureForge v14 + time-ordered gate | 0.9338 (rejects everything) | 0.0886 |
| FeatureForge v15 (entities + time-ordered gate) | **0.9522** | 0.0910 |
| Raw + v15's anchors and entity target maps only | 0.9553 | 0.0823 |

* **Hidden entities** (`entities=True`, default). A timestamp-like column minus a
  "days since X" column is constant per customer (the account-opening day);
  FeatureForge finds such anchors by checking that, within ID-like columns,
  `t - delta` takes clearly fewer distinct values than the control `t + delta`.
  On IEEE it finds `floor(TransactionDT / 86400) - D1` (also D15, D10) and
  ID columns card1, addr1, card2 by itself: the winning team's "UID". Anchors
  are added as columns, and counts, target maps and group statistics over
  ID x anchor composites join the candidates.
* **Time-ordered gate** (`time_col="auto"`). When the unlabeled rows lie beyond
  the training range of a column, the gate holds out the latest training rows
  instead of a random sample. Without it, kNN target features and out-of-fold
  models that only work within a period passed the gate and lost 1.3 AUC points
  on the later holdout.
* Benchmark tables (random holdouts, 15 tables with ID columns, 3 holdouts each):
  neutral, +3.41% vs +3.46% for v14 (all within ±0.1% per table except sf_crime
  0.25% -> 0.11% and kdd_upselling 5.2% -> 5.6%). Tables without ID columns are
  unchanged; the time gate never triggers on random splits.
* Also tried on IEEE (not adopted): 16 instead of 6 numerics aggregated per
  entity (AUC 0.9529 vs 0.9522, 25% slower; `entity_nums`); each row's place in
  its entity's history, i.e. time since previous / until next transaction and
  rows before it (0.9555 vs 0.9550 on top of the entity features, log loss
  worse; `entity_lags=True`). The search now screens candidates as it builds
  them, so only promising columns stay in memory (peak 6.1 -> 5.0 GB on 94k x
  430; the rest is the frame copies).
* Gap left on the table: raw + anchors + entity target maps alone score 0.9550;
  the full selection adds out-of-fold kNN and linear features that the gate's
  model likes but the judge's deeper trees do not carry to later rows.
* Confirmed on three time windows of the IEEE data (each 40% of the 590k rows,
  holdout = latest 20% of the window):

  | Window ends at | Raw AUC | v15 AUC | Raw log loss | v15 log loss |
  |---|---|---|---|---|
  | 40% | 0.9229 | 0.9461 | 0.0977 | 0.0960 |
  | 70% | 0.9334 | 0.9487 | 0.0987 | 0.0947 |
  | 100% | 0.9338 | 0.9522 | 0.0886 | 0.0910 |
  | mean | 0.9300 | **0.9490 (+1.9 pts)** | | -1.0% |

## Related tables: Home Credit Default Risk (Kaggle 2018, $70k)

`tabularaml.generate.relational.RelatedTables` aggregates child tables (and
their children) onto the main table: within-row differences and ratios of
same-unit columns (days late = paid day minus due day, paid / owed), then per
entity counts, mean / max / min / sum / std, category shares and distinct
counts, and means over the most recent rows. Label-free; FeatureForge selects
and searches interactions on top.

Home Credit, application table (307k loans), random stratified 80/20 holdouts,
one LightGBM judge, AUC:

| Columns | Holdout 0 | Holdout 1 | Mean |
|---|---|---|---|
| Raw application table | 0.7653 | 0.7659 | 0.7656 |
| + FeatureForge | 0.7772 | 0.7757 | 0.7764 |
| Winners' hand-made application features (reference) | 0.7744 | 0.7761 | 0.7753 |
| + RelatedTables (5 child tables, 1,055 columns, 3.5 min) | 0.7946 | 0.7974 | **0.7960** |
| + RelatedTables top 200 + FeatureForge | 0.7947 | 0.7996 | 0.7971 |

On the main table alone FeatureForge matches the hand-made features the
winners used (credit / annuity is its first pick); the side tables are where
the remaining gain is.

Follow-ups on Home Credit (one-model judge, same 2 holdouts, AUC):

| Columns | Holdout 0 | Holdout 1 | Mean |
|---|---|---|---|
| App + RelatedTables (with time-column pairs) | 0.7964 | 0.7988 | 0.7976 |
| + status-split aggregates (`n_split=2`, 1,962 columns) | 0.7958 | 0.8004 | 0.7981 (noise; opt-in) |
| + **child-row models** (`child_model_features`) | 0.7990 | 0.8019 | **0.8004** |

`child_model_features` labels every child row (past loan, payment, card month)
with its parent's target, fits a LightGBM on child rows with folds split by
*parent*, and aggregates the out-of-fold predictions per parent (mean, max,
min, std, most recent). It adds +0.28 AUC points on both holdouts over the
aggregates and costs about 14 minutes for the five Home Credit tables. Total on
Home Credit: 0.7656 -> 0.8004.

## Column families: Santander Customer Transaction Prediction (Kaggle 2019, $65k)

200,000 labelled rows of 200 anonymous numeric columns (`var_0` .. `var_199`).
The winning insight was that whether a value *repeats* in its column carries
the signal, once the synthetic half of the test set is set aside. FeatureForge
now finds this on its own: for any family of 8+ similarly named numeric
columns it proposes two blocks, label-free, computed over every row whose
features are known (training rows plus real test rows):

- `famcount__*`: how often each row's value occurs in its column;
- `fammask__*`: the value where it repeats, NaN where it is unique.

Each column alone adds little, so a block is screened by the sum of its
columns' novel gains. Protocol: two stratified 80/20 holdouts; the unlabelled
rows given to FeatureForge are the holdout plus the 100,000 real test rows
(synthetic test rows, those with no value unique to the test set, are dropped);
one LightGBM judge; AUC.

| Columns | Holdout 0 | Holdout 1 | Mean |
|---|---|---|---|
| Raw columns | 0.8958 | 0.8954 | 0.8956 |
| FeatureForge before family blocks | 0.8959 | 0.8948 | 0.8954 |
| Hand-built counts (reference) | 0.9053 | | |
| Hand-built counts + masks (the winners' features, reference) | 0.9181 | | |
| **FeatureForge with family blocks** | 0.9183 | 0.9183 | **0.9183** |

FeatureForge matches the winners' hand-built features with no hints, and log
loss drops 11%. About 1,300 s; about 400 columns added. On the benchmark
tables where a family can trigger (Allstate, KDD Cup appetency and upselling,
Fried, Pol) results are unchanged: the gate does not select the blocks there.

Tried and left opt-in (`family_nb=True`): per-column out-of-fold target maps
over (value band, value count) plus their sum, a naive-Bayes score. With the
family blocks already present it lowered holdout 0 from 0.9183 to 0.9138.

## Dates, events and drift: Rossmann Store Sales (Kaggle 2015, $35k)

1,115 stores, daily sales, a store table and a 48-day test period after the
training data. Protocol: two time-ordered holdouts of the last 6 weeks
(window 0 ends 2015-07-31, window 1 six weeks earlier); training rows are
open days with sales, all earlier; FeatureForge also gets every row whose
features are known (closed days, the holdout, the Kaggle test rows) as
unlabeled rows; `Date` is passed as a date. Judge: one LightGBM on log sales,
early-stopped on the latest 6 weeks of training. RMSPE (the contest metric):

| Columns | Window 0 | Window 1 |
|---|---|---|
| Raw columns (store table joined, date as a day number) | 0.1288 | 0.1349 |
| Raw + standard date fields | 0.1476 | 0.1258 |
| Hand-made: date fields + days since / until promo, holiday, closure (reference) | 0.1140 | 0.1220 |
| FeatureForge v15 | 0.1927 (worse; gate passed) | 0.1349 (gate rejected) |
| FeatureForge v17 | 0.1189 | 0.1279 |
| **FeatureForge v18** (nested target-encoding CV under time-blocked search) | **0.1201** | **0.1205** |

What v17 adds, all generic:

- **Dates.** Datetime and ISO date-string columns become day numbers, and
  calendar fields (weekday, day of month, month, year, day / week of year,
  days to month end) become candidates.
- **`EventRecency`.** With a time column: time since an entity's last row with
  a given level of a low-cardinality column, and until its next one (days since
  the store last closed, until the next state holiday, since the last promo).
  Label-free, over every known row. Its novelty is measured against the flag
  column, not the entity key: the key's own signal (a store's level) is not what
  the feature re-expresses, and measuring against it hid every event feature.
- **Time-blocked search.** When a time column is known, the search CV uses
  contiguous time blocks as folds, and screening is measured on the latest rows.
  Random folds let features that interpolate between neighbouring days win
  (v15's search CV read 0.011 against 0.035 to 0.046 on later rows).
- **Gate sized to the test horizon.** The time-ordered gate holds out the latest
  rows covering the unlabeled rows' horizon, not 20% of the rows. On Rossmann,
  20% was the 6 months after Christmas, where every raw model carries the
  December peak forward; the gate then preferred sets that were worse later.
  Early stopping inside the gate is time-ordered too.
- **No target maps over time.** Target-dependent candidates keyed on the time or
  a date column are not generated (v15 shipped `te__Date`); label-free counts
  and group statistics per date still are.
- `EntityLag` gains past / future window counts (rows of the entity within
  0.1%, 1% and 5% of the time span), opt-in with `entity_lags=True`.

v17 cost IEEE-CIS: its latest window read 0.9479 with time-blocked search
against 0.9522 before. The cause was target encodings fitted on random folds
inside a time-blocked CV: training rows' encodings carried the validation
block's labels. v18 recomputes them inside each fold whenever time-blocked
search is on (`nested_cv="auto"`): IEEE-CIS latest window **0.9531**, Rossmann
0.1201 / 0.1205 (mean 0.1203 vs 0.1234 for v17).

Tried and left opt-in (`lagged_te=True`): per-store and store x flag mean
target over a trailing window ending one test horizon before each row. On top
of the hand-made features it made Rossmann worse (0.114 -> 0.124, 0.122 ->
0.146): a full-horizon lag is too stale. Without time-blocked search, Rossmann window 1 ships a numeric target
encoding of the date and lands at 0.172, so it stays on. The 7 structured
benchmark tables (random holdouts, no time column) are unchanged.

## Order histories: Instacart Market Basket Analysis (Kaggle 2017, $25k)

Which previously bought products does a user reorder in their next order?
Rows are (user, product) pairs from the user's prior orders, labelled by the
user's next ("train") order. Protocol (`scripts/instacart_bench.py`, data from
the Hugging Face mirror `attik/Instacart-Market-Basket-Analysis` as parquet):
15,000 users sampled from those with a train order (≈ 950k pairs, 10%
positive); holdout = 20% of users, two seeds; one LightGBM judge with
early stopping on held-out users. Leak check: every feature comes from prior
orders only, never from the labelled order's items; holdouts and early
stopping split by user.

| Columns | Holdout 0 | Holdout 1 |
|---|---|---|
| Raw pair columns (ids, aisle, department, next order's weekday / hour / days since prior) | 0.6748 | 0.6740 |
| FeatureForge on the pair table alone | 0.6790 | 0.6765 |
| **+ `RelatedTables`**, default settings (prior order lines by user x product, orders by user, order lines by product) | **0.8285** | **0.8286** |
| + `RelatedTables` + FeatureForge v18 | 0.8269 | 0.8252 |
| **+ `RelatedTables` + FeatureForge, grouped validation** | **0.8316** | **0.8310** |

`RelatedTables` with no settings beyond naming each child table's key and time
column takes AUC from 0.674 to 0.829 (log loss 0.304 -> 0.250). FeatureForge
on top makes it slightly worse: its gate and folds split rows at random, so
target maps over `user_id` looked useful on users the search had seen, while
the holdout users are new.

## Blind run: Corporación Favorita Grocery Sales Forecasting (Kaggle 2018, $30k)

Run with the shipped defaults and no contest-specific settings (the
generalization check). Slice: stores 44, 45 and 47, 2017-05-01 to 2017-08-15,
zero-filled store x item x day grid (1.1M rows), item and store tables joined,
`date` passed as a date. Holdout: the last 16 days (the contest's horizon) and
the 16 days before; unlabeled rows = every row from the holdout start on.
Target log1p(sales); metric NWRMSLE (perishables weighted 1.25), lower is
better; one LightGBM judge early-stopped on the latest 16 training days.

| Columns | Window 0 | Window 1 |
|---|---|---|
| Raw columns (date as a day number) | 0.8913 | 0.8761 |
| Hand-made: store x item mean log sales over the last 7 / 14 / 28 / 56 days, lagged 16 days (reference) | 0.7545 | |
| FeatureForge, defaults (blind) | 0.7431 | 0.7782 |
| **FeatureForge, defaults, day-level dates detected as time** | **0.7488** | **0.7153** |

FeatureForge cuts the error by 17% and 11% and beats the hand-made recent-sales
features; its top picks are target maps of item and class, promotion
deviations per item and calendar fields.

The blind run exposed a detection gap: `time_col="auto"` required 5% distinct
values, so a day-level date shared by thousands of rows was never recognised
as time. Date columns now qualify at any granularity (other numerics need 20+
levels and test values beyond 98% of training). With the time machinery on,
Favorita's mean error drops further, 0.761 -> 0.732.

## Blind run: Home Credit - Credit Risk Model Stability (Kaggle 2024, $105k)

Public processed subset (Hugging Face `deburky/home-credit-credit-risk-model-stability`:
522k loans, 43 columns already aggregated from the bureau tables, weeks
50-91). Holdouts: the last 8 weeks and the 8 before, training on all earlier
weeks. Defaults, no settings.

| Columns | AUC, window 0 | AUC, window 1 |
|---|---|---|
| Raw | 0.8267 | 0.8067 |
| FeatureForge, defaults (time not detected) | 0.8261 | 0.8075 |
| FeatureForge, defaults (date detected as time) | 0.8267 (gate rejected: raw) | 0.8067 (gate rejected: raw) |

Neutral: on this pre-aggregated subset nothing FeatureForge proposes carries
to later weeks, and the time-ordered gate returns the raw columns.

## Blind run: M5 Forecasting - Accuracy (Kaggle 2020, $50k)

`scripts/m5_bench.py` (data: Hugging Face `denephew/M5_Forecasting`). One
store (CA_1, 3,049 items), the 150 days before each holdout, calendar
(events, SNAP) and weekly prices joined; holdout = the last 28 days (the
contest horizon) and the 28 before. Metric: RMSSE averaged over items (lower is
better); one Tweedie LightGBM judge early-stopped on the latest 28 training
days. Defaults, no settings; `date` is detected as time.

| Columns | Window 0 | Window 1 |
|---|---|---|
| Raw | 0.7821 | 0.7537 |
| **FeatureForge, defaults (blind)** | **0.7756** | **0.7200** |

-0.8% and -4.5%. The picks are item-level target maps by price band and
week. Columns whose test values are all new (week numbers beyond training)
are now excluded from target maps like the time column; on M5 the week column
overlaps the holdout's first week, so nothing changed. Trailing-window target
means (`lagged_te=True`) were not selected.

## Free text: Mercari Price Suggestion (Kaggle 2018, $100k)

`scripts/mercari_bench.py` (data: Hugging Face
`multabench/core-text-reg-mercari-marketplace`, 100k listings of the contest's
training set). Random 80/20 holdouts, as the contest's test set; the holdout's
features are the unlabeled rows. Metric: RMSLE of price (lower is better). The
judge gets string columns as categoricals.

| Columns | Seed 0 | Seed 1 | FE time |
|---|---|---|---|
| Raw | 0.5714 | 0.5754 | |
| FeatureForge before text features (blind) | 0.5635 | 0.5754 (gate rejects) | 90-140 s |
| **FeatureForge with text features (defaults)** | **0.4719** | **0.4740** | 550-575 s |

-17.4% and -17.6%. Text columns are detected from the data (`text=True`): a
string column of mostly distinct values written as several space-separated
words (here `name` and `item_description`; not codes such as
`AGRRES14DEL01` or category labels). Each gets:

- `TextStats`: characters, words, digit / capital / punctuation shares,
  distinct-word share, emptiness (label-free);
- `TextSVD`: 16 TF-IDF (words and word pairs) SVD components, fitted on train
  plus unlabeled rows (label-free);
- `TextLinearOOF`: an out-of-fold ridge / logistic model on word 1-2-gram and,
  for short texts, character 2-4-gram TF-IDF; plus one over all text columns
  with one-hot keys (brand, category, condition), the sparse linear model behind
  the Mercari winners' solutions, as one column.

The gate kept all three kinds on both holdouts; the all-text model ranks first.
Text columns are kept out of keys, ids and group detection.

## Porto Seguro Safe Driver Prediction (Kaggle 2017, $25k)

`scripts/porto_bench.py` (data: OpenML 42742, the full 595k-row training set).
Stratified 80/20 holdouts; normalized Gini (higher is better).

| Columns | Seed 0 | Seed 1 |
|---|---|---|
| Raw | 0.2594 | 0.2761 |
| Hand-made (missing count, ps_car_13 x ps_reg_03, calc columns dropped) | 0.2607 | 0.2786 |
| FeatureForge, defaults | **0.2648** | 0.2761 (gate rejects) |

Small, as on the capped suite copy (+0.31% logloss): the anonymised columns
carry little feature-engineering signal; the winners' margin came from
denoising-autoencoder networks (modelling, out of scope; DAE features were
neutral here earlier). On seed 0 the gain is the high-cardinality recode
alone (no columns added).

On the structured suite's sf_crime (the only suite table with a detected text
column, `Address`: "800 Block of BRYANT ST"), text features turn a gate
rejection into a held-out logloss gain on all three holdouts: 0.6700 / 0.6654 /
0.6676 to 0.6655 / 0.6623 / 0.6630 (-0.6%). No other suite or contest table
has a text column, so they are unchanged.

## M5: room left versus classic hand-made features

`scripts/m5_bench.py --arm hand`: each item's sales means over 7 / 28 / 56 / 112
days and std over 28 days, all ending 28 days before the row, relative price and
price momentum (the public M5 kernels' features).

| Columns (RMSSE, lower is better) | Window 0 | Window 1 | Mean |
|---|---|---|---|
| Raw | 0.7821 | 0.7537 | 0.7679 |
| Hand-made | 0.7711 | 0.7377 | 0.7544 |
| Hand-made + FeatureForge | 0.7734 | 0.7422 | 0.7578 |
| **FeatureForge alone, defaults (blind)** | 0.7756 | **0.7200** | **0.7478** |

FeatureForge alone already beats the classic hand-made features on average, so
those leave no room to automate on this slice.

### Text features on public text-tabular tables

`scripts/text_bench.py` on MulTaBench tables (Hugging Face `multabench/*`; up to
50k rows; two random 80/20 holdouts; RMSE or logloss, lower is better). Blind =
shipped defaults before text features. Mean of the two holdouts:

| Table | Metric | Raw | Blind | Text features | Change vs raw |
|---|---|---|---|---|---|
| fake-job-posting | logloss | 0.154 | 0.154 | **0.070** | -55% |
| wine-review (variety, 30 classes) | logloss | 1.648 | 1.648 | **0.641** | -61% |
| women-clothing-review (rating) | logloss | 0.894 | 0.894 | **0.689** | -23% |
| book-price | RMSE | 0.289 | 0.289 | **0.228** | -21% |
| data-scientist-salary | logloss | 1.319 | 1.319 | **1.149** | -13% |
| zomato-restaurants (rating) | RMSE | 0.119 | 0.090 | **0.085** | -29% (blind -24%) |
| kickstarter-funding | logloss | 0.587 | 0.587 | **0.515** | -12% |
| rotten-tomatoes | RMSE | 0.955 | 0.955 | 0.955 (gate rejects) | 0% |

Seven of eight tables gain; none got worse. Text fields are cut to their first 1,500 characters before
n-gram vectorising (`TEXT_CLIP`): zomato's 11,700-character review dumps took
43 minutes uncut (0.0829 on the first holdout) and 22 minutes cut (0.0849).
Runtime is the cost: 1-4 minutes on most tables, 22 minutes on wine-review (30
classes make every later model 30 times larger) and zomato.

## Blind run: Zillow Prize (Kaggle 2017-18, $1.2M total)

`scripts/zillow_bench.py` (data: Hugging Face `Kun-05/ML-Zillow-Prize`, the
contest zip). 168k sales of 2016-17 with that year's property table joined;
target logerror; holdouts = 2017-07 to 09 and 2017-04 to 06, training on all
earlier sales; MAE (lower is better).

| Columns | Window 0 | Window 1 |
|---|---|---|
| Predict the training median | 0.06971 | 0.06857 |
| Raw | 0.06908 | 0.06809 |
| **FeatureForge, defaults (blind)** | **0.06889** | **0.06784** |

-0.3% and -0.4%: about a third of what the raw model gains over a constant.
logerror is mostly noise (Zillow's own model already used these columns); the
contest was decided by fractions of a percent. Picks: kNN target means over
the coordinates and over the top numerics, key-pair target maps.

## Blind run: West Nile Virus Prediction (Kaggle 2015, $40k)

`scripts/wnv_bench.py` (data: GitHub `apnorton/ml-project`, the contest files).
Trap tests with station-1 weather joined by date; NumMosquitos dropped (absent
from the test set). Holdouts: 2013 (training on 2007-11) and 2011 (training on
2007-09); AUC.

| Columns | 2013 | 2011 |
|---|---|---|
| Raw | 0.698 | 0.687 |
| Hand-made: rows per (date, trap, species) | 0.702 | 0.704 |
| Hand-made: week and day of year | 0.713 | 0.704 |
| Hand-made: trailing 7/14/28-day weather means | 0.630 | 0.670 |
| Hand-made: all three | 0.617 | 0.677 |
| **FeatureForge, defaults (blind)** | **0.754** | 0.687 (gate rejects) |

On 2013 FeatureForge beats every hand-made set (picks: counts per date x heat
x species, precipitation spread per weather code). On 2011 (two training
years) the gate finds nothing that carries over. Trailing weather means hurt
with three or fewer training seasons.

## Event logs: Data Science Bowl 2019 (Kaggle, $160k)

`scripts/dsb_bench.py` (data: Hugging Face `pytorch-lifestream/datascience-bowl2019`).
Main rows: the 17,690 labelled assessments (installation, title, world, start
time); child: 11.3M game events (event_data JSON left out). Holdout: 20% of
installations (the contest's test children are new), two seeds. Metric:
quadratic weighted kappa with thresholds matching the training class shares.

| Columns | Seed 0 | Seed 1 | Mean |
|---|---|---|---|
| Raw (assessment title, world, start) | 0.431 | 0.431 | 0.431 |
| FeatureForge on raw (blind) | 0.431 (gate rejects) | 0.431 | 0.431 |
| As-of event aggregations | 0.544 | 0.576 | 0.560 |
| **As-of + FeatureForge** | **0.583** | **0.604** | **0.594** |

`asof_features` (tabularaml/generate/relational.py) aggregates a child event
table as of each main row: only the same key's events strictly before the
row's time. Count, time since first / last event, mean / sum / max / min / std
of numerics and their mean over the last 5 / 20 events, shares of frequent
levels (and counts over the last 5), distinct levels; prefix sums over the
child sorted by (key, time), 60 s for 11.3M events. `scripts/contest_features.py`
switches it on by itself when the main table has several rows per child key and
a time column comparable to the child's. FeatureForge then detects the new
installations (grouped validation) and adds differences of event-code shares
(attempts minus completions) and title-conditioned target maps of them.
