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
