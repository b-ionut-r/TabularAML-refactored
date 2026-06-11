# Competition-grade feature search

The genetic feature search defaults are tuned for ML competitions
(Kaggle / CrunchDAO / DrivenData): every acceptance decision is guarded
against CV overfitting, throughput is spent on candidates that matter, and
era-structured data is handled natively.

## What runs by default

| Mechanism | Parameter(s) | Effect |
|---|---|---|
| Statistical acceptance | `acceptance="statistical"`, `acceptance_folds_frac=0.7` | A candidate must clear the adaptive mean-gain threshold AND improve ≥ ceil(0.7·K) of K paired CV folds (sign test). `acceptance="mean"` restores the old rule. |
| Generation confirmation | `confirmation_seeds=1` (2 in best/extreme presets) | An improving generation is re-tested new-vs-previous-best under alternate CV seeds before being committed; unconfirmed improvements are reverted. |
| Null-importance selection | `null_importance_selection=True`, `null_importance_n_perm=4`, `null_importance_pct=75` | Post-search, generated features (and base-expansion outputs) must beat the 75th percentile of their own target-permutation importance distribution. |
| Expansion block gate | automatic | Datetime-part and row-stat blocks are kept only if they don't degrade the paired-fold baseline at search start. |
| Two-stage batched proxy | `proxy_mode="batched"`, `proxy_ram_budget_mb=512` | One residual-boosting LightGBM over all candidates coarsely filters the batch; per-candidate FeatureBoost refines the survivors. `"featureboost"` / `"none"` restore old behaviors. |
| Base-table expansion | `expand_datetime=True`, `expand_row_stats=True` | Datetime columns are decomposed (year/month/dow/hour/weekend/cyclical/epoch-days) and row stats (mean/std/max/min/NaN-count) join the base table before the search. |
| Global transforms | op family `"global"` | `rank_pct`, `qbin`, `zscore_winsor`, `log_rank` — fitted on train folds inside the pipeline, leakage-free and batch-independent at transform time. |
| ES-leak-free CV | `cross_val_score(eval_set_policy="auto")` | When the baseline model uses early stopping, the ES set is carved from the train fold; the validation fold is never seen during fit. |
| Parallel folds | `cv_n_jobs="auto"` | Threads across CV folds with per-fold model-thread clamping. |

## Era mode (CrunchDAO / Numerai-style)

```python
from tabularaml.generate.features import FeatureGenerator
from tabularaml.eval.scorers import PREDEFINED_REG_SCORERS

gen = FeatureGenerator(
    mode="best",
    era_col="moon",                                   # or "era", "date", ...
    scorer=PREDEFINED_REG_SCORERS["era_spearman"],    # or "era_spearman_sharpe"
    task="regression",
)
X_new, pipeline, generation, interactions = gen.search(X, y)
```

With `era_col` set:
- CV becomes era-grouped (`RotatedGroupKFold`; pass a `PurgedTimeSeriesSplit`
  yourself for purged/embargoed evaluation),
- `search_sample_size` samples whole eras, never splitting one,
- fitness can be mean per-era Spearman (`era_spearman`) or its
  stability-rewarding Sharpe variant (`era_spearman_sharpe`),
- acceptance additionally requires a candidate to help in ≥ 55% of eras
  (`era_acceptance_frac`),
- null-importance permutes the target within eras.

Feature-exposure neutralization (per-era correlation of generated features
with the existing feature span) is future work — see
`FeatureGenerator._era_feature_corr_report`.

## Adversarial validation

Pass the unlabeled test features to flag generated features that encode
train-specific structure:

```python
gen.search(X_train, y_train, X_test=X_test)
print(gen.adversarial_report)   # {"auc": ..., "top_shift_features": [...]}
```

AUC above `adversarial_auc_warn` (0.75) logs the shift drivers; set
`adversarial_drop=True` to also drop generated features contributing > 10%
of the shift signal.

## Benchmarking old vs new

`scripts/run_acceptance_ab_benchmark.py` reruns the pre-overhaul
configuration against the current defaults at an identical budget, holdout
split and baseline model, and applies pre-registered criteria (Wilcoxon /
win-rate on held-out gains, overfit-gap reduction, per-dataset regression
check, throughput ratio). Results land in `reports/acceptance_ab.csv` and a
verdict in `reports/acceptance_ab.md`.
