# Changes — FeatureGenerator SOTA gap closure

This round closes four state-of-the-art gaps in the genetic feature engine that
were absent from both the code and the prior `reports/feature_engine_upgrade_plan_full.md`.
All changes are backward compatible: new behavior is opt-in or governed by new
parameters with conservative defaults.

## P2 — Ordered / smoothed target encoding

**Files:** `tabularaml/preprocessing/encoders.py`, `tabularaml/generate/features.py`,
`tabularaml/configs/feature_gen.py`

- `CategoricalEncoder` gains `target_encoding_strategy` (`"mean"|"smoothed"|"catboost"`,
  default `"smoothed"`) and `te_smoothing` (default `10.0`).
  - `"smoothed"` uses `ce.TargetEncoder(smoothing=...)` (Bayesian shrinkage).
  - `"catboost"` uses `ce.CatBoostEncoder` (ordered, leakage-resistant).
  - `"mean"` recovers plain mean target encoding.
  - Output column names are unchanged across strategies, so all downstream name
    tracking and the multiclass `PolynomialWrapper` path are unaffected.
- `FeatureGenerator` gains `target_encoding_strategy` / `te_smoothing` and threads them
  through every encoder via a new `_make_cat_encoder()` helper.
- Presets: `lite`/`medium` use `"smoothed"`; `best`/`extreme` use `"catboost"`.

## P1 — Adversarial validation (train/test shift-aware feature pruning)

**Files:** new `tabularaml/inspect/adversarial.py`, `tabularaml/generate/features.py`

- New `AdversarialValidator`: trains a train-vs-test classifier (LightGBM > XGBoost >
  HistGradientBoosting), exposes `auc_`, `feature_drift_scores()` (drift in [0,1],
  scaled by `2*(AUC-0.5)` so no penalty when distributions match), `oof_test_likeness()`,
  and `cv_sample_weights()`. Uses only test *features* — never a target.
- `FeatureGenerator.search(X, y, X_test=None)` accepts unlabeled test features.
  When `use_adversarial_validation=True` and `X_test` is given:
  - baseline drift is computed once on original columns; engineered features inherit
    drift from their parents (`_candidate_drift`),
  - `_select_elites` discounts a candidate's gain by `adv_drift_weight * drift`,
  - after search, `_adv_final_drift_drop` engineers the test matrix (via a throwaway
    fitted copy) and drops generated features with drift `> adv_drift_max` (capped at
    50% of generated features; originals never dropped).
- New params: `use_adversarial_validation=False`, `adv_drift_weight=0.5`, `adv_drift_max=0.1`.

## P3 — Seeded 2nd-order + groupby-cross templates

**Files:** `tabularaml/generate/features.py`, `tabularaml/configs/feature_gen.py`

- `_seed_template_candidates` deterministically seeds the first generation with the
  high-value region of the search space: numeric×numeric arithmetic crosses among
  top-`seed_top_k` numeric features, every categorical×numeric groupby aggregation,
  and count/freq of top categoricals. Reuses existing `Interaction`, `AGG_OPS`,
  `NUM_OPS`, and the leakage-safe encoders — no new operators.
- New params: `seed_templates=True`, `seed_top_k=15`, `seed_max_candidates=500`.
  Off in `lite`, on for `medium`+.

## P4 — Joint selection + redundancy pruning

**Files:** `tabularaml/generate/features.py`

- `_is_redundant` rejects near-duplicate candidates (abs Pearson corr ≥
  `redundancy_corr_threshold`) against already-accepted generated features, unless the
  candidate's standalone gain is materially higher. Wired into `_select_elites`.
- `_select_elites_batch` evaluates non-pipeline candidates in groups (one model fit per
  batch), admitting complementary "suppressor" features together; pipeline-required
  candidates defer to the sequential selector. Opt-in via `batch_evaluation`.
- New params: `redundancy_prune=True`, `redundancy_corr_threshold=0.95`,
  `batch_evaluation=False`, `batch_size=5`.

## Out of scope (recommended next track)

Neural tabular models (TabM/FT-Transformer/TabPFN v2), multi-level stacking/blending,
pseudo-labeling, polars/cuDF groupby backend, and LLM-assisted proposals (CAAFE/OCTree)
— these are the bigger levers for winning competitions but sit outside the
FeatureGenerator component.
