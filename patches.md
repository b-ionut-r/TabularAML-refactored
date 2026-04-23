# OpenFE / LightGBM Patch Findings

Date: 2026-04-23

## Scope

This note records the source-level conclusion about the OpenFE multiclass patches in
`tabularaml/benchmarks/feature_gen/adapters/openfe_adapter.py`.

The question checked here was:

- Did the multiclass `init_score` flattening patch break OpenFE?
- Did the multiclass probability-to-log patch ("logits" patch) break OpenFE?

## Repos Inspected

Upstream repos cloned locally for direct inspection:

- `tmp/upstream/OpenFE`
- `tmp/upstream/LightGBM-4.6.0`

## Conclusion

Source + short-runtime conclusion: the OpenFE multiclass patches do **not** make OpenFE semantically wrong.

They repair an upstream OpenFE inconsistency so that OpenFE matches LightGBM's actual multiclass
`init_score` contract.

More specifically:

- The Fortran / column-major flattening patch is correct.
- The multiclass probability-to-log patch is correct in substance.
- "Logits" is slightly imprecise naming here; for multiclass this is better described as a
  log-probability / raw-margin patch.

This means the poor mean gain in the benchmark is **not** explained by these patches making
OpenFE's multiclass `init_score` handling invalid.

## Evidence From Upstream OpenFE

OpenFE default multiclass path builds `init_scores` from class-frequency probabilities:

- `tmp/upstream/OpenFE/openfe/openfe.py:442-447`

But the same file also says classification `init_scores` should be raw scores:

- `tmp/upstream/OpenFE/openfe/openfe.py:457-460`

OpenFE's `feature_boosting=True` path uses raw scores from LightGBM:

- `tmp/upstream/OpenFE/openfe/openfe.py:433-436`

OpenFE passes `init_score=train_init` and `eval_init_score=[val_init]` directly into LightGBM:

- `tmp/upstream/OpenFE/openfe/openfe.py:548-549`
- `tmp/upstream/OpenFE/openfe/openfe.py:612-613`

OpenFE computes multiclass initial metric by applying `softmax(pred)`:

- `tmp/upstream/OpenFE/openfe/openfe.py:565-569`

That only makes sense if `pred` is a raw margin, not already a probability matrix.

Therefore upstream OpenFE is internally inconsistent on default multiclass initialization:

- default path produces probabilities
- downstream checks and metric code assume raw margins

## Evidence From LightGBM 4.6.0

LightGBM Python flattens 2-D multiclass `init_score` in Fortran order:

- `tmp/upstream/LightGBM-4.6.0/python-package/lightgbm/basic.py:2819-2825`

LightGBM reshapes multiclass `init_score` back using `order="F"`:

- `tmp/upstream/LightGBM-4.6.0/python-package/lightgbm/basic.py:2913-2917`

When LightGBM creates init score from a predictor, it uses `raw_score=True` and then regroups
into class-major layout:

- `tmp/upstream/LightGBM-4.6.0/python-package/lightgbm/basic.py:2071-2094`

LightGBM multiclass objective reads scores as raw margins and applies `Softmax` internally:

- `tmp/upstream/LightGBM-4.6.0/src/objective/multiclass_objective.hpp:86-132`

LightGBM's own multiclass base score is `log(class_init_prob)`:

- `tmp/upstream/LightGBM-4.6.0/src/objective/multiclass_objective.hpp:155-157`

This is the key point: LightGBM multiclass expects raw margins, and its own built-in
boost-from-score value is the log of class probability, not the probability itself.

## Implication For Our Adapter

Our adapter patch does the following:

- flatten 2-D multiclass `init_score` to 1-D Fortran order
- if the matrix looks like probabilities, convert `p -> log(p)`

Relevant local code:

- `tabularaml/benchmarks/feature_gen/adapters/openfe_adapter.py:95-166`

Those behaviors match the upstream LightGBM contract and fix the upstream OpenFE inconsistency.

## Short Runtime Checks

Short local checks were also run in `C:\ml_env` against OpenFE's own internals on tiny real
multiclass datasets (`iris` and `wine`).

What was checked:

- what `OpenFE.get_init_score(None)` actually returns for default multiclass initialization
- whether `softmax(p)` distorts that returned matrix
- whether `OpenFE.get_init_metric()` matches true multiclass log loss under `p` versus `log(p)`
- whether OpenFE's own stage-1-style one-feature gains change under `p` versus `log(p)`

Observed:

- `OpenFE.get_init_score(None)` returned true probability matrices on both datasets:
  values were in `[0, 1]` and row sums were exactly `1.0`
- therefore the adapter's probability detector is firing on a real OpenFE output, not a
  hypothetical edge case
- `softmax(p) != p`, so passing raw probabilities as multiclass `init_score` does distort the
  prior before LightGBM training

Concrete numbers:

- `iris`
  - mean absolute difference between `softmax(p)` and `p`: about `0.00264`
  - `OpenFE.get_init_metric(p, y)`: `1.0985680`
  - `OpenFE.get_init_metric(log(p), y)`: `1.0985328`
  - true log loss of `p`: `1.0985328`
- `wine`
  - mean absolute difference between `softmax(p)` and `p`: about `0.02880`
  - `OpenFE.get_init_metric(p, y)`: `1.0917982`
  - `OpenFE.get_init_metric(log(p), y)`: `1.0863153`
  - true log loss of `p`: `1.0863153`

This means:

- `log(p)` reproduces the correct multiclass prior exactly
- raw `p` does not

Fast stage-1 gain check on `wine`:

- using raw `p` instead of `log(p)` inflated OpenFE's internal one-feature gains for all
  `13 / 13` base features tested
- mean absolute score delta: about `0.00547`
- max absolute score delta: about `0.01059`
- the top-2 ranked features swapped between the two variants

So the no-log path is not just theoretically imperfect; it measurably changes OpenFE's internal
feature scoring.

## Practical Interpretation

The OpenFE patch is not the credible explanation for the overall bad mean benchmark result.

The remaining likely causes are upstream OpenFE instability elsewhere, including:

- timeout-heavy behavior
- rare-class / split instability
- other dataset-specific catastrophic failures

## Important Precision

This note is a source-contract conclusion, not a universal runtime proof of zero regressions on
every possible edge case.

What is established from source is:

- the patch aligns OpenFE with LightGBM's intended multiclass handling
- the patch did not invent an invalid multiclass `init_score` interpretation
