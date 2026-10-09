# Per-customer history contests, run blind (Amex, Elo, Riiid)

Every contest went through `scripts/contest_features.py` at its defaults (PR #2 at f9ce09d), with the contest's own
layout: the main table holds one row per customer, card or question event, and the history is a child table. Holdouts
keep whole customers out, as the test files do. The judge is the same bagged LightGBM throughout: early stopping on 15%
of the training keys, then 3 seeds. Public Kaggle copies of the files were used (named in each bench's docstring).

| Contest (metric) | Sample | Raw | Blind | Blind + `--history auto` | Build | Shuffled labels |
|---|---|---|---|---|---|---|
| Amex Default Prediction (Amex metric, higher is better) | 100k customers, 20% unseen, 2 samples | 0.7851 / 0.7786 (last statement) | 0.7840 / 0.7733 | **0.7938 / 0.7833** | 5-7 min | 0.012 / 0.030 (chance) |
| Elo Merchant Category (RMSE, lower is better) | 60k cards, 20% unseen, 2 samples | 3.866 / 3.791 | **3.741 / 3.712** | 3.742 / 3.716 | 3.5-4.5 min | 3.867 (constant 3.867) |
| Riiid Answer Correctness (AUC) | 5k students: 10% new, last 10% of the rest | 0.612 / 0.618 | **invalid: leak** (0.922 / 0.922) | | 26 min | 0.52 |

Amex: the winners' aggregates (last, mean, std, min, max, last - mean, last - previous, count) score 0.7906 / 0.7848
on their own. Blind lacked the latest-state ones: `RelatedTables` summarises all rows but has no latest value.
`tabularaml/generate/history.py` adds, for child tables with a time column and repeated keys, each numeric column's
last value, last - mean and last - previous (`--history auto`, off by default). That brings blind to the winners'
level. The Amex test file's customers also come from a later period, which these holdouts cannot copy.

Elo: blind cuts RMSE by 2-3%; the latest-state columns add nothing there. Most of Elo's error is the -33 outliers
(about 1% of cards), which winners handled with a separate outlier model (modelling).

## Riiid: a leak in cross-row statistics over an event log

With earlier answers revealed as the contest's API did (`--reveal`: each row sees answers from strictly earlier
timestamps only), blind scores AUC 0.922. That is far above the winners (about 0.82), so it is a leak. Split by
feature on sample 1 (one LightGBM, 400 rounds):

| Columns | Held-out AUC |
|---|---|
| raw + as-of aggregates of the log | 0.694 |
| + FeatureForge's target encoding of the question | 0.760 |
| + FeatureForge's other kept columns (`anchor__...`, `grp_dev__... by user_id`, `count__user_id__...`) | 0.914 |

FeatureForge treats the as-of aggregates as label-free columns and builds group statistics over all of a student's
rows, the later ones included. A later row's as-of `answered_correctly` mean contains this row's answer, so the group
statistic hands each row its own label. The as-of step itself is clean (0.694 to 0.760 is a plausible single-model
level). With held-out answers hidden instead, training rows still leak this way while held-out rows cannot, and blind
falls to 0.622 / 0.563 against raw 0.612 / 0.618. Any main table with repeated keys and an as-of child holding the
target (DSB 2019's event logs too) is exposed. Cross-row statistics on such tables would need to read earlier rows
only, or label-derived as-of columns kept out of them; that is FeatureForge's code, left to the feature thread.

## After the fix (PR #2 bb533a9, `label_echo`) and with outcome history

Riiid, earlier answers revealed (`--reveal`), 5k students per sample, AUC:

| Run | Sample 0 | Sample 1 | Build |
|---|---|---|---|
| Raw | 0.612 | 0.618 | |
| Blind, before the fix (leak) | 0.922 | 0.922 | 26 min |
| **Blind, with the fix** | **0.753** | **0.763** | 7-27 min |
| Blind + `--history auto` (outcome history) | **0.766** | **0.768** | +3 s for the history, 27-31 min in all |
| Shuffled labels and log answers (with the fix; + history) | 0.522; 0.514 | | |
| Raw columns, shuffled labels (noise floor of a near-empty model) | 0.511 | | |

Outcome history (`event_log_features` in `tabularaml/generate/history.py`, on with `--history auto` when an as-of
log carries the target column): the student's earlier outcomes (count, mean, the last 5 / 20 / 100, the last five
one by one), time since the latest, 3rd and 10th latest event, and the student's earlier attempts on the same
question (count, mean, time since). Only events at strictly earlier timestamps count, so a bundle's answers stay
hidden from each other (`tests/test_history.py`). The label-echo detector flags the overall columns but not the
lags 2-5 or the same-question mean; FeatureForge could still group on those.

The winners' 0.82 also used the question metadata (`questions.csv`: part, tags; not in this copy), every student's
full history (these samples hold 5k of 394k students, so the question difficulty is estimated from 1M answers
instead of 100M), and sequence models (modelling).

## DSB 2019 audit (same split as the toolkit's table)

| Run | Seed 0 | Seed 1 |
|---|---|---|
| As-of aggregations | 0.535 | 0.533 |
| + FeatureForge (as before the fix) | 0.553 | 0.538 |
| + FeatureForge, each draw's assessments transformed alone (no later held-out rows visible) | 0.553 | 0.538 |
| + FeatureForge without group statistics or entities | 0.539 | 0.533 |
| Shuffled labels | -0.001 | |
| Assessment attempt codes shuffled in the history | 0.510 | |

The held-out number does not depend on later rows, so 0.546 holds as a held-out score. Group statistics carry most
of FeatureForge's gain here.

## Default check (history families on from structure, `--history auto`)

Blind `scripts/contest_features.py` at the current head (label_echo guard included), same bagged LightGBM:

| Contest | Metric | Blind, history off | History on | Shuffled labels | Build time |
|---|---|---|---|---|---|
| Amex Default, sample 0 / 1 | Amex | 0.784 / 0.773 | 0.794 / 0.783 | 0.03 | 6-8 min |
| Elo, seed 0 / 1 | RMSE | 3.742 / 3.712 | 3.742 / 3.716 | 3.867 (= constant) | ~same |
| Riiid (revealed log) | AUC | 0.753 / 0.763 | 0.766 / 0.768 | 0.514 (floor 0.511) | ~same |
| Home Credit, 100k loans, 5 child tables | AUC | 0.7895 | 0.7889 (noise) | | 14 vs 18 min |
| Rossmann, 150 stores | RMSPE | 0.1034 | 0.1034 (off: no child table) | | 13 min |

Off by structure: DSB 2019 (its event log has no outcome column; latest-state needs a keyed child table),
Santander and IEEE-CIS (no keyed child tables; IEEE's identity table is one row per transaction).
