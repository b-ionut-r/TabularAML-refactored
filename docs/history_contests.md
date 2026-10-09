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
