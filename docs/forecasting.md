# Forecasting family (`tabularaml/generate/forecast.py`)

Sales-forecasting contests (Rossmann, Favorita, M5, Recruit, Walmart, ...) are won with
what is known about each entity's target at forecast time: recent levels, same-weekday
values, last year's shape, and days to and from promotions and holidays. `ForecastFeatures`
builds those automatically. `scripts/contest_features.py` runs it before FeatureForge
(`--forecast auto`, the default; `--forecast off` disables it).

## When it switches on

From the data's structure only:

- a date column (datetime or ISO date strings), daily or coarser;
- unlabeled rows (the test file) lying after the last labelled date (95%+ of them);
- an entity key: the smallest set of code columns (1-3) under which (key, date) is unique,
  with rows repeating over time (a store, store x item).

Otherwise it does nothing (IEEE-CIS, Home Credit, Santander and the suites are untouched:
none has a date column with test rows after the training period).

Detected as well: groups of entities (text columns constant within an entity, and the key's
components: store type, item family), covariates known in advance (low-cardinality columns
that vary within an entity and are not a weekday name: promo, holiday, closure, SNAP days),
and covariate levels that force a zero target (a closed store): those periods are left out of
the target panel rather than read as zero demand. The target is read as log1p when it is
non-negative and skewed.

## No label from the row's own period or later

Test rows are forecast from the last labelled period `T`, at horizons `h = t - T` over the
test file's range. Each training row gets its own origin `t - h`, with `h` drawn from the same
range (a hash of entity and date, so it does not depend on which rows are transformed
together), shifted to the test origin's weekday on daily data. Every target statistic uses
periods up to the origin only, so a training row never reads its own or a later label, exactly
as a test row cannot. `tests/test_forecast.py` checks this by changing every label from a row's
date onwards and asserting its features do not move.

Per-row origins matter. With one origin per block of test-length periods, every row of a block
shares one set of window values and the judge memorises block levels (Rossmann 48-day holdout
0.1162 vs 0.1071 with per-row origins).

## Features

Per entity, as of the origin: horizon; mean over 1 / 3 / 7 / 14 / 28 / 56 / 112 / 364 days;
spread; for intermittent targets the non-zero share, non-zero means and days since the last
sale; mean of the last 1 / 4 / 8 same-weekday values; last year's week around the date, last
year's change from the origin's week to it and a naive seasonal forecast; age; trends.
Per group (and the whole panel): 7 / 28 / 112-day means, same-weekday mean, last year's
change, the entity's level against its group. Per covariate: days since its last and until its
next event (until is cut at the origin plus the longest horizon, the end of what a test file
shows), and the entity's recent target difference between event and non-event days. Calendar
fields of the date. Covariates are label-free and read from every row, the test file included.

## Unseen contests (run blind with the defaults above, then after one generic change)

Walmart Store Sales (weekly; Hugging Face mirror `large-traversaal/Walmart-sales`;
`scripts/walmart_bench.py`, last 39 weeks / 39 before, rows of the test file's columns with the
store and features tables joined; WMAE, holidays weigh 5) and Recruit Restaurant Visitor
Forecasting (daily; the contest files from a GitHub mirror; `scripts/recruit_bench.py`, last 39 days
/ 39 before, the full store x day grid as the test file lists it; RMSLE). Kaggle refuses these
downloads here (rules not accepted).

| Contest | Raw | Blind: family | Blind: full pipeline | After the change | Build |
|---|---|---|---|---|---|
| Walmart (WMAE) | 3038 / 3935 | **1676 / 2639** | 1651 (window 0) | 1676 / 2639 (weekly: unaffected) | 6-9 s (pipeline 27 min) |
| Recruit (RMSLE) | 0.5305 / 0.5233 | 0.5188 / 0.5209 | 0.5249 (window 0) | **0.5153 / 0.5191** | 5-7 s |

Shuffled labels: Walmart 13420 (constant 13421), Recruit 0.8326 (constant 0.8326).
Recruit's full pipeline (reservations as of each date + FeatureForge) is worse than the family
alone; the earlier pipeline without the family scored 0.546 / 0.537.

The change (`long_season`, now on): half a year of same-weekday values (mean, median, spread, and
the weekday's level against the entity's), a stable weekday profile for short noisy series. On the
earlier contests: Rossmann 0.1079 / 0.1019 -> 0.1060 / 0.1006, Favorita 0.6776 / 0.6693 ->
0.6789 / 0.6702, M5 0.7506 / 0.7124 -> 0.7510 / 0.7187 (M5's judge stops after ~50 trees and moves
by this much between runs), Walmart unchanged.

## Results (held-out windows built like each contest's test file; same judges as the benches)

| Contest (metric) | Raw | Earlier best | Forecast family | Build time |
|---|---|---|---|---|
| Rossmann, last 48 days / 48 before (RMSPE) | 0.1266 / 0.1583 | | **0.1079 / 0.1019** | 17 s |
| Favorita, stores 44/45/47, last 16 days / 16 before (NWRMSLE) | 0.9087 / 0.8697 | FeatureForge 0.785 / 0.739 | **0.6776 / 0.6693** | 23 s |
| M5, store CA_1, last 28 days / 28 before (RMSSE) | 0.7821 / 0.7537 | FeatureForge 0.776 / 0.720; hand-made 0.7711 / 0.7377 | **0.7506 / 0.7124** | 6 s |

Shuffled training labels land at chance: Rossmann 0.4926 (raw with shuffled labels 0.4924),
Favorita 1.2211 (a constant scores 1.2214).

Rossmann's holdouts list only test-file columns and rows (the last 48 days, no `Sales`, no
`Customers`); Favorita's are the full store x item grid without `onpromotion`, as its test
file would be; M5's are the next 28 days of every item on sale. Reproduce with
`scripts/{rossmann,favorita,m5}_bench.py --arm fc [--win 1] [--shuffle]`.
