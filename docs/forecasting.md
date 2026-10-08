# Forecasting family (`tabularaml/generate/forecast.py`)

Sales-forecasting contests (Rossmann, Favorita, M5, Recruit, Walmart, ...) are won with
what is known about each entity's target at forecast time: recent levels, same-weekday
values, last year's shape, and days to and from promotions and holidays. `ForecastFeatures`
builds those automatically. `scripts/contest_features.py` runs it before FeatureForge
(`--forecast auto`, the default; `--forecast off` disables it).

## When it switches on

From the data's structure only:

- a date column (datetime or ISO date strings): hourly, daily, weekly or monthly (one date per calendar month,
  counted in months);
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

### Web Traffic Time Series Forecasting (Kaggle 2017, $25k)

Monash archive copy (Zenodo 4656080; page names replaced by ids, so no project / access / agent groups);
`scripts/webtraffic_bench.py`: 5,000 random pages, the last 500 days before a cut, then a 2-day gap and
62 days of every page (stage 2's layout; days without a value are not scored); SMAPE.

| Windows | Raw | Blind: family | + trailing medians (`medians=True`, opt-in) | Build |
|---|---|---|---|---|
| last / 64 days earlier | 40.40 / 40.53 | **38.16 / 37.10** | 38.03 / 37.08 | 45-60 s |

Shuffled labels 121.10 (constant 121.11). Medians moved SMAPE by 0.1-0.3%, within noise, so they stay
off and the shipped defaults (and every earlier number) are unchanged. The test horizon starting at 3
(the gap) is handled by the same per-row origins.

### ASHRAE Great Energy Predictor III (Kaggle 2019, $25k): hourly data

The contest's source, Building Data Genome 2 (github.com/buds-lab/building-data-genome-project-2), electricity
meters; `scripts/ashrae_bench.py`: 300 random buildings with readings in both years (two samples), trained on
2016, scored on every 2017 hour of the same buildings (the contest scored the following years), weather
joined as the test file had it; RMSLE of the reading.

| Building sample | Raw | Blind (shipped defaults) | Hourly support | Build |
|---|---|---|---|---|
| seed 0 / seed 1 | 0.7043 / 0.5614 | 0.7043 / - (family off: "sub-daily timestamps") | **0.6682 / 0.5211** | 140 s |

Shuffled labels 1.7195 (constant 1.7192). The change: hourly timestamps switch the family on, with the
same hour of the week as the season (26 weeks of it), the same hour on the latest 1 / 7 days, hour and
weekday fields, windows from 1 hour to 52 weeks, and training origins aligned to the test origin's hour of
the week. Daily and weekly data take the same code path as before (M5 reproduces 0.75098 exactly).

### GoDaddy Microbusiness Density Forecasting (Kaggle 2023, $60k): monthly data

The contest's train and revealed-test months (2019-08 .. 2022-12, 3,135 counties, census columns joined by year)
from the public Kaggle dataset `jjithin/business-density`; `scripts/godaddy_bench.py`: every county for the next
6 months after a cut (window 0 ends on 2022-12, window 1 six months earlier); SMAPE of the density.

| Windows | Raw | Blind (shipped defaults) | Calendar months | Last known value | Build |
|---|---|---|---|---|---|
| last 6 months / 6 before | 5.65 / 12.71 | 3.69 / 3.94 | **3.53 / 3.84** | 3.25 / 2.99 | 2 s |

Shuffled labels 55.02 (constant 55.04). Blind, the family switched on but counted periods as 28 days (the
shortest gap between month starts), so months drifted. The change counts one period per calendar month when
every date falls in its own month; other cadences take the old path (M5 0.75098 and Walmart 1676 reproduce).

Neither the raw columns nor the features beat carrying the last value forward: the density is close to a random
walk, and a tree model predicting the level cannot place 3,135 counties' levels that precisely. Starting the
judge from the last known value (`--base`, LightGBM's starting score, a modelling choice outside this family)
scores 3.29 / 3.00, with early stopping after one tree: the features carry nothing beyond persistence here.

### GEFCom2012 load track (Kaggle 2012, $7.5k): hourly load, no temperatures for the test week

The organiser's files (GEFCom2012.zip); `scripts/gefcom12_bench.py`: 20 zones, the week after a cut, scored with the
contest's WRMSE (zones weight 1, their sum weight 20). Window 0 is the contest's own forecast week (2008-07-01 .. 07,
scored with the published solution, horizons 19-186 hours as in the contest); windows 1-8 are the 8 full weeks before
it. Temperatures are left out because the contest week had none.

| Weeks | Raw (mean WRMSE) | Blind (shipped defaults) | Build |
|---|---|---|---|
| contest week | 150,477 | **104,836** | 12 s |
| all 9 weeks | 165,012 | **153,440** (6% lower geometric mean; better on 5 of 9) | 12 s |

Shuffled labels 738,105 (per-zone constant 735,627). No change followed: week-ahead load without temperatures swings
with the weather, and single weeks move by up to 35% either way, so a change could not be judged within noise.

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
