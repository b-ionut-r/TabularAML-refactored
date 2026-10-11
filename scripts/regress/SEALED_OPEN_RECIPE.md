# ICR, CIBMTR, JPX blind recipe (generalization bench)

Script: sealed_bench.py (here, and on branch claude/generalization-bench-hnjxf2 at scripts/regress/sealed_bench.py).
    python3 sealed_bench.py --contest icr|cibmtr|jpx --arm raw|ff --seed N [--shuffle] --repo <worktree of the commit>
Reads data from /home/user/data/sealed/<contest>/ (edit D at the top if elsewhere). FF arm = <repo>/scripts/contest_features.py
at shipped defaults, --budget 900, held-out rows' features as --test; held-out labels reach only the scorer. Last stdout line
"RESULT {json}" has the score. Versions: pandas 2.3.3, numpy 2.4.6, lightgbm 4.7.0, scikit-learn 1.9.1, pyarrow 24.0.0, scipy 1.17.1.
Judge (all arms): LightGBM lr 0.05, 63 leaves, early stop on 15% of train rows (time-ordered for JPX), refit 3 seeds at 1.1x best iter.

Data (Kaggle via proxy; unzip with `unzip`, Python 3.13 zipfile fails on Kaggle zips):
- icr: dataset bigswipp/icr-identify-age-related-conditions (byte-identical to the competition). Uses train.csv only
  (greeks.csv not used: Alpha leaks the label). Holdout: stratified 20% by Class, seeds 0-4 (617 rows, noisy: band 0.03).
  Metric: balanced log loss, lower better.
- cibmtr: competition equity-post-HCT-survival-predictions (rules accepted on this account), train.csv.
  Holdout: 20% stratified by race_group, seeds 0-1. Target: KM survival at efs_time fitted on train rows, minus 0.15 for
  censored rows; scored as risk = -prediction with the stratified C-index by race_group (higher better).
- jpx: dataset sonzai/jpx-tokyo-stock-exchange (final version; not the yutoyasuda1999 copy). Train = train_files/stock_prices
  (2017-01-04..2021-12-03), holdout = supplemental_files/stock_prices (2021-12-06..2022-06-24), stock_list.csv merged
  (sector codes, market cap, issued shares). One seed. Metric: Sharpe of the daily top-200 minus bottom-200 rank spread.

Probe lessons (38fef7e): ICR and CIBMTR have no high-cardinality columns (rank-fit switch never reached). JPX switches the
forecasting family on and skips the search. Raw ICR seed 0 = 0.4948. Run arms one at a time on an idle box: FF's search is
time-budgeted, so a second job changes its picks.
