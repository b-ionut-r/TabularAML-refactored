# Hull Tactical Market Prediction (2025) recipe

Data: the competition's own files (download allowed on this account):
`curl -L -o all.zip https://www.kaggle.com/api/v1/competitions/data/download-all/hull-tactical-market-prediction`,
unzip train.csv (8,465,196 B; 9,048 trading days x 98) to /home/user/data/new3/hull/.

Prep (`prep_hull` in scripts/recipes/new3_bench.py; set `--data` to the folder holding hull/): rows sorted by date_id, then date_id dropped, so the file is a daily series
without a timestamp (the series-rule test case, like DRW). Label: market_forward_excess_returns; forward_returns and
risk_free_rate (the label's own parts) dropped. 94 features (D, E, I, M, P, S, V). Windows: `hull` trains on the first
80% of days and holds out the last 20% (1,810 days); `hull_w1` trains on the first 60% and holds out the next 20%.
Score: Pearson correlation (one held-out day = one row; SE about 0.024 at 1,810 rows). Judge as for DRW.

    for c in hull hull_w1; do
      python3 new3_bench.py --contest $c --arm raw --seed 0
      python3 new3_bench.py --contest $c --arm ff --seed 0 --repo <worktree> [--shuffle]
    done
