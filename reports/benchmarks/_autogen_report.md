# Before vs After — paired targeted-benchmark comparison

- Paired (status=ok both arms): **74** runs over **25** datasets; unpaired/failed rows: 1 (see unpaired_runs.csv)
- NoFE identity check: 0 rows drifted (> 1e-9) out of 75 ✅

## Run level (dataset × seed)

| | before | after |
|---|---|---|
| median pct_improvement vs NoFE | 0.00689 | 0.00000 |
| mean pct_improvement vs NoFE | 0.06514 | 0.07914 |

- Δ(after−before): median 0.00213, mean 0.01399
- Wins / losses / ties: **42 / 32 / 0**
- Wilcoxon signed-rank (after > before): p = **0.08687** (two-sided p = 0.1737, n = 74)

## Dataset level (mean over seeds)

- Δ median 0.00084, mean 0.01184; wins/losses/ties: **14 / 11 / 0** (n = 25)
- Wilcoxon (after > before): p = **0.1979**

## Reliability and cost

- Status counts before: `{'ok': 74, 'timeout': 1}`
- Status counts after: `{'ok': 75}`
- Mean wall time (paired runs): 460s → 323s
- Mean features added: 7.1 → 3.1

## Per-dataset deltas (mean over seeds, sorted)

| dataset | task seeds | pct before | pct after | Δ |
|---|---|---|---|---|
| shuttle | 2 | 0.26425 | 0.11662 | -0.14762 |
| ionosphere | 3 | 0.06161 | 0.01922 | -0.04239 |
| vehicle | 3 | 0.01491 | -0.02154 | -0.03645 |
| sonar | 3 | -0.04517 | -0.07590 | -0.03072 |
| pollen | 3 | 0.04798 | 0.02674 | -0.02124 |
| satimage | 3 | -0.00537 | -0.01003 | -0.00466 |
| adult | 3 | 0.00788 | 0.00350 | -0.00438 |
| breast-cancer | 3 | -0.00941 | -0.01373 | -0.00432 |
| monk1 | 3 | 0.67612 | 0.67220 | -0.00393 |
| SWD | 3 | 0.00304 | 0.00000 | -0.00304 |
| BNG-lowbwt | 3 | -0.00021 | -0.00058 | -0.00037 |
| titanic | 3 | -0.00405 | -0.00330 | +0.00075 |
| wind | 3 | 0.00353 | 0.00437 | +0.00084 |
| waveform-21 | 3 | -0.00212 | 0.00049 | +0.00261 |
| spambase | 3 | -0.00091 | 0.00266 | +0.00357 |
| mfeat-factors | 3 | -0.00285 | 0.00241 | +0.00526 |
| kr-vs-kp | 3 | -0.00250 | 0.00762 | +0.01012 |
| optdigits | 3 | -0.01292 | 0.00639 | +0.01932 |
| letter | 3 | -0.01799 | 0.00840 | +0.02638 |
| mushroom | 3 | 0.18950 | 0.24358 | +0.05408 |
| chess | 3 | 0.03669 | 0.09973 | +0.06304 |
| car-evaluation | 3 | 0.31775 | 0.38792 | +0.07018 |
| Friedman-c3 | 3 | 0.06409 | 0.15326 | +0.08917 |
| pendigits | 3 | -0.09126 | 0.01368 | +0.10494 |
| nursery | 3 | 0.20239 | 0.34719 | +0.14479 |
