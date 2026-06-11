# Acceptance A/B benchmark — old vs new genetic search

Paired runs: 18 (dataset x seed)

## Pre-registered criteria

- (a) test-gain superiority: win/tie/loss = 8/1/9, win-rate=47%, Wilcoxon one-sided p=0.4309 -> FAIL
- (b) overfit gap reduced: old=0.0028, new=0.0022 -> PASS
- (c) no dataset regression beyond old's seed noise: PASS
- (d) throughput >= 1.5x: mean evals/min ratio = 2.24x -> PASS

- mean test gain: old=0.0077, new=0.0091

## Verdict

TEST-GAIN CRITERION FAILED: per the decision rule, the new acceptance stack should ship opt-in (defaults stay legacy).

## Per-dataset means (test gain)

              mean             std        
config         new     old     new     old
dataset                                   
503_wind    0.0028  0.0028  0.0004  0.0040
537_houses  0.0341  0.0256  0.0250  0.0086
churn       0.0055  0.0201  0.0050  0.0582
satimage    0.0490  0.0000  0.0065  0.0000
spambase   -0.0192  0.0050  0.0265  0.0259
splice     -0.0174 -0.0070  0.0442  0.0465