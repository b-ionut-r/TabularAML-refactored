# A/B benchmark — gitold vs full genetic search

Paired runs: 36 (dataset x seed)

## Pre-registered criteria

- (a) test-gain superiority: win/tie/loss = 14/6/16, win-rate=47%, Wilcoxon one-sided p=0.3916 -> FAIL
- (b) overfit gap reduced: gitold=0.0391, full=0.0101 -> PASS
- (c) no dataset regression beyond the baseline's seed noise: FAIL [('4544_GeographicalOriginalofMusic', 0.00041414355647919975, -0.008032500588888901, 0.00744771298370047), ('optdigits', 0.0, -0.0086722004197332, 0.0), ('spambase', 0.0018069921065124665, -0.0269708290715383, 0.011615857955192137), ('splice', 0.0, -0.013035756881139999, 0.0)]
- (d) throughput >= 1.5x: mean evals/min ratio = infx -> PASS

- mean test gain: gitold=0.0000, full=0.0059
- mean generations completed: full=5.2, gitold=5.7

## Verdict

TEST-GAIN CRITERION FAILED: per the decision rule, the strict acceptance stack stays opt-in (defaults stay as shipped).

## Per-dataset means (test gain)

                                    mean             std        
config                              full  gitold    full  gitold
dataset                                                         
4544_GeographicalOriginalofMusic -0.0080  0.0004  0.0032  0.0074
503_wind                          0.0043  0.0047  0.0027  0.0072
537_houses                        0.0297  0.0400  0.0257  0.0117
573_cpu_act                       0.0228  0.0031  0.0402  0.0048
ann_thyroid                       0.0000 -0.0101  0.0000  0.0176
churn                             0.0180 -0.0017  0.0256  0.0414
coil2000                          0.0087  0.0030  0.0225  0.0093
hypothyroid                      -0.0053 -0.0515  0.0095  0.1111
optdigits                        -0.0087  0.0000  0.0150  0.0000
satimage                          0.0490  0.0105  0.0065  0.0091
spambase                         -0.0270  0.0018  0.0152  0.0116
splice                           -0.0130  0.0000  0.0424  0.0000

## Era showcase (synthetic, held-out FUTURE eras, ABSOLUTE mean per-era Spearman delta vs raw features)

       base_test         new_test        test_gain        
config      full  gitold     full gitold      full  gitold
seed                                                      
42        0.6879  0.6879   0.6983  0.699    0.0105  0.0111
43        0.6879  0.6879   0.6906  0.695    0.0027  0.0071
44        0.6879  0.6879   0.6931  0.692    0.0052  0.0041