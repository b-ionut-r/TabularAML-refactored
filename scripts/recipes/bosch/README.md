# Bosch Production Line Performance (2016), run blind

Data: public copy `uniface/update-file` (train_numeric / train_date / train_categorical, byte sizes and column counts
as the contest's; the contest download needs its rules accepted). `prep.py` takes the first 300k parts by Id (the
files are in Id order) and holds out the latest 60k; all 4,264 fields kept (numeric and date as float32,
categorical as int8 codes). `hand.py` rebuilds the winners' features; `judge.py <arm>` is the shared bagged
LightGBM (AUC; MCC at the best threshold on the held-out rows, same rule for every arm). The judge needs
`bin_construct_sample_cnt` capped: LightGBM's 200k-row binning sample of 4,264 columns alone is 11 GB.

| arm | AUC | MCC | build | peak |
|---|---|---|---|---|
| raw | 0.704 | 0.271 | | |
| winners' features | 0.796 | 0.345 | 1 min | |
| winners' features without the Id-order ones | 0.717 | 0.268 | | |
| FeatureForge 6c034b5 (25 kept, gate +0.67%) | 0.711 | 0.271 | 33 min | 13.1 GB |
| shuffled labels, 6c034b5 (0 kept) | 0.528 | 0.005 | 13 min | 11.8 GB |

The winners' lift is the Id-order "magic": Id gaps to the neighbouring parts by start time. FeatureForge never sees
the Id (it is the id column), so it matches the rest of the hand set but not that.
