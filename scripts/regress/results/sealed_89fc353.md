## Sealed contests, final head 89fc353 (one run each, seed s0, pandas 3.0.5, normal memory limit, fixed bagged LightGBM judge)

| Contest | Metric (higher better) | Raw | FF | FF build s | FF wall s | FF peak GB | Gate skipped / memory guard line |
|---|---|---|---|---|---|---|---|
| CMI PIU | QWK | 0.43478 | 0.44700 (+0.0122) | 82 | 86 | 0.37 | no |
| ISIC 2024 (metadata) | pAUC above 80% TPR | 0.04304 | 0.03357 (-0.0095) | 678 | 714 | 2.27 | no |
| Student Game Play | macro F1 (best threshold) | 0.65178 | 0.68278 (+0.0310) | 343 | 382 | 3.18 | no |
| OTTO | AUC | 0.77739 | 0.78805 (+0.0107) | 1,143 | 1,171 | 2.16 | no |
| Jane Street | weighted R2 | 0.00721 | 0.00651 (-0.0007) | 572 | 647 | 5.56 | no |

All five exit 0, no crash; the memory guard never fired. Gains 3 of 5, losses 2 of 5, one seed only.

Where each change came from (FF job logs):
- CMI PIU, Student Game Play, OTTO: the gain is the related-table columns (13 / 168 / 25); the search gate rejected its own rounds (+0.00%).
- ISIC 2024: 0 columns added; the change is the high-cardinality recode of patient_id (frequency ranks fitted on all known rows because every holdout patient is new). The grouped gate passed it on log loss (+10.9%, z=+5.0), yet holdout pAUC80 fell 22%.
- Jane Street: the panel family (entity symbol_id, moment date_id/time_id) added 72 columns, not gated; the search found nothing (round 1 stopped). Weighted R2 fell 0.0007 on a metric this small (noise unmeasured).
