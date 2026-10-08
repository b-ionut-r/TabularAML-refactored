"""Score contest_features.py outputs on a labelled holdout: the holdout went in as the unlabeled
test file, its labels only reach this scorer. Same LightGBM judge for every arm (bagged over
``--bags`` seeds). Prints and appends one JSON line.

    python scripts/score_holdout.py --dir feats/ --labels holdout_labels.parquet --target TARGET --id SK_ID_CURR
"""
import argparse, json, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, mean_absolute_error

ap = argparse.ArgumentParser()
ap.add_argument('--dir', required=True); ap.add_argument('--labels', required=True)
ap.add_argument('--target', required=True); ap.add_argument('--id', required=True)
ap.add_argument('--metric', default='auc', choices=['auc', 'mae_log200']); ap.add_argument('--bags', type=int, default=3)
ap.add_argument('--tag', default=''); ap.add_argument('--log', default='holdout.jsonl')
ap.add_argument('--drop', default='', help='comma-separated column prefixes to drop (ablation)')
a = ap.parse_args()
tr = pd.read_parquet(a.dir + '/train_features.parquet'); te = pd.read_parquet(a.dir + '/test_features.parquet')
lab = pd.read_parquet(a.labels).set_index(a.id)[a.target]
y = tr.pop(a.target).to_numpy(dtype=float); yte = lab.reindex(te[a.id].to_numpy()).to_numpy(dtype=float)
tr, te = tr.drop(columns=[a.id]), te.drop(columns=[a.id])
if a.drop:
    keep = [c for c in tr.columns if not any(c.startswith(p) for p in a.drop.split(','))]
    tr, te = tr[keep], te[keep]
for c in tr.columns:
    if not (pd.api.types.is_numeric_dtype(tr[c]) or isinstance(tr[c].dtype, pd.CategoricalDtype)):
        tr[c] = tr[c].astype(str).astype('category')
    if isinstance(tr[c].dtype, pd.CategoricalDtype):
        te[c] = pd.Categorical(te[c].astype(str), categories=tr[c].cat.categories)
z = np.log(y + 200) if a.metric == 'mae_log200' else y
P = dict(objective='binary' if a.metric == 'auc' else 'regression_l1', learning_rate=0.03, num_leaves=31,
         min_child_samples=100, feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
         num_threads=4, verbose=-1, max_bin=255)
t0 = time.time(); preds = []
for s in range(a.bags):
    P['seed'] = s
    perm = np.random.default_rng(s).permutation(len(tr)); i, v = np.sort(perm[:int(.85 * len(perm))]), np.sort(perm[int(.85 * len(perm)):])
    b = lgb.train(P, lgb.Dataset(tr.iloc[i], z[i]), 10000, valid_sets=[lgb.Dataset(tr.iloc[v], z[v])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    preds.append(lgb.train(P, lgb.Dataset(tr, z), int(b.best_iteration * 1.1) + 1).predict(te))
p = np.mean(preds, 0)
score = roc_auc_score(yte, p) if a.metric == 'auc' else mean_absolute_error(yte, np.exp(p) - 200)
res = dict(dir=a.dir, tag=a.tag, ncol=tr.shape[1], **{a.metric: float(score)}, judge_s=round(time.time() - t0))
print('RESULT', json.dumps(res)); open(a.log, 'a').write(json.dumps(res) + '\n')
