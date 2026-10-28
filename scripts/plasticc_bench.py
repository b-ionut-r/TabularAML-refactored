"""PLAsTiCC Astronomical Classification (Kaggle 2018, $25k): classify objects from light curves.

Data: the unblinded release (Zenodo 2539456): training metadata and light curves (7,848 objects),
test metadata with true classes, and the first test light-curve file (the 32,926 deep-drilling-field
test objects). The holdout is the contest's own test set (that part of it): trained on the contest's
biased training sample, scored on test objects of the 14 training classes (the unseen class 99 is left
out). Unblinded columns (true_*, tflux_*, libid_cadence) are dropped; -9 placeholders become missing,
as in the contest files. Metric: the contest's weighted multiclass log loss (every class weighs the
same, classes 15 and 64 double).

Arms: ``raw`` (metadata only), ``pipe`` (metadata + light curves given to ``scripts/contest_features.py``
at its defaults). ``--shuffle`` permutes the training labels (leakage control).
"""
import sys, time, json, warnings, argparse, subprocess, tempfile
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='data/plasticc/'); ap.add_argument('--log', default='plasticc.jsonl')
ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
D = Path(a.data)
keep = ['object_id', 'ra', 'decl', 'ddf_bool', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distmod', 'mwebv']
def meta(f):
    m = pd.read_csv(D / f)
    for c in ('hostgal_specz', 'distmod'):
        m[c] = m[c].where(m[c] != -9)
    return m
tr = meta('plasticc_train_metadata.csv.gz')
lc_te_path = D / 'plasticc_test_lightcurves_01.csv.gz'
te_ids = pd.read_csv(lc_te_path, usecols=['object_id']).object_id.unique()
te = meta('plasticc_test_metadata.csv.gz')
te = te[te.object_id.isin(te_ids)].reset_index(drop=True)
classes = np.sort(tr.target.unique())
ytr = np.searchsorted(classes, tr.target.to_numpy())
known = te.true_target.isin(classes).to_numpy()
yte = np.searchsorted(classes, te.true_target.to_numpy())
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
tr, te = tr[keep], te[keep]
t0 = time.time(); info = {}
if a.arm == 'raw':
    Xtr, Xte = tr.drop(columns=['object_id']), te.drop(columns=['object_id'])
else:
    tmp = Path(tempfile.mkdtemp())
    tr.assign(target=classes[ytr]).to_csv(tmp / 'train.csv', index=False); te.to_csv(tmp / 'test.csv', index=False)
    lc = pd.concat([pd.read_csv(D / 'plasticc_train_lightcurves.csv.gz'), pd.read_csv(lc_te_path)], ignore_index=True)
    lc.to_parquet(tmp / 'lightcurves.parquet'); del lc
    cmd = [sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.csv'),
           '--test', str(tmp / 'test.csv'), '--target', 'target', '--id', 'object_id', '--task', 'multiclass',
           '--budget', str(a.budget), '--out-dir', str(tmp / 'out'), '--table', f'lightcurves={tmp / "lightcurves.parquet"}']
    subprocess.run(cmd, check=True)
    Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['target', 'object_id'])
    Xte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Xtr.columns]
    info = dict(n_cols=Xtr.shape[1])
fe_t = time.time() - t0
for c in Xtr.columns:
    if not pd.api.types.is_numeric_dtype(Xtr[c]):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
K = len(classes)
cw = np.where(np.isin(classes, [15, 64]), 2.0, 1.0)
def wloss(y, P):
    P = np.clip(P, 1e-15, 1); P = P / P.sum(1, keepdims=True)
    per = [-np.mean(np.log(P[y == k, k])) for k in range(K) if (y == k).any()]
    w = [cw[k] for k in range(K) if (y == k).any()]
    return float(np.dot(per, w) / np.sum(w))
# Train with weights that make every class count as the metric does.
sw = cw[ytr] / np.bincount(ytr, minlength=K)[ytr]
P = dict(objective='multiclass', num_class=K, learning_rate=0.03, num_leaves=15, min_child_samples=10, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1, seed=a.seed)
va = np.random.default_rng(a.seed).random(len(ytr)) < 0.15
b = lgb.train(P, lgb.Dataset(Xtr[~va], ytr[~va], weight=sw[~va]), 10000, valid_sets=[lgb.Dataset(Xtr[va], ytr[va], weight=sw[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr, weight=sw), int(b.best_iteration * 1.1) + 1).predict(Xte)
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, wlogloss=wloss(yte[known], p[known]),
           wlogloss_uniform=float(np.log(K)), acc=float((p[known].argmax(1) == yte[known]).mean()), best_it=b.best_iteration,
           fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), n_te=int(known.sum()), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
