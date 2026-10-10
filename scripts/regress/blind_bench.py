"""Blind contest bench: raw columns vs scripts/contest_features.py at its defaults, same bagged LightGBM.
Holdout: rows with --time >= --cut (time split like the real test), else a random --frac stratified split."""
import sys, time, json, argparse, subprocess, tempfile, shutil
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
ap = argparse.ArgumentParser()
ap.add_argument('--data'); ap.add_argument('--target'); ap.add_argument('--name')
ap.add_argument('--time'); ap.add_argument('--cut', type=float); ap.add_argument('--frac', type=float, default=0.2)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--arm', default='final'); ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--drop', default=''); ap.add_argument('--keep', default='', help='save the feature files here')
ap.add_argument('--budget', default='900'); ap.add_argument('--extra', default='')
ap.add_argument('--repo', default='/home/user/TabularAML-refactored'); ap.add_argument('--log', default='/tmp/claude-0/runs/blind.jsonl')
a = ap.parse_args()
d = pd.read_parquet(a.data)
y = d.pop(a.target)
for c in a.drop.split(',') if a.drop else []:
    d.pop(c)
binary = y.nunique() == 2
multi = (not binary) and (not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 30)
if multi:
    classes = sorted(pd.unique(y.astype(str))); y = y.astype(str).map({c: i for i, c in enumerate(classes)}).astype(int)
if a.time:
    ho = d[a.time].to_numpy() >= a.cut
    itr, ite = np.flatnonzero(~ho), np.flatnonzero(ho)
else:
    itr, ite = train_test_split(np.arange(len(d)), test_size=a.frac, random_state=a.seed, stratify=y if (binary or multi) else None)
    itr, ite = np.sort(itr), np.sort(ite)
tr, te = d.iloc[itr].reset_index(drop=True), d.iloc[ite].reset_index(drop=True)
ytr, yte = y.iloc[itr].to_numpy(), y.iloc[ite].to_numpy()
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
t0 = time.time()
if a.arm == 'raw':
    Xtr, Xte = tr, te
else:
    tmp = Path(tempfile.mkdtemp(dir='/tmp/claude-0/tmp'))
    tr.assign(**{a.target: np.array(classes)[ytr] if multi else ytr}).to_parquet(tmp / 'train.parquet'); te.to_parquet(tmp / 'test.parquet')
    cmd = [sys.executable, str(Path(a.repo) / 'scripts' / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
           '--test', str(tmp / 'test.parquet'), '--target', a.target, '--budget', a.budget, '--out-dir', str(tmp / 'out')] + (a.extra.split() if a.extra else [])
    subprocess.run(cmd, check=True)
    Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=[a.target])
    Xte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Xtr.columns]
    if a.keep: Path(a.keep).mkdir(parents=True, exist_ok=True); Xtr.assign(_y=ytr).to_parquet(Path(a.keep) / 'tr.parquet'); Xte.assign(_y=yte).to_parquet(Path(a.keep) / 'te.parquet')
    shutil.rmtree(tmp, ignore_errors=True)
fe_t = time.time() - t0
Xtr, Xte = Xtr.copy(), Xte.copy()
for c in Xtr.columns:
    if not pd.api.types.is_numeric_dtype(Xtr[c]):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
P = dict(objective='binary' if binary else ('multiclass' if multi else 'regression'), **({'num_class': len(classes)} if multi else {}), learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1, max_bin=255, max_cat_to_onehot=8, cat_smooth=50)
# early stopping on the latest 15% of training rows (time order) or a random 15%
n = len(Xtr); fit, va = (np.arange(int(0.85 * n)), np.arange(int(0.85 * n), n)) if a.time else (lambda p: (np.sort(p[:int(0.85 * n)]), np.sort(p[int(0.85 * n):])))(np.random.default_rng(a.seed + 1).permutation(n))
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 5000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros((len(Xte), len(classes))) if multi else np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
res = dict(name=a.name, arm=a.arm + ('_shuffled' if a.shuffle else ''), seed=a.seed, n_tr=len(Xtr), n_te=len(Xte), n_cols=Xtr.shape[1],
           best_it=b.best_iteration, fe_s=round(fe_t), total_s=round(time.time() - t0))
if multi:
    pp = np.clip(p, 1e-15, 1); pp /= pp.sum(1, keepdims=True)
    res.update(logloss=float(log_loss(yte, pp, labels=np.arange(len(classes)))), acc=float(np.mean(p.argmax(1) == yte)))
elif binary:
    res.update(auc=float(roc_auc_score(yte, p)), logloss=float(log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6))))
else:
    res.update(rmse=float(mean_squared_error(yte, p) ** 0.5))
print('RESULT', json.dumps(res)); open(a.log, 'a').write(json.dumps(res) + '\n')
