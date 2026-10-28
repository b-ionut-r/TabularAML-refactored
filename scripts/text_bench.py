"""Text-tabular tables (MulTaBench, Hugging Face ``multabench/<name>``): raw vs FeatureForge on held-out rows.

Random 80/20 holdout per seed (tables capped at 50k rows); the holdout's features are
the unlabeled rows. Metric: RMSE (regression) or logloss (classification), lower is
better. String columns go to the judge as categoricals.
"""
import sys, time, json, warnings, argparse
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--arm', default='raw')
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--budget', type=float, default=600)
ap.add_argument('--kw', default='{}'); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='data/multabench/'); ap.add_argument('--log', default='text_bench.jsonl')
a = ap.parse_args()
d = Path(a.data) / a.name
meta = json.loads((d / 'metadata.json').read_text())
df = pd.read_parquet(d / 'data.parquet')
for c in [meta.get('image_col')]:
    if c and c in df: df = df.drop(columns=[c])
df = df[df[meta['target']].notna()]
if len(df) > 50_000:
    df = df.sample(50_000, random_state=0)
df = df.reset_index(drop=True)
reg = meta['task_type'] == 'reg'
y = df.pop(meta['target'])
y = y.astype(float) if reg else pd.Series(pd.factorize(y.astype(str), sort=True)[0])
for c in df.columns:
    if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype):
        df[c] = df[c].astype(object)
K = 0 if reg else int(y.max()) + 1
strat = None if reg else y
itr, ite = train_test_split(np.arange(len(df)), test_size=0.2, random_state=a.seed, stratify=strat)
Xtr, Xte = df.iloc[itr].reset_index(drop=True), df.iloc[ite].reset_index(drop=True)
ytr, yte = y.iloc[itr].reset_index(drop=True), y.iloc[ite].reset_index(drop=True)
t0 = time.time(); info = {}
if a.arm == 'forge':
    from tabularaml.generate.forge import FeatureForge
    task = 'regression' if reg else ('binary' if K == 2 else 'multiclass')
    f = FeatureForge(task=task, time_budget=a.budget, random_state=a.seed, n_jobs=4, verbose=True,
                     **json.loads(a.kw)).fit(Xtr, ytr, X_unlabeled=Xte)
    info = dict(n_added=len(f.new_columns_), gate=f.gate_passed_, text=f.text_cols_, feats=f.new_columns_[:20])
    Xtr, Xte = f.transform_train(Xtr), f.transform(Xte)
fe_t = time.time() - t0
for c in [c for c in Xtr.columns if pd.api.types.is_datetime64_any_dtype(Xtr[c])]:
    Xtr[c] = (Xtr[c] - pd.Timestamp('1970-01-01')).dt.days.astype(float); Xte[c] = (Xte[c] - pd.Timestamp('1970-01-01')).dt.days.astype(float)
for c in [c for c in Xtr.columns if Xtr[c].dtype == object or isinstance(Xtr[c].dtype, pd.CategoricalDtype)]:
    cats = pd.Index(pd.unique(Xtr[c].astype(str)))
    Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
P = dict(objective='regression' if reg else ('binary' if K == 2 else 'multiclass'), learning_rate=0.05, num_leaves=31,
         min_child_samples=20, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, cat_smooth=20,
         num_threads=4, verbose=-1, seed=a.seed, **({} if K <= 2 else {'num_class': K}))
perm = np.random.default_rng(a.seed).permutation(len(Xtr)); tr, va = np.sort(perm[:int(.85 * len(perm))]), np.sort(perm[int(.85 * len(perm)):])
b = lgb.train(P, lgb.Dataset(Xtr.iloc[tr], ytr.iloc[tr]), 10000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr.iloc[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte)
score = float(np.sqrt(np.mean((p - yte.to_numpy()) ** 2))) if reg else float(log_loss(yte, p, labels=list(range(K))))
res = dict(name=a.name, arm=a.arm + a.tag, seed=a.seed, metric='rmse' if reg else 'logloss', score=score,
           fe_s=round(fe_t), total_s=round(time.time() - t0), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
