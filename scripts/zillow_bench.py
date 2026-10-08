"""Zillow Prize: Zillow's Home Value Prediction (Kaggle 2017-18, $1.2M total): raw vs FeatureForge on held-out months.

Data: Hugging Face ``Kun-05/ML-Zillow-Prize`` (the contest zip). Rows: the 2016 and 2017 sales
(train_2016_v2, train_2017) with that year's property table joined. Target: logerror (log Zestimate
minus log sale price). Holdouts by time, like the contest's later test months: --win 0 = sales from
2017-07-01 on (the last three months), --win 1 = 2017-04-01 to 06-30, training on all earlier sales.
The holdout's features are the unlabeled rows. Metric: MAE (lower is better), as the contest.
Prepare ``train.parquet`` with ``--prep`` from ``z.zip`` once.
"""
import sys, time, json, warnings, argparse
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--win', type=int, default=0)
ap.add_argument('--budget', type=float, default=1200); ap.add_argument('--kw', default='{}'); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='data/zillow/'); ap.add_argument('--log', default='zillow.jsonl')
ap.add_argument('--prep', action='store_true')
a = ap.parse_args()
D = Path(a.data)
if a.prep:
    import zipfile
    z = zipfile.ZipFile(D / 'z.zip'); out = []
    for yr, f in ((2016, 'train_2016_v2.csv'), (2017, 'train_2017.csv')):
        t = pd.read_csv(z.open(f), parse_dates=['transactiondate']); ids = set(t.parcelid)
        p = pd.concat(ch[ch.parcelid.isin(ids)] for ch in pd.read_csv(z.open(f'properties_{yr}.csv'), chunksize=300_000, low_memory=False))
        out.append(t.merge(p, on='parcelid', how='left'))
    d = pd.concat(out, ignore_index=True)
    for c in d.columns:
        if d[c].dtype == object: d[c] = d[c].astype(str).where(d[c].notna(), None)
    d.to_parquet(D / 'train.parquet'); sys.exit()
df = pd.read_parquet(D / 'train.parquet').drop(columns=['parcelid', 'year'], errors='ignore')
df = df.sort_values('transactiondate', kind='stable').reset_index(drop=True)
y = df.pop('logerror').to_numpy(dtype=float)
lo, hi = [(pd.Timestamp('2017-07-01'), pd.Timestamp('2018-01-01')), (pd.Timestamp('2017-04-01'), pd.Timestamp('2017-07-01'))][a.win]
dt = df['transactiondate']
tr, ho = (dt < lo).to_numpy(), ((dt >= lo) & (dt < hi)).to_numpy()
Xtr, Xho, ytr, yho = df[tr].reset_index(drop=True), df[ho].reset_index(drop=True), y[tr], y[ho]
def raw(X):
    X = X.copy(); X['transactiondate'] = (X['transactiondate'] - pd.Timestamp('2016-01-01')).dt.days.astype(float)
    for c in X.columns:
        if X[c].dtype == object: X[c] = X[c].astype('category')
    return X
t0 = time.time(); info = {}
if a.arm == 'forge':
    from tabularaml.generate.forge import FeatureForge
    f = FeatureForge(task='regression', time_budget=a.budget, random_state=0, n_jobs=4, verbose=True,
                     **json.loads(a.kw)).fit(Xtr, ytr, X_unlabeled=Xho)
    info = dict(n_added=len(f.new_columns_), gate=f.gate_passed_, time_col=f.time_col_, feats=f.new_columns_[:40])
    Xtr, Xho = f.transform_train(Xtr), f.transform(Xho)
fe_t = time.time() - t0
Xtr, Xho = raw(Xtr), raw(Xho)
for c in Xtr.columns:
    if isinstance(Xtr[c].dtype, pd.CategoricalDtype):
        Xho[c] = pd.Categorical(Xho[c].astype(object), categories=Xtr[c].cat.categories)
P = dict(objective='regression_l1', learning_rate=0.02, num_leaves=31, min_child_samples=200, feature_fraction=0.6,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1, seed=0)
n_va = int(0.1 * len(Xtr)); i_tr, i_va = np.arange(len(Xtr) - n_va), np.arange(len(Xtr) - n_va, len(Xtr))
b = lgb.train(P, lgb.Dataset(Xtr.iloc[i_tr], ytr[i_tr]), 10000, valid_sets=[lgb.Dataset(Xtr.iloc[i_va], ytr[i_va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xho)
res = dict(arm=a.arm + a.tag, win=a.win, mae=float(np.mean(np.abs(p - yho))), mae_median=float(np.mean(np.abs(np.median(ytr) - yho))),
           best_it=b.best_iteration, fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), n_ho=len(Xho), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
