"""Telstra Network Disruptions (Kaggle 2016, $30k): fault severity (0/1/2) of an outage report at a location.

Data: the contest files (public Kaggle copy ``yifanxie/telstra-competition-dataset``): train.csv (7,381 reports: id,
location, fault_severity), test.csv (11,171 reports), and four tables keyed by report id: event_type, log_feature
(with volume), resource_type, severity_type. The test reports were a random draw of the same locations and period,
so the holdout is a stratified random 20% of train (``--seed``); the held-out rows and the contest's test rows are
the unlabeled rows. Metric: multiclass log loss, as the contest.
Arms (same bagged LightGBM: early stopping on 15% of the training rows, then 3 seeds on all of them):
  ``raw``     location as a number;
  ``ff``      blind: ``scripts/contest_features.py`` at its defaults with the four tables (``--table name=...:id``);
  ``hand``    the top public kernels' features: per report, event / resource / severity one-hots, log feature
              volumes pivoted, their count and sum; per location, its report count;
  ``hand_order`` adds the winners' "magic": the report's position in severity_type.csv within its location (the
              file lists each location's reports in time order) and that position over the location's count. This
              is a file-assembly signal, not a column of the data.
``--shuffle`` permutes the training labels (leakage control).

    python scripts/telstra_bench.py --arm ff --seed 0
"""
import sys, re, time, json, warnings, argparse, subprocess, tempfile, shutil
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default=''); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='/tmp/claude-0/data/telstra/'); ap.add_argument('--log', default='telstra.jsonl')
ap.add_argument('--shuffle', action='store_true')
ap.add_argument('--drop-re', default='', help='ablation: drop blind columns matching this regex')
ap.add_argument('--ff-cache', default='', help='directory to keep / reuse the blind output (ablation arms ff+group,...)')
a = ap.parse_args()
D = Path(a.data)
tr, te_real = pd.read_csv(D / 'train.csv'), pd.read_csv(D / 'test.csv')
y = tr.pop('fault_severity').to_numpy()
itr, iho = train_test_split(np.arange(len(tr)), test_size=0.2, random_state=a.seed, stratify=y)
Xtr, Xho = tr.iloc[itr].reset_index(drop=True), tr.iloc[iho].reset_index(drop=True)
ytr, yho = y[itr], y[iho]
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
Xun = pd.concat([Xho, te_real], ignore_index=True)  # held-out + the contest's test reports, all unlabeled
tables = {n: pd.read_csv(D / f'{n}.csv') for n in ('event_type', 'log_feature', 'resource_type', 'severity_type')}
t0 = time.time()
def num(s):
    return s.str.split(' ').str[-1].astype(int)
def hand_groups(A):
    G = {'loc': pd.DataFrame({'loc': num(A.location)})}
    G['loc_count'] = pd.DataFrame({'loc_count': G['loc']['loc'].map(G['loc']['loc'].value_counts())})
    for n in ('event_type', 'resource_type', 'severity_type'):
        P = pd.crosstab(tables[n].id, tables[n][n]); P.columns = [f'{n}_{num(pd.Series(P.columns)).iloc[i]}' for i in range(P.shape[1])]
        G[n] = A[['id']].merge(P, left_on='id', right_index=True, how='left').drop(columns='id')
    L = tables['log_feature']
    P = L.pivot_table(index='id', columns='log_feature', values='volume', aggfunc='sum', fill_value=0)
    P.columns = [f'log_{num(pd.Series([c])).iloc[0]}' for c in P.columns]
    G['log'] = A[['id']].merge(P, left_on='id', right_index=True, how='left').drop(columns='id')
    agg = L.groupby('id')['volume'].agg(['count', 'sum', 'max']).add_prefix('vol_')
    G['vol'] = A[['id']].merge(agg, left_on='id', right_index=True, how='left').drop(columns='id')
    return {k: v.reset_index(drop=True) for k, v in G.items()}
if a.arm.startswith('ff'):
    cache = Path(a.ff_cache) if a.ff_cache else None
    if cache and (cache / 'train_features.parquet').exists():
        Ftr = pd.read_parquet(cache / 'train_features.parquet'); Fun = pd.read_parquet(cache / 'test_features.parquet')
    else:
        tmp = Path(tempfile.mkdtemp(dir='/tmp/claude-0'))
        Xtr.assign(fault_severity=ytr).to_parquet(tmp / 'train.parquet'); Xun.to_parquet(tmp / 'test.parquet')
        cmd = [sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
               '--test', str(tmp / 'test.parquet'), '--target', 'fault_severity', '--id', 'id', '--task', 'multiclass',
               '--budget', str(a.budget), '--out-dir', str(tmp / 'out')]
        for n, T in tables.items():
            T.to_parquet(tmp / f'{n}.parquet'); cmd += ['--table', f'{n}={tmp / (n + ".parquet")}:id']
        subprocess.run(cmd + (a.extra.split() if a.extra else []), check=True)
        Ftr = pd.read_parquet(tmp / 'out' / 'train_features.parquet'); Fun = pd.read_parquet(tmp / 'out' / 'test_features.parquet')
        if cache:
            cache.mkdir(parents=True, exist_ok=True); Ftr.to_parquet(cache / 'train_features.parquet'); Fun.to_parquet(cache / 'test_features.parquet')
        shutil.rmtree(tmp, ignore_errors=True)
    assert (Ftr.id.to_numpy() == Xtr.id.to_numpy()).all() and (Fun.id.to_numpy() == Xun.id.to_numpy()).all()
    nho = len(Xho)
    if '+' in a.arm:  # ablation: add hand groups to the blind output
        G = hand_groups(pd.concat([Xtr, Xun], ignore_index=True))
        add = pd.concat([G[g].add_prefix('h_') for g in a.arm.split('+', 1)[1].split(',')], axis=1)
        Ftr = pd.concat([Ftr.reset_index(drop=True), add.iloc[:len(Xtr)].reset_index(drop=True)], axis=1)
        Fun = pd.concat([Fun.reset_index(drop=True), add.iloc[len(Xtr):].reset_index(drop=True)], axis=1)
    Xtr = Ftr.drop(columns=['id', 'fault_severity']); Xho = Fun.drop(columns=['id'])[Xtr.columns].iloc[:nho].reset_index(drop=True)
    if a.drop_re:
        keep = [c for c in Xtr.columns if not re.search(a.drop_re, c)]; Xtr, Xho = Xtr[keep], Xho[keep]
elif a.arm.startswith('hg'):  # ablation: the hand groups minus the listed ones (hg-log,vol)
    A = pd.concat([Xtr, Xun], ignore_index=True); G = hand_groups(A)
    drop = a.arm.split('-', 1)[1].split(',') if '-' in a.arm else []
    F = pd.concat([v for k, v in G.items() if k not in drop], axis=1)
    Xtr, Xho = F.iloc[:len(Xtr)].reset_index(drop=True), F.iloc[len(Xtr):len(Xtr) + len(Xho)].reset_index(drop=True)
elif a.arm.startswith('hand'):
    A = pd.concat([Xtr, Xun], ignore_index=True)
    F = pd.DataFrame({'id': A.id, 'loc': num(A.location)})
    F['loc_count'] = F.groupby('loc')['id'].transform('size')
    for n, col in (('event_type', 'event_type'), ('resource_type', 'resource_type'), ('severity_type', 'severity_type')):
        P = pd.crosstab(tables[n].id, tables[n][col]); P.columns = [f'{n}_{num(pd.Series(P.columns)).iloc[i]}' for i in range(P.shape[1])]
        F = F.merge(P, left_on='id', right_index=True, how='left')
    L = tables['log_feature']
    P = L.pivot_table(index='id', columns='log_feature', values='volume', aggfunc='sum', fill_value=0)
    P.columns = [f'log_{num(pd.Series([c])).iloc[0]}' for c in P.columns]
    agg = L.groupby('id')['volume'].agg(['count', 'sum', 'max']).add_prefix('vol_')
    F = F.merge(P, left_on='id', right_index=True, how='left').merge(agg, left_on='id', right_index=True, how='left')
    if a.arm == 'hand_order':
        S = tables['severity_type'][['id']].copy(); S['row'] = np.arange(len(S))
        F = F.merge(S, on='id', how='left')
        F['order_in_loc'] = F.groupby('loc')['row'].rank(method='first')
        F['order_frac'] = F['order_in_loc'] / F['loc_count']
        F = F.drop(columns=['row'])
    F = F.drop(columns=['id'])
    Xtr, Xho = F.iloc[:len(Xtr)].reset_index(drop=True), F.iloc[len(Xtr):len(Xtr) + len(Xho)].reset_index(drop=True)
else:
    Xtr, Xho = pd.DataFrame({'loc': num(Xtr.location)}), pd.DataFrame({'loc': num(Xho.location)})
fe_t = time.time() - t0
for X in (Xtr, Xho):
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
            X[c] = X[c].astype(str).astype('category')
for c in Xtr.columns:
    if isinstance(Xtr[c].dtype, pd.CategoricalDtype):
        Xho[c] = pd.Categorical(Xho[c].astype(str), categories=Xtr[c].cat.categories)
P = dict(objective='multiclass', num_class=3, learning_rate=0.02, num_leaves=31, min_child_samples=20, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1)
perm = np.random.default_rng(a.seed + 1).permutation(len(Xtr)); n = int(0.85 * len(Xtr))
fit, va = np.sort(perm[:n]), np.sort(perm[n:])
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 10000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros((len(Xho), 3))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xho) / 3
res = dict(arm=a.arm + a.tag + (f'_drop[{a.drop_re}]' if a.drop_re else '') + ('_shuffled' if a.shuffle else ''), seed=a.seed, mlogloss=log_loss(yho, p, labels=[0, 1, 2]),
           best_it=b.best_iteration, n_cols=Xtr.shape[1], fe_s=round(fe_t), total_s=round(time.time() - t0))
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
