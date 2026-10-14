"""Google Brain Ventilator Pressure Prediction (Kaggle 2021, $7.5k): raw vs FeatureForge vs the public GBM features.

Data: the contest's train.csv (75,450 breaths x 80 time steps: breath_id, lung attributes R and C, time_step,
the inspiratory valve input u_in, the expiratory valve flag u_out; target pressure), from the public Kaggle copy
``jarupula/google-vpp-train-5-folds`` (train_folds.csv; its kfold column is dropped). A random ``--breaths``
breaths (``--seed``); the contest's test file held new breaths, so 20% of them are held out whole and their
rows are the unlabeled rows. Metric: MAE over the inspiratory phase (u_out == 0), as the contest.
Arms (same bagged LightGBM with an L1 objective: early stopping on 15% of the training breaths, then 3 seeds):
  ``raw``    R, C, time_step, u_in, u_out;
  ``ff``     blind: ``scripts/contest_features.py`` at its defaults (breath_id is a column of the table);
  ``hand``   the top public GBM kernels' per-breath features: cumulative u_in and its area (sum of u_in x dt),
             step index and dt, u_in lags and leads 1-4 and their differences, the breath's mean / max / last
             u_in and u_in minus the mean, u_out lag and lead, R x C, R / C and the R_C combination.
``--shuffle`` permutes the training labels by breath (leakage control).

    python scripts/vpp_bench.py --arm ff --seed 0
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--breaths', type=int, default=10_000)
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default=''); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='/tmp/claude-0/data/vpp/train_folds.csv'); ap.add_argument('--log', default='vpp.jsonl')
ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
df = pd.read_csv(a.data).drop(columns=['kfold', 'id'])
rng = np.random.default_rng(a.seed)
br = rng.choice(df.breath_id.unique(), a.breaths, replace=False)
ho = set(rng.choice(br, a.breaths // 5, replace=False))
df = df[df.breath_id.isin(br)].sort_values(['breath_id', 'time_step']).reset_index(drop=True)
is_ho = df.breath_id.isin(ho).to_numpy()
y = df.pop('pressure').to_numpy()
Xtr, Xte = df[~is_ho].reset_index(drop=True), df[is_ho].reset_index(drop=True)
ytr, yte = y[~is_ho], y[is_ho]
if a.shuffle:  # whole breaths' pressure curves move to other breaths
    b_ids = Xtr.breath_id.unique(); perm = dict(zip(b_ids, np.random.default_rng(0).permutation(b_ids)))
    curves = pd.Series(list(ytr.reshape(-1, 80)), index=Xtr.breath_id.to_numpy()[::80])
    ytr = np.concatenate([curves[perm[b]] for b in curves.index])
gtr = Xtr.breath_id.to_numpy()
t0 = time.time()
def hand(X):
    X = X.copy(); g = X.groupby('breath_id')
    X['step'] = g.cumcount(); X['dt'] = g['time_step'].diff().fillna(0)
    X['u_in_cumsum'] = g['u_in'].cumsum(); X['area'] = (X['u_in'] * X['dt']).groupby(X['breath_id']).cumsum()
    for k in (1, 2, 3, 4):
        X[f'u_in_lag{k}'] = g['u_in'].shift(k); X[f'u_in_lead{k}'] = g['u_in'].shift(-k)
        X[f'u_in_diff{k}'] = X['u_in'] - X[f'u_in_lag{k}']
    X['u_out_lag1'] = g['u_out'].shift(1); X['u_out_lead1'] = g['u_out'].shift(-1)
    for st in ('mean', 'max', 'last'):
        X[f'u_in_{st}'] = g['u_in'].transform(st)
    X['u_in_dev'] = X['u_in'] - X['u_in_mean']
    X['RC'] = X['R'] * X['C']; X['R_div_C'] = X['R'] / X['C']; X['R_C'] = (X['R'].astype(str) + '_' + X['C'].astype(str)).astype('category')
    return X
if a.arm == 'ff':
    tmp = Path(tempfile.mkdtemp(dir='/tmp/claude-0'))
    Xtr.assign(pressure=ytr).to_parquet(tmp / 'train.parquet'); Xte.to_parquet(tmp / 'test.parquet')
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
                    '--test', str(tmp / 'test.parquet'), '--target', 'pressure', '--task', 'regression', '--budget', str(a.budget),
                    '--out-dir', str(tmp / 'out')] + (a.extra.split() if a.extra else []), check=True)
    Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['pressure'])
    Xte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Xtr.columns]
    shutil.rmtree(tmp, ignore_errors=True)
elif a.arm == 'hand':
    Xtr, Xte = hand(Xtr), hand(Xte)
fe_t = time.time() - t0
Xtr = Xtr.drop(columns=['breath_id'], errors='ignore'); Xte = Xte.drop(columns=['breath_id'], errors='ignore')
for X in (Xtr, Xte):
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
            X[c] = X[c].astype(str).astype('category')
for c in Xtr.columns:
    if isinstance(Xtr[c].dtype, pd.CategoricalDtype):
        Xte[c] = pd.Categorical(Xte[c].astype(str), categories=Xtr[c].cat.categories)
P = dict(objective='l1', learning_rate=0.05, num_leaves=127, min_child_samples=50, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1)
ub = np.unique(gtr); vb = set(np.random.default_rng(a.seed + 1).choice(ub, int(0.15 * len(ub)), replace=False))
va = np.isin(gtr, list(vb)); fit = ~va
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr[fit], ytr[fit]), 20000, valid_sets=[lgb.Dataset(Xtr[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
insp = Xte['u_out'].to_numpy() == 0
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, mae=float(np.mean(np.abs(p - yte)[insp])),
           best_it=b.best_iteration, n_cols=Xtr.shape[1], fe_s=round(fe_t), total_s=round(time.time() - t0))
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
