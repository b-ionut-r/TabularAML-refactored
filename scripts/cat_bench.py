"""Caterpillar Tube Pricing (Kaggle 2015, $30k): raw vs FeatureForge on held-out tube assemblies.

Data: GitHub ``timmyshen/Cat_Tube`` competition_data/ (the contest's 21 tables). Raw columns: the
price quotes (supplier, quote date, annual usage, minimum order, bracket pricing, quantity) with
the one-row-per-assembly tables joined as given: tube (dimensions, bends, ends), bill_of_materials
(component_id_1..8, quantity_1..8) and specs (spec1..10). The component attribute tables (comp_*)
are not joined in the raw arm. Holdout: 20% of tube assemblies (the contest's test assemblies are
new), two seeds. Target log1p(cost); metric RMSLE (lower is better).
"""
import sys, time, json, warnings, argparse
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--kw', default='{}'); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='data/cat/'); ap.add_argument('--log', default='cat.jsonl'); ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
D = Path(a.data)
tr = pd.read_csv(D / 'train_set.csv')
for t in ('tube.csv', 'bill_of_materials.csv', 'specs.csv'):
    tr = tr.merge(pd.read_csv(D / t), on='tube_assembly_id', how='left')
y = np.log1p(tr.pop('cost').to_numpy(dtype=float))
if a.arm.startswith('hand'):
    # Hand-made features of top kernels: component attributes per bill-of-materials slot.
    comp = pd.concat([pd.read_csv(f)[lambda d: [c for c in d.columns if c in ('component_id', 'weight', 'component_type_id')]]
                      for f in sorted(D.glob('comp_*.csv'))], ignore_index=True).drop_duplicates('component_id')
    w = comp.set_index('component_id')['weight']
    W = np.column_stack([tr[f'component_id_{k}'].map(w).to_numpy(dtype=float) * tr[f'quantity_{k}'].to_numpy(dtype=float) for k in range(1, 9)])
    tr['bom_weight'] = np.nansum(W, 1); tr['bom_wmax'] = np.nanmax(np.where(np.isnan(W), -1, W), 1)
    tr['bom_n'] = sum(tr[f'component_id_{k}'].notna() for k in range(1, 9)).astype(float)
    tr['bom_qty'] = np.nansum(np.column_stack([tr[f'quantity_{k}'] for k in range(1, 9)]), 1)
    tr['n_specs'] = sum(tr[f'spec{k}'].notna() for k in range(1, 11)).astype(float)
    tr['inv_qty'] = 1 / tr['quantity'].clip(lower=1)
ta = tr['tube_assembly_id'].to_numpy()
u = np.unique(ta); ho_ta = set(np.random.default_rng(a.seed).choice(u, int(0.2 * len(u)), replace=False))
ho = np.array([t in ho_ta for t in ta]); itr = ~ho
X = tr
Xtr, Xho, ytr, yho = X[itr].reset_index(drop=True), X[ho].reset_index(drop=True), y[itr], y[ho]
perm = np.random.default_rng(0).permutation(len(ytr)) if a.shuffle else np.arange(len(ytr))  # leakage control
ytr = ytr[perm]
t0 = time.time(); info = {}
if a.arm.startswith('pipe'):
    # The one-command pipeline on the contest's files as given: quotes plus every other table.
    import subprocess, tempfile
    q = pd.read_csv(D / 'train_set.csv')
    qtr, qho = q[itr].reset_index(drop=True), q[ho].reset_index(drop=True)
    qtr['cost'] = qtr['cost'].to_numpy()[perm]
    tmp = Path(tempfile.mkdtemp())
    qtr.to_csv(tmp / 'train.csv', index=False); qho.drop(columns=['cost']).to_csv(tmp / 'test.csv', index=False)
    tabs = ['tube', 'bill_of_materials', 'specs', 'components'] + sorted(f.stem for f in D.glob('comp_*.csv'))
    cmd = [sys.executable, str(Path(__file__).resolve().parents[0] / 'contest_features.py'), '--train', str(tmp / 'train.csv'),
           '--test', str(tmp / 'test.csv'), '--target', 'cost', '--task', 'regression', '--log-target', '--budget', str(a.budget),
           '--out-dir', str(tmp / 'out')] + sum([['--table', f'{t}={D / (t + ".csv")}'] for t in tabs], [])
    subprocess.run(cmd, check=True)
    Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['cost'])
    Xho = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Xtr.columns]
    info = dict(n_cols=Xtr.shape[1])
if a.arm.endswith('forge') and not a.arm.startswith('pipe'):
    from tabularaml.generate.forge import FeatureForge
    f = FeatureForge(task='regression', time_budget=a.budget, random_state=a.seed, n_jobs=4, verbose=True,
                     **json.loads(a.kw)).fit(Xtr, ytr, X_unlabeled=Xho)
    info = dict(n_added=len(f.new_columns_), gate=f.gate_passed_, group_col=f.group_col_, feats=f.new_columns_[:40])
    Xtr, Xho = f.transform_train(Xtr), f.transform(Xho)
fe_t = time.time() - t0
g_tr = Xtr.pop('tube_assembly_id').to_numpy(); Xho = Xho.drop(columns=['tube_assembly_id'])
for c in Xtr.columns:
    if Xtr[c].dtype == object or isinstance(Xtr[c].dtype, pd.CategoricalDtype):
        if c == 'quote_date':
            Xtr[c] = (pd.to_datetime(Xtr[c].astype(str)) - pd.Timestamp('1980-01-01')).dt.days.astype(float)
            Xho[c] = (pd.to_datetime(Xho[c].astype(str)) - pd.Timestamp('1980-01-01')).dt.days.astype(float)
            continue
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xho[c] = pd.Categorical(Xho[c].astype(str), categories=cats)
P = dict(objective='regression', learning_rate=0.03, num_leaves=63, min_child_samples=10, feature_fraction=0.6,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1, seed=a.seed)
gu = np.unique(g_tr); va_g = set(np.random.default_rng(a.seed + 1).choice(gu, int(0.15 * len(gu)), replace=False))
va = np.array([g in va_g for g in g_tr])
b = lgb.train(P, lgb.Dataset(Xtr[~va], ytr[~va]), 20000, valid_sets=[lgb.Dataset(Xtr[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xho)
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, rmsle=float(np.sqrt(np.mean((p - yho) ** 2))), best_it=b.best_iteration,
           fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), n_ho=len(Xho), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
