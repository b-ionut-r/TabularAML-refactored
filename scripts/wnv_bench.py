"""West Nile Virus Prediction (Kaggle 2015, $40k): raw vs FeatureForge on a held-out year.

Data: GitHub ``apnorton/ml-project`` data/ (the contest files; weatherOld.csv is the original
weather table). Rows: mosquito trap tests (date, trap, species, coordinates), with the weather of
station 1 on that date joined; NumMosquitos is dropped (not in the contest's test set). Training
years are 2007, 2009, 2011, 2013: --win 0 holds out 2013, --win 1 holds out 2011 (training on the
earlier years). The holdout's features are the unlabeled rows. Metric: AUC, as the contest.
"""
import sys, time, json, warnings, argparse
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--win', type=int, default=0)
ap.add_argument('--budget', type=float, default=600); ap.add_argument('--kw', default='{}'); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='data/wnv/'); ap.add_argument('--log', default='wnv.jsonl')
a = ap.parse_args()
D = Path(a.data)
tr = pd.read_csv(D / 'train.csv')
w = pd.read_csv(D / 'weatherOld.csv'); w = w[w.Station == 1].drop(columns=['Station'])
for c in w.columns:
    if c not in ('Date', 'CodeSum'):
        w[c] = pd.to_numeric(w[c].replace({'T': '0.005', 'M': np.nan, '-': np.nan}), errors='coerce')
df = tr.merge(w, on='Date', how='left').drop(columns=['NumMosquitos'])
y = df.pop('WnvPresent').to_numpy()
yr = df['Date'].str[:4].astype(int)
hold = [2013, 2011][a.win]
itr, iho = (yr < hold).to_numpy(), (yr == hold).to_numpy()
Xtr, Xho, ytr, yho = df[itr].reset_index(drop=True), df[iho].reset_index(drop=True), y[itr], y[iho]
def raw(X):
    X = X.copy(); X['Date'] = (pd.to_datetime(X['Date']) - pd.Timestamp('2007-01-01')).dt.days.astype(float)
    for c in X.columns:
        if X[c].dtype == object: X[c] = X[c].astype('category')
    return X
def hand(Xtr, Xho, parts):
    """Hand-made features of top public kernels: rows per (date, trap, species) over all rows with
    known features (the contest's 'duplicate rows' signal), calendar week, trailing weather means."""
    A = pd.concat([Xtr, Xho], ignore_index=True)
    if 'dup' in parts:
        A['n_dup'] = A.groupby(['Date', 'Trap', 'Species'])['Trap'].transform('size').astype(float)
    if 'cal' in parts:
        dt = pd.to_datetime(A['Date']); A['week'] = dt.dt.isocalendar().week.astype(float); A['doy'] = dt.dt.dayofyear.astype(float)
    if 'roll' in parts:
        wd = w.copy(); wd['Date'] = pd.to_datetime(wd['Date']); wd = wd.set_index('Date').sort_index()
        num = ['Tmax', 'Tmin', 'Tavg', 'DewPoint', 'WetBulb', 'PrecipTotal', 'AvgSpeed']
        for k in (7, 14, 28):
            r = wd[num].rolling(f'{k}D').mean().add_suffix(f'_r{k}')
            A = A.merge(r.reset_index().assign(Date=lambda d: d['Date'].dt.strftime('%Y-%m-%d')), on='Date', how='left')
    return A.iloc[:len(Xtr)].reset_index(drop=True), A.iloc[len(Xtr):].reset_index(drop=True)
if a.arm.startswith('hand'):
    parts = a.arm.split('_')[1:] or ['dup', 'cal', 'roll']
    Xtr, Xho = hand(Xtr, Xho, parts)
t0 = time.time(); info = {}
if a.arm == 'forge' or a.arm.endswith('_forge'):
    from tabularaml.generate.forge import FeatureForge
    f = FeatureForge(task='binary', time_budget=a.budget, random_state=0, n_jobs=4, verbose=True,
                     **json.loads(a.kw)).fit(Xtr, ytr, X_unlabeled=Xho)
    info = dict(n_added=len(f.new_columns_), gate=f.gate_passed_, time_col=f.time_col_, feats=f.new_columns_[:40])
    Xtr, Xho = f.transform_train(Xtr), f.transform(Xho)
fe_t = time.time() - t0
Xtr, Xho = raw(Xtr), raw(Xho)
for c in Xtr.columns:
    if isinstance(Xtr[c].dtype, pd.CategoricalDtype):
        Xho[c] = pd.Categorical(Xho[c].astype(object), categories=Xtr[c].cat.categories)
P = dict(objective='binary', learning_rate=0.02, num_leaves=15, min_child_samples=50, feature_fraction=0.6,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1, seed=0)
# Early stopping on the latest training year, refit on all.
ytr_year = Xtr['Date'].to_numpy(); cut = np.sort(np.unique(ytr_year))[-1]
last = (pd.Timestamp('2007-01-01') + pd.to_timedelta(ytr_year, 'D')).year == (pd.Timestamp('2007-01-01') + pd.to_timedelta(cut, 'D')).year
b = lgb.train(P, lgb.Dataset(Xtr[~last], ytr[~last]), 5000, valid_sets=[lgb.Dataset(Xtr[last], ytr[last])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xho)
res = dict(arm=a.arm + a.tag, win=a.win, auc=roc_auc_score(yho, p), best_it=b.best_iteration, fe_s=round(fe_t),
           total_s=round(time.time() - t0), n_tr=len(Xtr), n_ho=len(Xho), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
