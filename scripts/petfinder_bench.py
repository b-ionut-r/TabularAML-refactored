"""PetFinder.my Adoption Prediction (Kaggle 2019, $25k): raw vs FeatureForge on held-out rescuers.

Data: GitHub ``antoinewg/PetFinder`` in/ (train.csv, test.csv; no images or sentiment files).
The contest's test rescuers never appear in training, so the holdout is 20% of rescuers (two
seeds). Unlabeled rows: the holdout plus the contest's 3,948 test rows (features only). Target
AdoptionSpeed 0-4; metric quadratic weighted kappa with thresholds matching the training class
shares (the usual decoding of a regression output), and RMSE.
"""
import sys, time, json, warnings, argparse
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import cohen_kappa_score
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--kw', default='{}'); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='data/petfinder/'); ap.add_argument('--log', default='petfinder.jsonl'); ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
D = Path(a.data)
df = pd.read_csv(D / 'train.csv').drop(columns=['PetID'])
te = pd.read_csv(D / 'test.csv').drop(columns=['PetID'])
y = df.pop('AdoptionSpeed').to_numpy(dtype=float)
r = df['RescuerID'].to_numpy(); u = np.unique(r)
ho_r = set(np.random.default_rng(a.seed).choice(u, int(0.2 * len(u)), replace=False))
ho = np.array([x in ho_r for x in r])
Xtr, Xho, ytr, yho = df[~ho].reset_index(drop=True), df[ho].reset_index(drop=True), y[~ho], y[ho]
if a.shuffle:  # leakage control: permuted training labels
    ytr = np.random.default_rng(0).permutation(ytr)
t0 = time.time(); info = {}
if a.arm == 'forge':
    from tabularaml.generate.forge import FeatureForge
    f = FeatureForge(task='regression', time_budget=a.budget, random_state=a.seed, n_jobs=4, verbose=True,
                     **json.loads(a.kw)).fit(Xtr, ytr, X_unlabeled=pd.concat([Xho, te[Xtr.columns]], ignore_index=True))
    info = dict(n_added=len(f.new_columns_), gate=f.gate_passed_, group_col=f.group_col_, text=f.text_cols_, feats=f.new_columns_[:30])
    Xtr, Xho = f.transform_train(Xtr), f.transform(Xho)
fe_t = time.time() - t0
g_tr = Xtr['RescuerID'].astype(str).to_numpy()
for c in Xtr.columns:
    if Xtr[c].dtype == object or isinstance(Xtr[c].dtype, pd.CategoricalDtype):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xho[c] = pd.Categorical(Xho[c].astype(str), categories=cats)
P = dict(objective='regression', learning_rate=0.02, num_leaves=31, min_child_samples=30, feature_fraction=0.6,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, cat_smooth=20, num_threads=4, verbose=-1, seed=a.seed)
gu = np.unique(g_tr); va_g = set(np.random.default_rng(a.seed + 1).choice(gu, int(0.15 * len(gu)), replace=False))
va = np.array([g in va_g for g in g_tr])
b = lgb.train(P, lgb.Dataset(Xtr[~va], ytr[~va]), 10000, valid_sets=[lgb.Dataset(Xtr[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xho)
share = np.cumsum(np.bincount(ytr.astype(int), minlength=5) / len(ytr))[:4]
qwk = cohen_kappa_score(yho.astype(int), np.digitize(p, np.quantile(p, share)), weights='quadratic')
# Cut-points fit on training rows only: the early-stopping model's predictions on its validation rescuers.
qwk_trcut = cohen_kappa_score(yho.astype(int), np.digitize(p, np.quantile(b.predict(Xtr[va]), share)), weights='quadratic')
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, qwk=float(qwk), qwk_trcut=float(qwk_trcut), rmse=float(np.sqrt(np.mean((p - yho) ** 2))), best_it=b.best_iteration,
           fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), n_ho=len(Xho), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
