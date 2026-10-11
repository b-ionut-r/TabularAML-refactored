"""Homesite Quote Conversion (Kaggle 2016, $20k): raw vs FeatureForge vs the public hand features.

Data: the contest's train.csv (260,753 quotes, 296 anonymous Field / CoverageField / SalesField / PersonalField /
PropertyField / GeographicField columns, Original_Quote_Date; QuoteConversion_Flag 1 = bought, 19%) and test.csv
(173,836 quotes), from the public Kaggle copy ``thanhvv/homesite-quote-conversion``. The test quotes span the same
dates as the training quotes, so the holdout is a stratified random 20% (``--seed``); the held-out rows and the
contest's test rows are the unlabeled rows. Metric: AUC, as the contest.
Arms (same bagged LightGBM: early stopping on 15% of the training rows, then 3 seeds on all of them):
  ``raw``    the columns as given (strings as categories, the date as days);
  ``ff``     blind: ``scripts/contest_features.py`` at its defaults;
  ``hand``   the top public kernels' features: the quote date's year, month and weekday; per row the count of
             -1s, zeros and missing values; Field10 read as a number; each string column's count over training and
             test rows; and the winners' "golden features": differences of every pair of the 12 raw columns a
             LightGBM fit on the training rows ranks most important.
``--shuffle`` permutes the training labels (leakage control).

    python scripts/homesite_bench.py --arm ff --seed 0
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil, itertools
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default=''); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='/tmp/claude-0/data/homesite/'); ap.add_argument('--log', default='homesite.jsonl')
ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
D = Path(a.data)
tr, te_real = pd.read_csv(D / 'train.csv'), pd.read_csv(D / 'test.csv')
y = tr.pop('QuoteConversion_Flag').to_numpy()
tr, te_real = tr.drop(columns=['QuoteNumber']), te_real.drop(columns=['QuoteNumber'])
itr, iho = train_test_split(np.arange(len(tr)), test_size=0.2, random_state=a.seed, stratify=y)
Xtr, Xho = tr.iloc[itr].reset_index(drop=True), tr.iloc[iho].reset_index(drop=True)
ytr, yho = y[itr], y[iho]
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
Xun = pd.concat([Xho, te_real], ignore_index=True)
t0 = time.time()
def model_view(X):
    X = X.copy(); X['Original_Quote_Date'] = (pd.to_datetime(X['Original_Quote_Date']) - pd.Timestamp('2013-01-01')).dt.days
    return X
if a.arm == 'ff':
    tmp = Path(tempfile.mkdtemp(dir='/tmp/claude-0'))
    Xtr.assign(QuoteConversion_Flag=ytr).to_parquet(tmp / 'train.parquet'); Xun.to_parquet(tmp / 'test.parquet')
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
                    '--test', str(tmp / 'test.parquet'), '--target', 'QuoteConversion_Flag', '--task', 'binary', '--budget', str(a.budget),
                    '--out-dir', str(tmp / 'out')] + (a.extra.split() if a.extra else []), check=True)
    Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['QuoteConversion_Flag'])
    Xho = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Xtr.columns].iloc[:len(Xho)].reset_index(drop=True)
    shutil.rmtree(tmp, ignore_errors=True)
    for X in (Xtr, Xho):
        if X['Original_Quote_Date'].dtype == object or str(X['Original_Quote_Date'].dtype).startswith(('str', 'string')):
            X['Original_Quote_Date'] = model_view(X[['Original_Quote_Date']])['Original_Quote_Date']
else:
    A = pd.concat([Xtr, Xun], ignore_index=True)
    if a.arm == 'hand':
        d = pd.to_datetime(A['Original_Quote_Date'])
        new = {'year': d.dt.year, 'month': d.dt.month, 'weekday': d.dt.weekday}
        num = A.select_dtypes('number')
        new['n_neg1'] = (num == -1).sum(1); new['n_zero'] = (num == 0).sum(1); new['n_missing'] = A.isna().sum(1)
        new['Field10_num'] = pd.to_numeric(A['Field10'].astype(str).str.replace(',', ''), errors='coerce')
        for c in A.columns:
            if not pd.api.types.is_numeric_dtype(A[c]) and c != 'Original_Quote_Date':
                k = pd.factorize(A[c].astype(object).fillna('nan').astype(str))[0]; new['n_' + c] = np.bincount(k)[k]
        M = model_view(A.iloc[:len(Xtr)])
        for c in M.columns:
            if not pd.api.types.is_numeric_dtype(M[c]):
                M[c] = M[c].astype(str).astype('category')
        imp = lgb.train(dict(objective='binary', learning_rate=0.1, num_leaves=31, num_threads=4, verbose=-1, seed=0),
                        lgb.Dataset(M, ytr), 200).feature_importance('gain')
        top = [c for c in M.columns[np.argsort(-imp)] if pd.api.types.is_numeric_dtype(A[c])][:12]
        for p, q in itertools.combinations(top, 2):
            new[f'{p}-{q}'] = A[p] - A[q]
        A = pd.concat([A, pd.DataFrame(new)], axis=1)
    A = model_view(A)
    Xtr, Xho = A.iloc[:len(Xtr)].reset_index(drop=True), A.iloc[len(Xtr):len(Xtr) + len(Xho)].reset_index(drop=True)
fe_t = time.time() - t0
for X in (Xtr, Xho):
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
            X[c] = X[c].astype(str).astype('category')
for c in Xtr.columns:
    if isinstance(Xtr[c].dtype, pd.CategoricalDtype):
        Xho[c] = pd.Categorical(Xho[c].astype(str), categories=Xtr[c].cat.categories)
P = dict(objective='binary', learning_rate=0.03, num_leaves=63, min_child_samples=50, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, cat_smooth=20, num_threads=4, verbose=-1)
perm = np.random.default_rng(a.seed + 1).permutation(len(Xtr)); n = int(0.85 * len(Xtr))
fit, va = np.sort(perm[:n]), np.sort(perm[n:])
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 10000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros(len(Xho))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xho) / 3
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, auc=roc_auc_score(yho, p),
           best_it=b.best_iteration, n_cols=Xtr.shape[1], fe_s=round(fe_t), total_s=round(time.time() - t0))
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
