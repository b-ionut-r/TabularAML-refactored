"""Elo Merchant Category Recommendation (Kaggle 2019, $50k): a loyalty score per card from its transactions.

Data: a public Kaggle copy of the contest files (``ershisuila/eio-recommend``): train.csv (201,917 cards: first
active month, three anonymised features, target), historical_transactions.csv (29M rows) and
new_merchant_transactions.csv (2M rows), both keyed by card_id with a purchase date.

Holdout built like the test file: ``--cards`` random cards (``--seed``), 20% of them held out whole (the test file
lists other cards, with their transactions given). Labels reach only the training cards. Metric: RMSE.
Arms (same bagged LightGBM: early stopping on 15% of the training cards, then 3 seeds on all of them):
  ``raw``  the card's own columns;
  ``ff``   blind: ``scripts/contest_features.py`` at its defaults with both transaction tables
           (``hist=...:card_id:purchase_date``, ``new=...:card_id:purchase_date``);
  ``ff`` with ``--extra "--history auto"`` adds the latest-state history features.
``--shuffle`` permutes the training labels (leakage control).
"""
import sys, time, json, warnings, argparse, subprocess, tempfile
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--cards', type=int, default=60_000); ap.add_argument('--tag', default='')
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default='')
ap.add_argument('--data', default='data/elo/'); ap.add_argument('--log', default='elo.jsonl'); ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
D = Path(a.data)
main = pd.read_csv(D / 'train.csv')
rng = np.random.default_rng(a.seed)
cards = rng.choice(main.card_id.to_numpy(), a.cards, replace=False)
ho = set(rng.choice(cards, a.cards // 5, replace=False))
main = main[main.card_id.isin(cards)].sort_values('card_id').reset_index(drop=True)
is_ho = main.card_id.isin(ho).to_numpy()
mtr, mte = main[~is_ho].reset_index(drop=True), main[is_ho].drop(columns=['target']).reset_index(drop=True)
ytr, yte = mtr.target.to_numpy(), main.target[is_ho].to_numpy()
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr); mtr['target'] = ytr
cache = D / f'sub_{a.cards}_{a.seed}'
if not (cache / 'hist.parquet').exists():
    cache.mkdir(exist_ok=True)
    keep = set(cards)
    for name, f in (('hist', 'historical_transactions.csv'), ('new', 'new_merchant_transactions.csv')):
        parts = [ch[ch.card_id.isin(keep)] for ch in pd.read_csv(D / f, chunksize=3_000_000)]
        pd.concat(parts, ignore_index=True).to_parquet(cache / f'{name}.parquet')
print('train cards', len(mtr), 'held out', len(mte), flush=True)
t0 = time.time()
if a.arm == 'raw':
    Xtr, Xte = mtr.drop(columns=['card_id', 'target']), mte.drop(columns=['card_id'])
else:
    tmp = Path(tempfile.mkdtemp(dir='/home/user/tmp'))
    mtr.to_parquet(tmp / 'train.parquet'); mte.to_parquet(tmp / 'test.parquet')
    cmd = [sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
           '--test', str(tmp / 'test.parquet'), '--target', 'target', '--id', 'card_id', '--task', 'regression',
           '--budget', str(a.budget), '--out-dir', str(tmp / 'out'),
           '--table', f'hist={cache / "hist.parquet"}:card_id:purchase_date',
           '--table', f'new={cache / "new.parquet"}:card_id:purchase_date'] + (a.extra.split() if a.extra else [])
    subprocess.run(cmd, check=True)
    Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet'); Xte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')
    assert (Xtr.card_id.to_numpy() == mtr.card_id.to_numpy()).all() and (Xte.card_id.to_numpy() == mte.card_id.to_numpy()).all()
    Xtr = Xtr.drop(columns=['card_id', 'target']); Xte = Xte.drop(columns=['card_id'])[Xtr.columns]
fe_t = time.time() - t0
for X in (Xtr, Xte):
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
            X[c] = X[c].astype('category')
for c in Xtr.columns:
    if isinstance(Xtr[c].dtype, pd.CategoricalDtype):
        Xte[c] = pd.Categorical(Xte[c].astype(str), categories=Xtr[c].cat.categories)
P = dict(objective='regression', learning_rate=0.02, num_leaves=31, min_child_samples=50, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1)
perm = np.random.default_rng(a.seed + 1).permutation(len(Xtr)); n = int(0.85 * len(Xtr))
fit, va = np.sort(perm[:n]), np.sort(perm[n:])
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 5000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, rmse=float(np.sqrt(np.mean((p - yte) ** 2))),
           rmse_const=float(np.sqrt(np.mean((np.mean(ytr) - yte) ** 2))), best_it=b.best_iteration, n_cols=Xtr.shape[1],
           fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), n_te=len(Xte))
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
