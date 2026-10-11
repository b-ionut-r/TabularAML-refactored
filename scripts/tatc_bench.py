"""Optiver - Trading at the Close (Kaggle 2023, $100k): the next 60 seconds' move of a Nasdaq stock's weighted price
against a synthetic index of the ~200 stocks, during the 10-minute closing auction (one row per stock, day and
10-second step). MAE (target in basis points).

Data: a public Kaggle copy of the contest's train.csv (``nihalk17/optiver-trading-at-the-close``: 5,237,980 rows,
200 stocks, days 0-480). The contest's test was the days after the training file, so a window holds out the last
45 days before a cut (``--win`` 0: days 436-480, 1: days 391-435) and trains on the ``--train-days`` days before
them; held-out targets reach only the scorer. Columns typed as Kaggle's CSV reads (row_id text).
Arms (same bagged LightGBM: early stopping on the last 15% of training days, then 3 seeds, L1 objective):
  ``raw``   the file's columns;
  ``ff``    blind: ``scripts/contest_features.py`` at its defaults;
  ``hand``  raw + the features of top public kernels and winners' write-ups that need no outside data: size and price
            imbalances (bid / ask, imbalance / matched, pairwise price differences over sums, three-price imbalance),
            spreads and mid price, the stock's move since the previous steps (wap, prices and sizes 1-3 steps back on
            the same day), and same-moment cross-sectional features: the stock against all stocks at the same day
            and second (deviation from the equal-weight mean of wap and its 1-step return, ranks of imbalance and
            spread), plus the stock's median sizes over the training days;
  ``ff_hand`` both.
``--shuffle`` permutes the training targets (leakage control). Reports runtime and peak memory.
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil, resource
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--win', type=int, default=0)
ap.add_argument('--train-days', type=int, default=120); ap.add_argument('--tag', default='')
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default='')
ap.add_argument('--data', default='/home/user/data/tatc/'); ap.add_argument('--log', default='tatc.jsonl')
ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--groups', default='')
ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1])); ap.add_argument('--ff-cache', action='store_true')
a = ap.parse_args()
D = Path(a.data)
df = pd.read_csv(D / 'train.csv')
df = df[df.target.notna()].reset_index(drop=True)
end = 480 - 45 * a.win
ho0 = end - 44
tr = df[(df.date_id >= ho0 - a.train_days) & (df.date_id < ho0)].reset_index(drop=True)
te = df[(df.date_id >= ho0) & (df.date_id <= end)].reset_index(drop=True)
del df
ytr, yte = tr.pop('target').to_numpy(), te.pop('target').to_numpy()
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
print(f'train days {ho0 - a.train_days}-{ho0 - 1} ({len(tr)} rows), held out {ho0}-{end} ({len(te)} rows)', flush=True)


def raw(X):
    return X.drop(columns=['row_id']).copy()


def hand(tr, te):
    A = pd.concat([tr, te], ignore_index=True)
    n = len(tr)
    H = {}
    P = ['reference_price', 'far_price', 'near_price', 'bid_price', 'ask_price', 'wap']
    H['size_imb'] = (A.bid_size - A.ask_size) / (A.bid_size + A.ask_size)
    H['imb_matched'] = A.imbalance_size / A.matched_size
    H['signed_imb'] = A.imbalance_size * A.imbalance_buy_sell_flag
    H['signed_imb_matched'] = H['signed_imb'] / A.matched_size
    H['spread'] = A.ask_price - A.bid_price
    H['mid'] = (A.ask_price + A.bid_price) / 2
    H['liquidity_imb'] = (A.bid_size - A.ask_size) / (A.bid_size + A.ask_size)
    H['size_total'] = A.bid_size + A.ask_size + A.matched_size + A.imbalance_size
    for i, p in enumerate(P):
        for q in P[i + 1:]:
            H[f'{p}_{q}_imb'] = (A[p] - A[q]) / (A[p] + A[q])
    for t in (('ask_price', 'bid_price', 'wap'), ('reference_price', 'bid_price', 'ask_price'), ('matched_size', 'bid_size', 'ask_size')):
        v = A[list(t)].to_numpy(dtype=float)
        mx, mn = v.max(1), v.min(1); md = v.sum(1) - mx - mn
        with np.errstate(all='ignore'):
            H['triplet_' + '_'.join(t)] = np.where(md - mn > 0, (mx - md) / (md - mn), np.nan)
    H = pd.DataFrame(H)
    # The stock's own recent steps on the same day.
    g = A.groupby(['stock_id', 'date_id'], sort=False)
    for c in ['wap', 'reference_price', 'bid_price', 'ask_price', 'matched_size', 'imbalance_size']:
        for k in (1, 2, 3):
            H[f'{c}_ret{k}'] = A[c] / g[c].shift(k) - 1
    H['imb_flag_change'] = A.imbalance_buy_sell_flag - g.imbalance_buy_sell_flag.shift(1)
    # Same moment, all stocks.
    m = A.groupby(['date_id', 'seconds_in_bucket'], sort=False)
    H['wap_vs_mean'] = A.wap - m.wap.transform('mean')
    H['wap_ret1_vs_mean'] = H['wap_ret1'] - H['wap_ret1'].groupby([A.date_id, A.seconds_in_bucket]).transform('mean')
    for c in ['size_imb', 'imb_matched', 'spread', 'signed_imb_matched']:
        H[f'{c}_rank'] = H[c].groupby([A.date_id, A.seconds_in_bucket]).rank(pct=True)
    # The stock's typical sizes over the training days (label-free).
    for c in ['bid_size', 'ask_size', 'matched_size']:
        med = tr.groupby('stock_id')[c].median()
        H[f'{c}_stock_median'] = A.stock_id.map(med)
    H = H.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    return H.iloc[:n].reset_index(drop=True), H.iloc[n:].reset_index(drop=True)


t0 = time.time(); info = {}
Xtr, Xte = raw(tr), raw(te)
cache = D / f'ff_cache_{a.win}{a.tag}{"_shuf" if a.shuffle else ""}'
if a.arm.startswith('ff') and a.ff_cache and (cache / 'tr.parquet').exists():
    Ftr, Fte = pd.read_parquet(cache / 'tr.parquet'), pd.read_parquet(cache / 'te.parquet')
    Xtr, Xte = pd.concat([Xtr, Ftr], axis=1), pd.concat([Xte, Fte], axis=1); info['n_new'] = Ftr.shape[1]
elif a.arm.startswith('ff'):
    tmp = Path(tempfile.mkdtemp(dir='/home/user/tmp'))
    tr.assign(target=ytr).to_csv(tmp / 'train.csv', index=False); te.to_csv(tmp / 'test.csv', index=False)
    cmd = [sys.executable, str(Path(a.repo) / 'scripts' / 'contest_features.py'), '--train', str(tmp / 'train.csv'),
           '--test', str(tmp / 'test.csv'), '--target', 'target', '--id', 'row_id', '--task', 'regression',
           '--budget', str(a.budget), '--out-dir', str(tmp / 'out')] + (a.extra.split() if a.extra else [])
    subprocess.run(cmd, check=True, cwd=a.repo)
    info['pipe_peak_gb'] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2 ** 20, 2)
    Ftr = pd.read_parquet(tmp / 'out' / 'train_features.parquet'); Fte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')
    assert (Ftr['row_id'].astype(str).to_numpy() == tr.row_id.to_numpy()).all() and (Fte['row_id'].astype(str).to_numpy() == te.row_id.to_numpy()).all()
    new = [c for c in Ftr.columns if c not in tr.columns and c != 'target']
    Ftr, Fte = Ftr[new].reset_index(drop=True), Fte[new].reset_index(drop=True)
    Xtr, Xte = pd.concat([Xtr, Ftr], axis=1), pd.concat([Xte, Fte], axis=1)
    info['n_new'] = len(new)
    cache.mkdir(exist_ok=True); Ftr.to_parquet(cache / 'tr.parquet'); Fte.to_parquet(cache / 'te.parquet')
    shutil.rmtree(tmp, ignore_errors=True)
if a.arm.endswith('hand'):
    Htr, Hte = hand(tr, te)
    Xtr, Xte = pd.concat([Xtr, Htr], axis=1), pd.concat([Xte, Hte], axis=1)
fe_t = time.time() - t0
for c in Xtr.columns:
    if not pd.api.types.is_numeric_dtype(Xtr[c]):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)
P = dict(objective='l1', learning_rate=0.05, num_leaves=127, min_child_samples=200, feature_fraction=0.6,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1)
days = tr.date_id.to_numpy(); cut = np.quantile(days, 0.85)
fit, va = np.flatnonzero(days < cut), np.flatnonzero(days >= cut)
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 3000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), win=a.win, mae=float(np.mean(np.abs(p - yte))),
           mae_const=float(np.mean(np.abs(np.median(ytr) - yte))), best_it=b.best_iteration, n_cols=Xtr.shape[1],
           fe_s=round(fe_t), total_s=round(time.time() - t0), peak_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2),
           n_tr=len(Xtr), n_te=len(Xte), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
