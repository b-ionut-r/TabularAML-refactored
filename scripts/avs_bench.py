"""Acquire Valued Shoppers Challenge (Kaggle 2014, $30k): which customers given an offer become repeat buyers.

Data: Hugging Face ``pytorch-lifestream/acquire-valued-shoppers`` (trainHistory, offers, transactions).
The 350M-row transaction file is cut, as contest kernels did, to rows whose category, company or brand
is that of some offer, then to the training customers (16.4M rows, every one dated before the
customer's offer, as in the contest). Holdout built like the contest's test file, which held later
offers: ``--win 0`` holds out the offers dated from 2013-04-22 (training on earlier offers), ``--win 1``
those dated 2013-04-01 to 04-04 (training on offers before April). Metric: AUC over all held-out
customers, as the contest.

Arms: ``raw`` (history row with the offer's category, company, brand, value and quantity joined),
``pipe`` (trainHistory as the main table, offers and transactions given to ``scripts/contest_features.py``
at its defaults). ``--shuffle`` permutes the training labels (leakage control).
"""
import sys, time, json, warnings, argparse, subprocess, tempfile
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--win', type=int, default=0)
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='data/avs/'); ap.add_argument('--log', default='avs.jsonl')
ap.add_argument('--shuffle', action='store_true'); ap.add_argument('--extra', default='', help='extra contest_features.py flags')
a = ap.parse_args()
D = Path(a.data)
h = pd.read_csv(D / 'trainHistory.csv.gz').drop(columns=['repeattrips'])
y_all = (h.pop('repeater') == 't').to_numpy().astype(int)
lo, hi = [('2013-04-22', '2013-12-31'), ('2013-04-01', '2013-04-04')][a.win]
ho = ((h.offerdate >= lo) & (h.offerdate <= hi)).to_numpy()
tr = (h.offerdate < lo).to_numpy()
htr, hho, ytr, yho = h[tr].reset_index(drop=True), h[ho].reset_index(drop=True), y_all[tr], y_all[ho]
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
t0 = time.time(); info = {}
offers = pd.read_csv(D / 'offers.csv.gz')
if a.arm == 'raw':
    Xtr, Xho = htr.merge(offers, on='offer', how='left'), hho.merge(offers, on='offer', how='left')
else:
    tmp = Path(tempfile.mkdtemp())
    htr.assign(repeater=ytr).to_csv(tmp / 'train.csv', index=False); hho.to_csv(tmp / 'test.csv', index=False)
    offers.to_csv(tmp / 'offers.csv', index=False)
    T = pd.read_parquet(D / 'trans_train.parquet')
    T = T[T.id.isin(set(htr.id) | set(hho.id))]
    T['date'] = T['date'].dt.strftime('%Y-%m-%d')
    T.to_parquet(tmp / 'transactions.parquet'); del T
    cmd = [sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.csv'),
           '--test', str(tmp / 'test.csv'), '--target', 'repeater', '--id', 'id', '--task', 'binary', '--budget', str(a.budget),
           '--out-dir', str(tmp / 'out'), '--table', f'offers={tmp / "offers.csv"}',
           '--table', f'transactions={tmp / "transactions.parquet"}'] + a.extra.split()
    subprocess.run(cmd, check=True)
    Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['repeater'])
    Xho = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Xtr.columns]
    info = dict(n_cols=Xtr.shape[1])
fe_t = time.time() - t0
Xtr, Xho = Xtr.drop(columns=['id']), Xho.drop(columns=['id'])
for X in (Xtr, Xho):
    X['offerdate'] = (pd.to_datetime(X['offerdate'].astype(str)) - pd.Timestamp('2013-01-01')).dt.days.astype(float)
for c in Xtr.columns:
    if not pd.api.types.is_numeric_dtype(Xtr[c]):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xho[c] = pd.Categorical(Xho[c].astype(str), categories=cats)
P = dict(objective='binary', learning_rate=0.03, num_leaves=31, min_child_samples=100, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1, seed=0)
# Early stopping on the latest training offers (the holdout's offers are later and new).
d = Xtr['offerdate'].to_numpy(); va = d >= np.quantile(d, 0.8)
b = lgb.train(P, lgb.Dataset(Xtr[~va], ytr[~va]), 5000, valid_sets=[lgb.Dataset(Xtr[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xho)
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), win=a.win, auc=float(roc_auc_score(yho, p)), best_it=b.best_iteration,
           fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), n_ho=len(Xho), **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
