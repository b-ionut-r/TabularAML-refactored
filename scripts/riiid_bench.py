"""Riiid Answer Correctness Prediction (Kaggle 2021, $100k): will a student answer the next question correctly.

Data: a public Kaggle copy of the contest's train file (``rohanrao/riiid-train-data-multiple-formats``,
riiid_train.parquet: 101M events of 393,656 students; questions and lectures, timestamps in ms since each
student's first event).

Holdout built like the test file (the test continues known students' logs and adds new students): ``--users``
random students (``--seed``); 10% of them held out whole (new students), and for the rest the last ``--tail`` of
their events (continuing students). Main rows are question events; every training event (lectures included)
forms the event log they are aggregated over, as of each row's timestamp. The held-out events are not in that
log, so their answers reach only the scorer (stricter than the contest, which revealed earlier test answers).
Metric: AUC. Arms (same bagged LightGBM: early stopping on 15% of the training students, then 3 seeds):
  ``raw``  the event's own columns (question id, bundle, timestamp, prior elapsed time and explanation);
  ``ff``   blind: ``scripts/contest_features.py`` at its defaults with the log as a table
           (``log=...:user_id:timestamp``), so its as-of aggregation switches on.
``--reveal`` puts the held-out events in the log too, as the contest's API did: each row then sees answers from
strictly earlier timestamps only (its own bundle's answers stay hidden). ``--shuffle`` permutes the training labels
and the log's answers (leakage control).
"""
import sys, time, json, warnings, argparse, subprocess, tempfile
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--users', type=int, default=5000); ap.add_argument('--tail', type=float, default=0.1); ap.add_argument('--tag', default='')
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default='')
ap.add_argument('--data', default='data/riiid/'); ap.add_argument('--log', default='riiid.jsonl'); ap.add_argument('--shuffle', action='store_true')
ap.add_argument('--reveal', action='store_true', help='the log also holds held-out events, as the contest revealed earlier answers')
a = ap.parse_args()
D = Path(a.data)
cache = D / f'sub_{a.users}_{a.seed}.parquet'
if not cache.exists():
    u = pd.read_parquet(D / 'riiid_train.parquet', columns=['user_id'])['user_id'].unique()
    pick = np.random.default_rng(a.seed).choice(u, a.users, replace=False)
    E = pd.read_parquet(D / 'riiid_train.parquet', filters=[('user_id', 'in', list(pick))])
    E.to_parquet(cache)
E = pd.read_parquet(cache).sort_values(['user_id', 'timestamp', 'row_id']).reset_index(drop=True)
E = E.drop(columns=['row_id', 'user_answer'])
E['prior_question_had_explanation'] = E['prior_question_had_explanation'].astype(float)
rng = np.random.default_rng(a.seed + 7)
users = E.user_id.unique()
new = set(rng.choice(users, len(users) // 10, replace=False))
rank = E.groupby('user_id').cumcount(ascending=False) / E.groupby('user_id').user_id.transform('size')
ho = E.user_id.isin(new).to_numpy() | (rank < a.tail).to_numpy()
log = (E if a.reveal else E[~ho]).reset_index(drop=True)   # events whose answers are known before later rows
q = E.content_type_id.to_numpy() == 0
tr, te = E[~ho & q].reset_index(drop=True), E[ho & q].reset_index(drop=True)
ytr, yte = tr.pop('answered_correctly').to_numpy(), te.pop('answered_correctly').to_numpy()
tr, te = tr.drop(columns=['content_type_id']), te.drop(columns=['content_type_id'])
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
    log.loc[log.content_type_id == 0, 'answered_correctly'] = np.random.default_rng(1).permutation(
        log.loc[log.content_type_id == 0, 'answered_correctly'].to_numpy())
print('log', len(log), 'train questions', len(tr), 'held out', len(te), flush=True)
t0 = time.time()
if a.arm == 'raw':
    Xtr, Xte = tr.drop(columns=['user_id']), te.drop(columns=['user_id'])
else:
    tmp = Path(tempfile.mkdtemp(dir='/home/user/tmp'))
    tr.assign(answered_correctly=ytr).to_parquet(tmp / 'train.parquet'); te.to_parquet(tmp / 'test.parquet')
    log.to_parquet(tmp / 'log.parquet')
    cmd = [sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
           '--test', str(tmp / 'test.parquet'), '--target', 'answered_correctly', '--task', 'binary',
           '--budget', str(a.budget), '--out-dir', str(tmp / 'out'),
           '--table', f'log={tmp / "log.parquet"}:user_id:timestamp'] + (a.extra.split() if a.extra else [])
    subprocess.run(cmd, check=True)
    Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['answered_correctly'])
    Xte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Xtr.columns]
    Xtr, Xte = Xtr.drop(columns=['user_id'], errors='ignore'), Xte.drop(columns=['user_id'], errors='ignore')
fe_t = time.time() - t0
for X in (Xtr, Xte):
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
            X[c] = X[c].astype('category')
P = dict(objective='binary', learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1)
uid = tr.user_id.to_numpy(); uu = np.unique(uid)
va_u = set(np.random.default_rng(a.seed + 1).choice(uu, int(0.15 * len(uu)), replace=False))
va = np.isin(uid, list(va_u))
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr[~va], ytr[~va]), 5000, valid_sets=[lgb.Dataset(Xtr[va], ytr[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
p = np.zeros(len(Xte))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
isnew = te.user_id.isin(new).to_numpy()
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, auc=float(roc_auc_score(yte, p)),
           auc_continuing=float(roc_auc_score(yte[~isnew], p[~isnew])), auc_new=float(roc_auc_score(yte[isnew], p[isnew])),
           best_it=b.best_iteration, n_cols=Xtr.shape[1], fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), n_te=len(Xte))
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')

import shutil
if 'tmp' in globals() and not getattr(a, 'keep', ''):
    shutil.rmtree(tmp, ignore_errors=True)   # bench outputs fill the disk otherwise
