"""Data Science Bowl 2019 (Kaggle, $160k): event logs per child (installation) and labelled assessments.

Data: Hugging Face ``pytorch-lifestream/datascience-bowl2019`` (train.csv.gz, train_labels.csv.gz).
Main rows: the labelled assessments (installation, title, start time); target accuracy_group 0-3.
Child: the 11.3M game events (without the event_data JSON). Holdout: 20% of installations (the
contest's test set is new children), two seeds. Metric: quadratic weighted kappa of the regression
output cut at thresholds matching the training class shares (the usual contest decoding), and RMSE.

Arms: ``raw`` (assessment title, world, start time), ``asof`` (+ event-log aggregations as of each
assessment's start: only the child's earlier events), ``forge`` / ``asof_forge`` (+ FeatureForge).
"""
import sys, time, json, warnings, argparse, gc
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import cohen_kappa_score
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=1200); ap.add_argument('--kw', default='{}'); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='data/dsb/'); ap.add_argument('--log', default='dsb.jsonl')
a = ap.parse_args()
D = Path(a.data)
lab = pd.read_csv(D / 'train_labels.csv.gz')
cache = D / 'events.parquet'
if not cache.exists():
    cols = ['event_id', 'game_session', 'timestamp', 'installation_id', 'event_count', 'event_code', 'game_time', 'title',
            'event_type', 'world']
    ids = set(lab.installation_id)
    parts = []
    for ch in pd.read_csv(D / 'train.csv.gz', usecols=cols, chunksize=2_000_000):
        ch = ch[ch.installation_id.isin(ids)]
        ch['timestamp'] = pd.to_datetime(ch['timestamp']).astype('int64') / 1e9
        for c in ('event_id', 'game_session', 'installation_id', 'title', 'event_type', 'world'):
            ch[c] = ch[c].astype('category')
        parts.append(ch)
    ev = pd.concat(parts, ignore_index=True)
    for c in ('event_id', 'game_session', 'installation_id', 'title', 'event_type', 'world'):
        ev[c] = ev[c].astype(str).astype('category')
    ev['event_code'] = ev['event_code'].astype(str).astype('category')
    ev.to_parquet(cache); del parts; gc.collect()
ev = pd.read_parquet(cache)
st = ev.groupby('game_session', observed=True).agg(start=('timestamp', 'min'), world=('world', 'first'))
main = lab[['installation_id', 'game_session', 'title', 'accuracy_group']].merge(st, left_on='game_session', right_index=True)
main = main.sort_values('start').reset_index(drop=True)
y = main.pop('accuracy_group').to_numpy(dtype=float)
main['world'] = main['world'].astype(str)
inst = main['installation_id'].to_numpy()
u = np.unique(inst); rng = np.random.default_rng(a.seed); ho_inst = set(rng.choice(u, int(0.2 * len(u)), replace=False))
ho = np.array([i in ho_inst for i in inst]); tr = ~ho
X = main.drop(columns=['game_session'])
info = {}
t0 = time.time()
if a.arm.startswith('asof'):
    from tabularaml.generate.relational import Child, asof_features
    evc = ev.drop(columns=['game_session'])
    F = asof_features(X, 'installation_id', 'start', Child('ev', evc, key='installation_id', time='timestamp'))
    X = pd.concat([X, F], axis=1); info['n_asof'] = F.shape[1]
    del evc; gc.collect()
del ev; gc.collect()
Xtr, Xho, ytr, yho = X[tr].reset_index(drop=True), X[ho].reset_index(drop=True), y[tr], y[ho]
if a.arm.endswith('forge'):
    from tabularaml.generate.forge import FeatureForge
    f = FeatureForge(task='regression', time_budget=a.budget, random_state=a.seed, n_jobs=4, verbose=True,
                     **json.loads(a.kw)).fit(Xtr, ytr, X_unlabeled=Xho)
    info.update(n_added=len(f.new_columns_), gate=f.gate_passed_, group_col=f.group_col_, time_col=f.time_col_, feats=f.new_columns_[:30])
    Xtr, Xho = f.transform_train(Xtr), f.transform(Xho)
fe_t = time.time() - t0
g_tr = Xtr.pop('installation_id').astype(str).to_numpy(); Xho = Xho.drop(columns=['installation_id'])
for c in Xtr.columns:
    if Xtr[c].dtype == object or isinstance(Xtr[c].dtype, pd.CategoricalDtype):
        cats = pd.Index(pd.unique(Xtr[c].astype(str)))
        Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats); Xho[c] = pd.Categorical(Xho[c].astype(str), categories=cats)
P = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_child_samples=50, feature_fraction=0.6,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, num_threads=4, verbose=-1, seed=a.seed)
gu = np.unique(g_tr); va_g = set(np.random.default_rng(a.seed + 1).choice(gu, int(0.15 * len(gu)), replace=False))
va = np.array([g in va_g for g in g_tr])
b = lgb.train(P, lgb.Dataset(Xtr[~va], ytr[~va]), 10000, valid_sets=[lgb.Dataset(Xtr[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = lgb.train(P, lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xho)
share = np.cumsum(np.bincount(ytr.astype(int), minlength=4) / len(ytr))[:3]
cuts = np.quantile(p, share)
qwk = cohen_kappa_score(yho.astype(int), np.digitize(p, cuts), weights='quadratic')
res = dict(arm=a.arm + a.tag, seed=a.seed, qwk=float(qwk), rmse=float(np.sqrt(np.mean((p - yho) ** 2))), best_it=b.best_iteration,
           fe_s=round(fe_t), total_s=round(time.time() - t0), n_tr=len(Xtr), n_ho=len(Xho), n_cols=Xtr.shape[1], **info)
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
