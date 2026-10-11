"""Forest Cover Type Prediction (Kaggle 2015): raw vs FeatureForge vs the public hand features.

Data: OpenML 1596 (the full 581,012-cell UCI table the contest was cut from: 10 terrain measurements, 4 wilderness and
40 soil-type flags; 7 cover types). The contest trained on 15,120 cells, 2,160 per cover type, and tested on the
rest; so the training rows are 2,160 random cells per type (``--seed``) and the holdout is 100,000 random cells of
the rest, at their natural mix. The holdout's features are the unlabeled rows. Columns are typed as the contest's
CSV: integers. Metric: accuracy, as the contest (multiclass log loss reported too).
Arms (same bagged LightGBM: early stopping on 15% of the training rows, then 3 seeds on all of them):
  ``raw``    the columns as given;
  ``ff``     blind: ``scripts/contest_features.py`` at its defaults;
  ``hand``   the top public kernels' features: distance to water (sqrt of horizontal^2 + vertical^2), elevation
             minus vertical and minus 0.2 x horizontal distance to water, sums and absolute differences of the
             three horizontal distances (water, fire points, roads), mean hillshade, aspect sine and cosine, and
             the soil type and wilderness area as one code each.
``--shuffle`` permutes the training labels (leakage control).

    python scripts/cover_bench.py --arm ff --seed 0
"""
import sys, time, json, warnings, argparse, subprocess, tempfile, shutil
from pathlib import Path
warnings.filterwarnings('ignore'); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import log_loss, accuracy_score
ap = argparse.ArgumentParser(); ap.add_argument('--arm', default='raw'); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--budget', type=float, default=900); ap.add_argument('--extra', default=''); ap.add_argument('--tag', default='')
ap.add_argument('--data', default='/tmp/claude-0/data/cover/cover.pq'); ap.add_argument('--log', default='cover.jsonl')
ap.add_argument('--shuffle', action='store_true')
a = ap.parse_args()
df = pd.read_parquet(a.data)
y = df.pop('class').astype(int).to_numpy() - 1
for c in df.columns:
    df[c] = pd.to_numeric(df[c].astype(str) if isinstance(df[c].dtype, pd.CategoricalDtype) else df[c]).astype(np.int64)
rng = np.random.default_rng(a.seed)
itr = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == k), 2160, replace=False) for k in range(7)]))
rest = np.setdiff1d(np.arange(len(df)), itr); ite = np.sort(rng.choice(rest, 100_000, replace=False))
Xtr, Xte = df.iloc[itr].reset_index(drop=True), df.iloc[ite].reset_index(drop=True)
ytr, yte = y[itr], y[ite]
if a.shuffle:
    ytr = np.random.default_rng(0).permutation(ytr)
t0 = time.time()
def hand(X):
    X = X.copy(); H, V = X['Horizontal_Distance_To_Hydrology'], X['Vertical_Distance_To_Hydrology']
    F, R = X['Horizontal_Distance_To_Fire_Points'], X['Horizontal_Distance_To_Roadways']
    X['dist_water'] = np.sqrt(H ** 2 + V ** 2); X['elev_m_vdh'] = X['Elevation'] - V; X['elev_m_hdh'] = X['Elevation'] - 0.2 * H
    for n, (p, q) in {'hf': (H, F), 'hr': (H, R), 'fr': (F, R)}.items():
        X[n + '_sum'] = p + q; X[n + '_absdiff'] = (p - q).abs()
    X['hill_mean'] = X[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].mean(1)
    X['aspect_sin'] = np.sin(np.radians(X['Aspect'])); X['aspect_cos'] = np.cos(np.radians(X['Aspect']))
    soil = [c for c in X.columns if c.startswith('Soil_Type')]; wild = [c for c in X.columns if c.startswith('Wilderness')]
    X['soil'] = X[soil].to_numpy().argmax(1); X['wild'] = X[wild].to_numpy().argmax(1)
    return X
if a.arm == 'ff':
    tmp = Path(tempfile.mkdtemp(dir='/tmp/claude-0'))
    Xtr.assign(cover=ytr).to_parquet(tmp / 'train.parquet'); Xte.to_parquet(tmp / 'test.parquet')
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
                    '--test', str(tmp / 'test.parquet'), '--target', 'cover', '--task', 'multiclass', '--budget', str(a.budget),
                    '--out-dir', str(tmp / 'out')] + (a.extra.split() if a.extra else []), check=True)
    Xtr = pd.read_parquet(tmp / 'out' / 'train_features.parquet').drop(columns=['cover'])
    Xte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')[Xtr.columns]
    shutil.rmtree(tmp, ignore_errors=True)
elif a.arm == 'hand':
    Xtr, Xte = hand(Xtr), hand(Xte)
fe_t = time.time() - t0
for X in (Xtr, Xte):
    for c in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)):
            X[c] = X[c].astype(str).astype('category')
for c in Xtr.columns:
    if isinstance(Xtr[c].dtype, pd.CategoricalDtype):
        Xte[c] = pd.Categorical(Xte[c].astype(str), categories=Xtr[c].cat.categories)
P = dict(objective='multiclass', num_class=7, learning_rate=0.03, num_leaves=31, min_child_samples=10, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=4, verbose=-1)
perm = np.random.default_rng(a.seed + 1).permutation(len(Xtr)); n = int(0.85 * len(Xtr))
fit, va = np.sort(perm[:n]), np.sort(perm[n:])
b = lgb.train(dict(P, seed=0), lgb.Dataset(Xtr.iloc[fit], ytr[fit]), 10000, valid_sets=[lgb.Dataset(Xtr.iloc[va], ytr[va])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
p = np.zeros((len(Xte), 7))
for s in range(3):
    p += lgb.train(dict(P, seed=s), lgb.Dataset(Xtr, ytr), int(b.best_iteration * 1.1) + 1).predict(Xte) / 3
res = dict(arm=a.arm + a.tag + ('_shuffled' if a.shuffle else ''), seed=a.seed, acc=accuracy_score(yte, p.argmax(1)),
           mlogloss=log_loss(yte, p, labels=list(range(7))), best_it=b.best_iteration, n_cols=Xtr.shape[1],
           fe_s=round(fe_t), total_s=round(time.time() - t0))
print('RESULT', json.dumps(res, default=str)); open(a.log, 'a').write(json.dumps(res, default=str) + '\n')
