"""Regression-bench case "built like the real test file" (the Kaggle thread's protocol, kaggle_late.py):
full train, the whole Kaggle test file as unlabeled rows, scripts/contest_features.py of --repo at shipped
defaults, then kaggle_late's judge (local CV). Prints one RESULT line: cv, the FeatureForge gate (gain, PASS or
REJECT), the number of columns added, and the build and judge seconds.

    python rt_case.py --contest wnv --arm ff --repo /home/user/wt/2fbcbc2 --work /home/user/tmp/rt_wnv
    python rt_case.py --contest hc --arm raw
"""
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--contest', required=True); ap.add_argument('--arm', default='ff'); ap.add_argument('--repo', default='')
ap.add_argument('--work', default='/home/user/tmp/rt'); ap.add_argument('--budget', default='900')
ap.add_argument('--keep', action='store_true', help='keep the feature files')
a = ap.parse_args()
KL = str(Path(__file__).resolve().parent / 'kaggle_late.py')
env = dict(os.environ)
t0 = time.time(); info = {}
if a.arm != 'raw':
    work = Path(a.work); shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    env.update(KG_REPO=a.repo, KG_FEATS=str(work))
    r = subprocess.run([sys.executable, KL, 'feats', '--contest', a.contest, '--budget', a.budget], env=env,
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    (work.parent / f'{work.name}_feats.log').write_text(out)
    if r.returncode:
        print(out[-4000:]); sys.exit(r.returncode)
    g = re.findall(r'gate: raw=\S+ fe=\S+ \(([+-]?\d+\.\d+)%\) -> (PASS|REJECT)', out)
    info.update(gate_gain_pct=float(g[-1][0]) if g else None, gate=g[-1][1] if g else None)
    import pandas as pd
    raw_cols = pd.read_parquet(f'/tmp/claude-0/kg/{a.contest}/prep/train.parquet').columns
    info['n_new'] = sum(c not in raw_cols for c in pd.read_parquet(work / 'train_features.parquet').columns)
fe_t = time.time() - t0
r = subprocess.run([sys.executable, KL, 'fit', '--contest', a.contest, '--arm', 'raw' if a.arm == 'raw' else 'forge'],
                   env=env, capture_output=True, text=True)
line = [l for l in r.stdout.splitlines() if l.startswith('RESULT ')]
if r.returncode or not line:
    print(r.stdout[-3000:], r.stderr[-3000:]); sys.exit(r.returncode or 1)
res = json.loads(line[-1][7:])
if a.arm != 'raw' and not a.keep:
    shutil.rmtree(a.work, ignore_errors=True)
print('RESULT', json.dumps(dict(contest=a.contest, arm=a.arm, cv=res['cv'], n_feat=res['n_feat'], fe_s=round(fe_t),
                                judge_s=res['judge_s'], total_s=round(time.time() - t0), **info)))
