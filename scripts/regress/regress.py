"""Regression bench: rerun every contest used so far on its fixed held-out split, raw vs FeatureForge at a commit.

    python scripts/regress/regress.py run --commit 2fbcbc2 --tier fast [--cases ieee_l20,avazu] [--shuffled]
    python scripts/regress/regress.py compare --ref 2c26dfc --new 2fbcbc2 --tier fast
    python scripts/regress/regress.py list

Each case is one bench script (the protocol the thread that used the contest ran, with its seeds and splits)
and a list of samples (seeds or held-out windows). ``run`` checks the commit out in a git worktree, copies the
bench scripts of this branch into it (so every commit is judged by the same benches; FeatureForge and
``scripts/contest_features.py`` are the commit's own), and runs each case's samples one at a time, recording
the score, build and total seconds and the peak memory of the whole process tree. Raw arms do not depend on
the commit and are run once. Results: RESULTS/<commit>/<case>/<sample>_<arm>.json (a rerun skips what exists).

``compare`` prints one table: per case, raw, reference and new FeatureForge per sample, the paired change,
the noise band, runtime and peak memory, and a flag. A case is WORSE when the new commit is worse than the
reference on every sample and the mean paired change is beyond the case's noise band; BETTER the same way
round. The noise band is the larger of the case's known rerun noise (``noise``) and half the spread of the
paired changes across samples.
"""
import argparse, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = Path(os.environ.get('REGRESS_RESULTS', '/mnt/project-files/generalization/results'))
WT = Path(os.environ.get('REGRESS_WT', '/home/user/wt'))
DATA = Path(os.environ.get('REGRESS_DATA', '/home/user/data'))
KG = Path('/tmp/claude-0/kg')

# Every case: script (run from the worktree's scripts/ dir), args, arm args, metric key in the RESULT json,
# whether higher is better, samples (name -> extra args), known rerun noise, tier, and the data it needs.
# {repo} is the worktree under test, {data} the data root, {tmp} a scratch dir for this job.
CASES = {
    # IEEE-CIS (fraud, 2019): latest 40% of transactions, last 20% of that window held out (time order),
    # held-out features as unlabeled rows; feature thread's ieee_r.py. Second window ends at 80% of the file.
    'ieee_l20': dict(
        script='ieee_r.py', args=['--frac', '0.4', '--transductive', '--budget', '900'],
        env={'REPO': '{repo}', 'IEEE_DATA': '{data}/ieee/ieee.parquet'},
        arms={'raw': ['--arm', 'raw'], 'ff': ['--arm', 'forge']}, shuffle=['--shuffle'],
        metric='auc', higher=True, noise=0.002, tier='fast', needs=['{data}/ieee/ieee.parquet'],
        samples={'end100': ['--end', '1.0'], 'end80': ['--end', '0.8']}),
    # Avazu (click log): 5% of users with whole histories, last day held out; contest_features blind.
    'avazu': dict(
        script='blind_bench.py', args=['--repo', '{repo}', '--target', 'click', '--name', 'avazu_u', '--time', 'hour',
                                       '--log', '{tmp}/blind.jsonl'],
        arms={'raw': ['--arm', 'raw'], 'ff': ['--arm', 'ff']}, shuffle=['--shuffle'],
        metric='auc', higher=True, noise=0.0015, tier='fast', needs=['{data}/avazu/avazu_u5.parquet'],
        samples={'day10': ['--data', '{data}/avazu/avazu_u5.parquet', '--cut', '14103000'],
                 'day9': ['--data', '{data}/avazu/avazu_u5_d9.parquet', '--cut', '14102900']}),
    # West Nile (2015): built like the Kaggle entry (full train, the 116k-row test file as unlabeled rows),
    # judged by GroupKFold over years; plus the held-out-year bench (FeatureForge seed 0).
    'wnv_kaggle': dict(
        script='rt_case.py', args=['--contest', 'wnv', '--repo', '{repo}', '--work', '{tmp}/rt'],
        arms={'raw': ['--arm', 'raw'], 'ff': ['--arm', 'ff']}, metric='cv', higher=True, noise=0.004, tier='fast',
        needs=[str(KG / 'wnv/prep/train.parquet')], samples={'cv': []}),
    'wnv_year': dict(
        script='wnv_seed_bench.py', args=['--data', '{data}/wnv/', '--log', '{tmp}/wnv.jsonl'],
        arms={'raw': ['--arm', 'raw'], 'ff': ['--arm', 'forge']}, shuffle=['--shuffle'],
        metric='auc', higher=True, noise=0.01, tier='fast', needs=['{data}/wnv/train.csv'],
        samples={'y2013': ['--year', '2013'], 'y2011': ['--year', '2011']}),
    # Home Credit (2018): built like the Kaggle entry (5 child tables, child models, test as unlabeled rows),
    # 5-fold CV. The gate passes its search features by +0.18%: ~0.002 is noise (Kaggle thread).
    'hc_kaggle': dict(
        script='rt_case.py', args=['--contest', 'hc', '--repo', '{repo}', '--work', '{tmp}/rt'],
        arms={'raw': ['--arm', 'raw'], 'ff': ['--arm', 'ff']}, metric='cv', higher=True, noise=0.002, tier='fast',
        needs=[str(KG / 'hc/prep/train.parquet')], samples={'cv': []}),
    # Santander Transaction (2019): stratified 80/20; unlabeled = holdout + real test rows. Reruns 0.9172-0.9185.
    'santander': dict(
        script='sct.py', args=['--budget', '1200'], env={'REPO': '{repo}', 'SCT_DATA': str(KG / 'sct/prep') + '/'},
        arms={'raw': ['--arm', 'raw'], 'ff': ['--arm', 'forge']}, metric='auc', higher=True, noise=0.0015,
        tier='fast', needs=[str(KG / 'sct/prep/train.parquet')], samples={'s0': ['--seed', '0'], 's1': ['--seed', '1']}),
}


def _seeds(*xs):
    return {f's{x}': ['--seed', str(x)] for x in xs}


def _wins(*xs):
    return {f'w{x}': ['--win', str(x)] for x in xs}


def _bench(script, data, metric, higher, noise, samples, ff='ff', raw='raw', extra=(), tier='full', needs=None):
    """A repo bench with the usual flags (--arm, --data, --log, --shuffle)."""
    return dict(script=script, args=['--data', data, '--log', '{tmp}/log.jsonl', *extra],
                arms={'raw': ['--arm', raw], 'ff': ['--arm', ff]}, shuffle=['--shuffle'], metric=metric,
                higher=higher, noise=noise, tier=tier, needs=[needs or data], samples=samples)


# Full tier: every other contest used so far, on the bench and samples its thread reported. The FeatureForge
# arm is the bench's blind arm (scripts/contest_features.py at shipped defaults where the bench has one).
CASES.update({
    'amazon': _bench('amazon_bench.py', '{data}/amazon/amazon.pq', 'auc', True, 0.002, _seeds(0, 1)),
    'telstra': _bench('telstra_bench.py', '{data}/telstra/', 'mlogloss', False, 0.006, _seeds(0, 1),
                      needs='{data}/telstra/train.csv'),
    'bnp': _bench('bnp_bench.py', '{data}/bnp/train.csv', 'logloss', False, 0.002, _seeds(0, 1)),
    'cover': _bench('cover_bench.py', '{data}/cover/cover.pq', 'acc', True, 0.004, _seeds(0, 1)),
    'homesite': _bench('homesite_bench.py', '{data}/homesite/', 'auc', True, 0.0006, _seeds(0, 1),
                       needs='{data}/homesite/train.csv'),
    'kdd12': _bench('kdd12_bench.py', '{data}/kdd12/kdd12.pq', 'auc', True, 0.0015, _seeds(0, 1)),
    'liberty': _bench('liberty_bench.py', '{data}/liberty/', 'gini', True, 0.004, _seeds(0, 1),
                      needs='{data}/liberty/train.csv'),
    'scs': _bench('scs_bench.py', '{data}/scs/scs.pq', 'auc', True, 0.002, _seeds(0, 1)),
    'porto': dict(_bench('porto_bench.py', '{data}/porto/porto.pq', 'gini', True, 0.004, _seeds(0, 1), ff='forge'),
                  shuffle=None),
    'allstate': _bench('allstate_bench.py', '{data}/allstate/as.pq', 'mae', False, 3.0, _seeds(0, 1), ff='forge'),
    'loandefault': _bench('loandefault_bench.py', '{data}/loandefault/ld.pq', 'auc', True, 0.005, _seeds(0, 1), ff='forge'),
    'twosigma': _bench('twosigma_bench.py', '{data}/twosigma/', 'logloss', False, 0.002, _seeds(0, 1),
                       needs='{data}/twosigma/train.json'),
    'nyctaxi': _bench('nyctaxi_bench.py', '{data}/nyctaxi/', 'rmsle', False, 0.002, _seeds(0, 1),
                      needs='{data}/nyctaxi/NYC.csv'),
    'optiver': _bench('optiver_bench.py', '{data}/optiver/', 'rmspe', False, 0.01, _seeds(0, 1), extra=['--stocks', '12'],
                      needs='{data}/optiver/train.csv'),
    'vpp': _bench('vpp_bench.py', '{data}/vpp/train_folds.csv', 'mae', False, 0.02, _seeds(0, 1)),
    'rossmann': _bench('rossmann_bench.py', '{data}/rossmann/', 'rmspe', False, 0.003, _wins(0, 1), ff='fc',
                       needs='{data}/rossmann/train.csv'),
    'm5': _bench('m5_bench.py', '{data}/m5/', 'rmsse', False, 0.005, _wins(0, 1), ff='pipe',
                 needs='{data}/m5/sales_train_evaluation.csv'),
    'favorita': _bench('favorita_bench.py', '{data}/favorita/', 'nwrmsle', False, 0.005, _wins(0, 1), ff='pipe',
                       needs='{data}/favorita/train.parquet'),
    'walmart': _bench('walmart_bench.py', '{data}/walmart/', 'wmae', False, 30.0, _wins(0, 1), ff='fc',
                      needs='{data}/walmart/train.csv'),
    'recruit': _bench('recruit_bench.py', '{data}/recruit/', 'rmsle', False, 0.003, _wins(0, 1), ff='fc',
                      needs='{data}/recruit/air_visit_data.csv'),
    'riiid': _bench('riiid_bench.py', '{data}/riiid/', 'auc', True, 0.004, _seeds(0, 1), needs='{data}/riiid/riiid_train.parquet'),
    'elo': _bench('elo_bench.py', '{data}/elo/', 'rmse', False, 0.015, _seeds(0, 1), needs='{data}/elo/train.csv'),
    'amex': _bench('amex_bench.py', '{data}/amex/', 'amex', True, 0.004, _seeds(0, 1), needs='{data}/amex/train_labels.csv'),
    'instacart': _bench('instacart_bench.py', '{data}/instacart/', 'auc', True, 0.002, _seeds(0, 1), ff='rel_forge',
                        needs='{data}/instacart/orders.parquet'),
    'mercari': _bench('mercari_bench.py', '{data}/mercari/data.parquet', 'rmsle', False, 0.003, _seeds(0, 1), ff='forge'),
})


def fmt(x, ctx):
    if isinstance(x, list):
        return [fmt(v, ctx) for v in x]
    return x.format(**ctx) if isinstance(x, str) else x


def resolve(commit):
    return subprocess.run(['git', '-C', str(REPO), 'rev-parse', commit], check=True, capture_output=True,
                          text=True).stdout.strip()


def worktree(sha):
    d = WT / sha[:7]
    if not d.exists():
        subprocess.run(['git', '-C', str(REPO), 'worktree', 'add', '--detach', str(d), sha], check=True,
                       capture_output=True)
    # The benches of this branch, the same for every commit; FeatureForge and contest_features.py stay the commit's.
    src = list((REPO / 'scripts').glob('*_bench.py')) + [REPO / 'scripts' / '_pipe.py'] + \
        [p for p in HERE.iterdir() if p.suffix in ('.py', '.csv') and p.name != 'regress.py']
    for p in src:
        if p.name != 'contest_features.py':
            shutil.copy2(p, d / 'scripts' / p.name)
    return d


def peak_monitor(proc, out):
    import psutil
    try:
        root = psutil.Process(proc.pid)
    except psutil.Error:
        return
    while proc.poll() is None:
        rss = 0
        try:
            for p in [root] + root.children(recursive=True):
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
        out[0] = max(out[0], rss)
        time.sleep(2)


def run_job(case, cname, sample, arm, repo, outdir, shuffled=False, timeout=4 * 3600):
    tag = f'{sample}_{arm}' + ('_shuffled' if shuffled else '')
    out = outdir / cname / f'{tag}.json'
    if out.exists():
        return json.loads(out.read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path('/home/user/tmp') / f'regress_{cname}_{tag}'
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    ctx = dict(repo=str(repo), data=str(DATA), tmp=str(tmp))
    argv = [sys.executable, str(repo / 'scripts' / case['script'])] + fmt(case['args'], ctx) + \
        fmt(case['arms'][arm], ctx) + fmt(case['samples'][sample], ctx) + (case.get('shuffle', []) if shuffled else [])
    env = dict(os.environ, **{k: fmt(v, ctx) for k, v in case.get('env', {}).items()})
    log = out.with_suffix('.log')
    print(f'[{time.strftime("%H:%M:%S")}] {cname} {tag}: {" ".join(argv[1:])}', flush=True)
    t0, peak = time.time(), [0]
    with open(log, 'w') as f:
        proc = subprocess.Popen(argv, cwd=repo / 'scripts', env=env, stdout=f, stderr=subprocess.STDOUT)
        th = threading.Thread(target=peak_monitor, args=(proc, peak), daemon=True); th.start()
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill(); rc = 'timeout'
    wall = time.time() - t0
    lines = [l for l in log.read_text(errors='replace').splitlines() if l.startswith('RESULT ')]
    res = dict(case=cname, sample=sample, arm=arm, shuffled=shuffled, rc=rc, wall_s=round(wall),
               peak_gb=round(peak[0] / 2 ** 30, 2), argv=argv[1:])
    if rc == 0 and lines:
        r = json.loads(lines[-1][7:])
        res.update(score=r.get(case['metric']), fe_s=r.get('fe_s'), detail={k: v for k, v in r.items() if k not in
                                                                              ('feats',)})
    else:
        res.update(score=None, error=log.read_text(errors='replace')[-1500:])
    shutil.rmtree(tmp, ignore_errors=True)
    if res['score'] is not None:  # failures are rerun next time
        out.write_text(json.dumps(res, indent=1, default=str))
    print(f'    -> {res.get("score")} ({res["wall_s"]} s, {res["peak_gb"]} GB, rc={rc})', flush=True)
    return res


def cases_for(tier, names):
    if names:
        return {n: CASES[n] for n in names.split(',')}
    return {n: c for n, c in CASES.items() if tier == 'full' or c['tier'] == 'fast'}


def cmd_run(a):
    """Cases outer, commits inner: with --commit a,b each case's comparison is ready as soon as it is run."""
    shas = [resolve(c) for c in a.commit.split(',')]
    repos = {sha: worktree(sha) for sha in shas}
    rawdir = RESULTS / 'raw'
    for cname, case in cases_for(a.tier, a.cases).items():
        miss = [p for p in fmt(case['needs'], dict(data=str(DATA))) if not Path(p).exists()]
        if miss:
            print(f'skip {cname}: missing {miss}', flush=True); continue
        for sample in case['samples']:
            if not a.no_raw:
                run_job(case, cname, sample, 'raw', repos[shas[0]], rawdir)
            for sha in shas:
                run_job(case, cname, sample, 'ff', repos[sha], RESULTS / sha[:7])
        if (a.shuffled or a.tier == 'full') and case.get('shuffle'):
            for sha in shas:
                run_job(case, cname, next(iter(case['samples'])), 'ff', repos[sha], RESULTS / sha[:7], shuffled=True)


def load(d, cname, tag):
    p = d / cname / f'{tag}.json'
    return json.loads(p.read_text()) if p.exists() else None


def cmd_compare(a):
    ref, new = resolve(a.ref)[:7], resolve(a.new)[:7]
    rows, flags = [], []
    for cname, case in cases_for(a.tier, a.cases).items():
        hb = case['higher']
        per = []
        for s in case['samples']:
            r, x, y = load(RESULTS / 'raw', cname, f'{s}_raw'), load(RESULTS / ref, cname, f'{s}_ff'), load(RESULTS / new, cname, f'{s}_ff')
            per.append((s, r, x, y))
        got = [(s, r, x, y) for s, r, x, y in per if x and y]
        if not got:
            continue
        d = [(y['score'] - x['score']) * (1 if hb else -1) for s, r, x, y in got]  # > 0 = new better
        md = sum(d) / len(d)
        band = max(case['noise'], (max(d) - min(d)) / 2 if len(d) > 1 else 0)
        flag = ''
        if md < -band and all(v < 0 for v in d):
            flag = 'WORSE'
        elif md > band and all(v > 0 for v in d):
            flag = 'BETTER'
        if len(got) < len(per):
            flag += ' (partial)'
        f3 = lambda v: '-' if v is None else f'{v:.4f}'
        sc = lambda j: None if j is None else j['score']
        rows.append('| {} | {} | {} | {} | {} | {:+.4f} | ±{:.4f} | {} | {} | **{}** |'.format(
            cname, ' / '.join(f3(sc(r)) for s, r, x, y in got), ' / '.join(f3(sc(x)) for s, r, x, y in got),
            ' / '.join(f3(sc(y)) for s, r, x, y in got), ' / '.join(s for s, *_ in got), md, band,
            ' / '.join(f"{x.get('fe_s') or '-'}→{y.get('fe_s') or '-'}" for s, r, x, y in got),
            ' / '.join(f"{x['peak_gb']:.1f}→{y['peak_gb']:.1f}" for s, r, x, y in got), flag or 'same'))
        if flag.startswith('WORSE'):
            flags.append(cname)
    head = (f'## {ref} (reference) vs {new}, {a.tier} tier\n\n'
            '| case | raw | ref FF | new FF | samples | paired change (+ = new better) | noise band | build s ref→new | peak GB ref→new | flag |\n'
            '|---|---|---|---|---|---|---|---|---|---|')
    txt = head + '\n' + '\n'.join(rows) + f'\n\nWorse beyond noise: {", ".join(flags) or "none"}\n'
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run'); r.add_argument('--commit', required=True, help='one commit or a comma list'); r.add_argument('--tier', default='fast')
    r.add_argument('--cases', default=''); r.add_argument('--shuffled', action='store_true')
    r.add_argument('--no-raw', action='store_true')
    c = sp.add_parser('compare'); c.add_argument('--ref', required=True); c.add_argument('--new', required=True)
    c.add_argument('--tier', default='fast'); c.add_argument('--cases', default=''); c.add_argument('--out', default='')
    sp.add_parser('list')
    a = ap.parse_args()
    if a.cmd == 'run':
        cmd_run(a)
    elif a.cmd == 'compare':
        cmd_compare(a)
    else:
        for n, c in CASES.items():
            print(f"{n:14s} {c['tier']:5s} {c['metric']:8s} {'higher' if c['higher'] else 'lower'} "
                  f"samples={list(c['samples'])} noise={c['noise']}")


if __name__ == '__main__':
    main()
