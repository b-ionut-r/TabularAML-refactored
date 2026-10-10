"""Shared ``pipe`` arm for the forecasting benches: the bench's own train and test frames given to
``scripts/contest_features.py`` at its shipped defaults (nothing else). The original columns are kept as the
bench built them and the pipeline's new columns are appended, so the judge sees the same raw columns as the
other arms."""
import sys, shutil, subprocess, tempfile
from pathlib import Path
import numpy as np, pandas as pd


def pipe_features(Xtr, ytr, Xte, task='regression', budget=900, extra=()):
    tmp = Path(tempfile.mkdtemp(dir='/home/user/tmp'))
    try:
        Xtr.assign(__y=np.asarray(ytr)).to_parquet(tmp / 'train.parquet'); Xte.to_parquet(tmp / 'test.parquet')
        cmd = [sys.executable, str(Path(__file__).resolve().parent / 'contest_features.py'), '--train', str(tmp / 'train.parquet'),
               '--test', str(tmp / 'test.parquet'), '--target', '__y', '--task', task, '--budget', str(budget),
               '--out-dir', str(tmp / 'out'), *extra]
        subprocess.run(cmd, check=True)
        Ftr = pd.read_parquet(tmp / 'out' / 'train_features.parquet'); Fte = pd.read_parquet(tmp / 'out' / 'test_features.parquet')
        assert len(Ftr) == len(Xtr) and len(Fte) == len(Xte)
        new = [c for c in Ftr.columns if c not in Xtr.columns and c != '__y']
        Ftr, Fte = Ftr[new].reset_index(drop=True), Fte[new].reset_index(drop=True)
        for c in new:
            if not pd.api.types.is_numeric_dtype(Ftr[c]):
                cats = pd.Index(pd.unique(Ftr[c].astype(str)))
                Ftr[c] = pd.Categorical(Ftr[c].astype(str), categories=cats); Fte[c] = pd.Categorical(Fte[c].astype(str), categories=cats)
        return (pd.concat([Xtr.reset_index(drop=True), Ftr], axis=1), pd.concat([Xte.reset_index(drop=True), Fte], axis=1),
                dict(n_new=len(new)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
