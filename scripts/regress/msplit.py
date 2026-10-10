import pandas as pd, numpy as np, os
D = '/tmp/claude-0/data/malware'; H = f'{D}/hold'; os.makedirs(H, exist_ok=True)
d = pd.read_parquet(f'{D}/train.parquet')
for c in d.columns:
    if d[c].dtype == object or pd.api.types.is_string_dtype(d[c]):
        d[c] = d[c].astype('category')
v = d.AvSigVersion.astype(str).str.split('.', expand=True)
key = pd.to_numeric(v[1], errors='coerce').fillna(0) * 1e5 + pd.to_numeric(v[2], errors='coerce').fillna(0)
d['_k'] = key.to_numpy()
cut = np.quantile(d._k, 0.8)
hold = d._k > cut
print('cut', cut, 'holdout rows', hold.sum(), 'train rows', (~hold).sum(), 'pos rate', d.HasDetections[~hold].mean(), d.HasDetections[hold].mean())
tr = d[~hold].sort_values('_k', kind='stable').drop(columns='_k').reset_index(drop=True)
te = d[hold].drop(columns='_k').reset_index(drop=True)
tr.to_parquet(f'{H}/train.parquet', index=False)
te[['MachineIdentifier', 'HasDetections']].to_parquet(f'{H}/hold_y.parquet', index=False)
te.drop(columns='HasDetections').to_parquet(f'{H}/test.parquet', index=False)
tr2 = tr.copy(); tr2['HasDetections'] = np.random.default_rng(0).permutation(tr2.HasDetections.to_numpy())
tr2.to_parquet(f'{H}/train_shuf.parquet', index=False)
