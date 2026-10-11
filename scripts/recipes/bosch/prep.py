"""Bosch Production Line Performance (2016): the first N training parts by Id (the files are in Id order), the
latest 20% held out as the test. Numeric and date fields as float32, categorical fields dictionary-encoded, all
three files side by side (about 4,260 columns, mostly empty). Writes hold/{train,test,hold_y,train_shuf}.parquet.

    python3 -I prep.py <dir with train_numeric.csv, train_date.csv, train_categorical.csv> <out dir> [N]
"""
import os, sys
import numpy as np
import pyarrow as pa, pyarrow.csv as pc, pyarrow.compute as pcm, pyarrow.parquet as pq

src, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300_000
os.makedirs(out, exist_ok=True)


def batches(name, dtype):
    path = f"{src}/train_{name}.csv"
    cols = open(path).readline().strip().split(",")
    types = {c: (pa.int64() if c in ("Id", "Response") else dtype) for c in cols}
    r = pc.open_csv(path, read_options=pc.ReadOptions(block_size=1 << 24),
                    convert_options=pc.ConvertOptions(column_types=types, strings_can_be_null=True))
    n = 0
    for b in r:
        yield b.slice(0, N - n)
        n += min(b.num_rows, N - n)
        if n >= N:
            return


def head(name, dtype):
    return pa.Table.from_batches(list(batches(name, dtype))).combine_chunks()


def head_categorical():
    # Two passes, codes as int8 over each column's few values ("T1", "T16", ...): string arrays of
    # 300k x 2,140 mostly empty cells need gigabytes for their offsets alone.
    vals = {}
    for b in batches("categorical", pa.string()):
        for c in b.schema.names[1:]:
            vals.setdefault(c, set()).update(v for v in pcm.unique(b[c]).to_pylist() if v is not None)
    dicts = {c: pa.array(sorted(v), pa.string()) for c, v in vals.items()}
    assert max(len(d) for d in dicts.values()) < 127
    parts = {c: [] for c in dicts}
    for b in batches("categorical", pa.string()):
        for c in dicts:
            parts[c].append(pcm.index_in(b[c], value_set=dicts[c]).cast(pa.int8()))
    return pa.table({c: pa.DictionaryArray.from_arrays(pa.concat_arrays(parts[c]), dicts[c]) for c in dicts})


cat = head_categorical()
num = head("numeric", pa.float32())
date = head("date", pa.float32()).drop_columns(["Id"])
assert num.num_rows == date.num_rows == cat.num_rows == N
t = pa.table({**{c: num[c] for c in num.column_names}, **{c: date[c] for c in date.column_names},
              **{c: cat[c] for c in cat.column_names}})
del num, date, cat
k = int(0.8 * N)
tr, te = t.slice(0, k), t.slice(k)
pq.write_table(tr, f"{out}/train.parquet")
pq.write_table(te.select(["Response"]), f"{out}/hold_y.parquet")
pq.write_table(te.drop_columns(["Response"]), f"{out}/test.parquet")
y = tr["Response"].to_numpy()
pq.write_table(tr.set_column(tr.column_names.index("Response"), "Response",
                             pa.array(np.random.default_rng(0).permutation(y))), f"{out}/train_shuf.parquet")
print("train", tr.shape, "positives", int(y.sum()), "test", te.shape, "positives", int(te["Response"].to_numpy().sum()))
