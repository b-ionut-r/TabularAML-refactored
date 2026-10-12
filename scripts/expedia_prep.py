"""Expedia Hotel Recommendations (2016): every event (searches and bookings) of 3% of users, streamed from the
contest's train.csv (public copy monishc/expediatrainingset, 4,070,445,781 bytes = the contest file).
Usage: expedia_prep.py out.parquet [percent of users, default 3]; users are kept by hash(user_id) % 100 < percent,
so the 1% sample is a subset of the 3% one."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__file__))
from kaggle_zstream import open_csv
pct = int(sys.argv[2]) if len(sys.argv) > 2 else 3
parts = []
for ch in pd.read_csv(open_csv('monishc/expediatrainingset', 'train.csv'), chunksize=2_000_000):
    h = pd.util.hash_pandas_object(ch.user_id, index=False).to_numpy()
    parts.append(ch[(h % 100) < pct])
    print(sum(map(len, parts)), flush=True)
d = pd.concat(parts, ignore_index=True)
d.to_parquet(sys.argv[1])
print(d.shape, d.is_booking.mean(), d.user_id.nunique(), d.date_time.min(), d.date_time.max())
