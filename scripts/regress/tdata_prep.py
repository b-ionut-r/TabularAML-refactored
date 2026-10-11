"""TalkingData AdTracking (2018): stream train.csv (184.9M clicks) from stdin and keep every click of 2% of IPs
(hash(ip) % 50 == 0), so per-IP histories stay whole. Writes /home/user/data/sealed/talkingdata/clicks.parquet.
    unzip -p train.csv.zip | python3 tdata_prep.py"""
import sys
import pandas as pd
out = []
for ch in pd.read_csv(sys.stdin, usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'],
                      dtype={'ip': 'int64', 'app': 'int32', 'device': 'int32', 'os': 'int32', 'channel': 'int32',
                             'is_attributed': 'int8'}, chunksize=5_000_000):
    out.append(ch[(ch['ip'] * 2654435761 % 2**32) % 50 == 0])
d = pd.concat(out, ignore_index=True)
d['click_time'] = pd.to_datetime(d['click_time'])
d.to_parquet('/home/user/data/sealed/talkingdata/clicks.parquet')
print(d.shape, d['click_time'].min(), d['click_time'].max(), d['is_attributed'].mean())
