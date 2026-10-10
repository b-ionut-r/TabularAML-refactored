"""Fetch one contest's data for the regression bench into DATA/<name>/ (the public copies the benches name).

    python scripts/regress/fetch.py amazon telstra ...      # skips what is there (a .done file)
    python scripts/regress/fetch.py --drop elo                # delete a big one after its runs

Kaggle datasets and competition files go through the proxy (it injects the credentials); Hugging Face files are
spaced out (the mirror rate-limits bursts); OpenML through the openml package.
"""
import argparse, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

DATA = Path(os.environ.get('REGRESS_DATA', '/home/user/data'))
KG = 'https://www.kaggle.com/api/v1'


def curl(url, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['curl', '-sSL', '--fail', '--retry', '4', '-o', str(out), url], check=True)


def kaggle_files(ds, files, d, rename=None):
    for f in files:
        out = d / (rename or {}).get(f, Path(f).name)
        curl(f'{KG}/datasets/download/{ds}?fileName={f}', out)
        if open(out, 'rb').read(4) == b'PK\x03\x04':  # big files come zipped
            z = zipfile.ZipFile(out); inner = z.namelist()[0]; z.extract(inner, d); z.close()
            out.unlink(); (d / inner).rename(out)


def hf(repo, files, d):
    for f in files:
        curl(f'https://huggingface.co/datasets/{repo}/resolve/main/{f}', d / Path(f).name); time.sleep(20)


def openml(did, d, name):
    import openml as om
    om.config.set_root_cache_directory(str(DATA / '_openml_cache'))
    ds = om.datasets.get_dataset(did, download_data=True, download_qualities=False, download_features_meta_data=False)
    X, *_ = ds.get_data(dataset_format='dataframe')
    X.to_parquet(d / name)
    shutil.rmtree(DATA / '_openml_cache', ignore_errors=True)


def csv_to_parquet(d, names):
    import pandas as pd
    for n in names:
        pd.read_csv(d / f'{n}.csv').to_parquet(d / f'{n}.parquet'); (d / f'{n}.csv').unlink()


def amex_labels(d):
    import pyarrow.parquet as pq
    t = pq.read_table(d / 'labeled_train.parquet', columns=['customer_ID', 'target']).to_pandas()
    t.drop_duplicates('customer_ID').to_csv(d / 'train_labels.csv', index=False); (d / 'labeled_train.parquet').unlink()


def optiver(d):
    """train.csv in full; book and trade files only for the stocks the bench samples (12 stocks, stock seed 0),
    with an empty folder for every other stock so the bench's sampling over the folder list is unchanged."""
    import numpy as np, urllib.request
    kaggle_files('akshaymairal/optiver-realized-volatility-prediction', ['train.csv'], d)
    lst = json.load(urllib.request.urlopen(f'{KG}/datasets/list/akshaymairal/optiver-realized-volatility-prediction?pageSize=1000'))
    files = [f['name'] for f in (lst.get('datasetFiles') or [])]
    allst = sorted({int(f.split('stock_id=')[1].split('/')[0]) for f in files if f.startswith('book_train.parquet/')})
    pick = set(sorted(np.random.default_rng(0).choice(allst, 12, replace=False).tolist()))
    for s in allst:
        for kind in ('book', 'trade'):
            (d / f'{kind}_train.parquet' / f'stock_id={s}').mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.startswith(('book_train.parquet/', 'trade_train.parquet/')) and int(f.split('stock_id=')[1].split('/')[0]) in pick:
            kaggle_files('akshaymairal/optiver-realized-volatility-prediction', [f], d / f.rsplit('/', 1)[0])
    return sorted(pick)


SOURCES = {
    'amazon': lambda d: openml(4135, d, 'amazon.pq'),
    'kdd12': lambda d: openml(1216, d, 'kdd12.pq'),
    'cover': lambda d: openml(1596, d, 'cover.pq'),
    'scs': lambda d: openml(46634, d, 'scs.pq'),
    'porto': lambda d: openml(42742, d, 'porto.pq'),
    'allstate': lambda d: openml(42571, d, 'as.pq'),
    'loandefault': lambda d: openml(6331, d, 'ld.pq'),
    'telstra': lambda d: kaggle_files('yifanxie/telstra-competition-dataset', ['train.csv', 'test.csv', 'event_type.csv',
                                      'log_feature.csv', 'resource_type.csv', 'severity_type.csv'], d),
    'bnp': lambda d: kaggle_files('hjimbean/kaggle-classification-autofe-benchmark',
                                  ['data/bnp-paribas-cardif-claims-management/train.csv'], d),
    'liberty': lambda d: kaggle_files('hjimbean/kaggle-classification-autofe-benchmark',
                                      ['data/liberty-mutual-group-property-inspection-prediction/train.csv',
                                       'data/liberty-mutual-group-property-inspection-prediction/test.csv'], d),
    'homesite': lambda d: kaggle_files('thanhvv/homesite-quote-conversion', ['train.csv', 'test.csv'], d),
    'twosigma': lambda d: kaggle_files('logan1997/two-sigma-challenge', ['train.json'], d),
    'nyctaxi': lambda d: kaggle_files('yasserh/nyc-taxi-trip-duration', ['NYC.csv'], d),
    'vpp': lambda d: kaggle_files('jarupula/google-vpp-train-5-folds', ['train_folds.csv'], d),
    'riiid': lambda d: kaggle_files('rohanrao/riiid-train-data-multiple-formats', ['riiid_train.parquet'], d),
    'elo': lambda d: kaggle_files('ershisuila/eio-recommend', ['train.csv', 'historical_transactions.csv',
                                                               'new_merchant_transactions.csv'], d),
    'amex': lambda d: (kaggle_files('raddar/amex-data-integer-dtypes-parquet-format', ['train.parquet'], d),
                       kaggle_files('jeonbyungsu/amex-data', ['labeled_train.parquet'], d), amex_labels(d)),
    'optiver': optiver,
    'favorita': lambda d: hf('hsnalmasri/favorita', ['items.parquet', 'stores.parquet', 'train.parquet'], d),
    'm5': lambda d: hf('denephew/M5_Forecasting', ['calendar.csv', 'sales_train_evaluation.csv', 'sell_prices.csv'], d),
    'dsb': lambda d: hf('pytorch-lifestream/datascience-bowl2019', ['train.csv.gz', 'train_labels.csv.gz'], d),
    'instacart': lambda d: (hf('attik/Instacart-Market-Basket-Analysis', ['orders.csv', 'order_products__prior.csv',
                                                                          'order_products__train.csv', 'products.csv'], d),
                            csv_to_parquet(d, ['orders', 'order_products__prior', 'order_products__train', 'products'])),
    'mercari': lambda d: hf('multabench/core-text-reg-mercari-marketplace', ['data.parquet'], d),
    'walmart': lambda d: hf('large-traversaal/Walmart-sales', ['train.csv', 'stores.csv', 'features.csv'], d),
    'rossmann': lambda d: hf('AiiN-aini/rossmann-store-sales', ['train.csv', 'store.csv'], d),
    'recruit': lambda d: [curl('https://raw.githubusercontent.com/MengenL-ds/Forecasting-Restaurant-Visitor-Demand-with-'
                               f'Machine-Learning/main/data/raw/{f}.csv', d / f'{f}.csv')
                          for f in ['air_visit_data', 'air_store_info', 'hpg_store_info', 'store_id_relation',
                                    'date_info', 'air_reserve', 'hpg_reserve']],
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('names', nargs='+'); ap.add_argument('--drop', action='store_true')
    a = ap.parse_args()
    for n in a.names:
        d = DATA / n
        if a.drop:
            shutil.rmtree(d, ignore_errors=True); print('dropped', n); continue
        if (d / '.done').exists():
            print('have', n); continue
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            SOURCES[n](d)
        except Exception as e:
            print(f'FAILED {n}: {e!r}', flush=True); continue
        (d / '.done').touch()
        size = sum(p.stat().st_size for p in d.rglob('*') if p.is_file()) / 2 ** 20
        print(f'fetched {n}: {size:.0f} MB in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
