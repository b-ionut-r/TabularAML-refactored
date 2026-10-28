import numpy as np
import pandas as pd

from tabularaml.generate.returns import book_levels, return_features


def _book(n_keys=60, rows=80, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for k in range(n_keys):
        vol = 0.0005 * (1 + k % 5)
        mid = np.exp(np.cumsum(rng.normal(0, vol, rows)))
        spread = 0.0002 * mid
        parts.append(pd.DataFrame({"key": k, "sec": np.sort(rng.choice(600, rows, replace=False)),
                                   "bid_price1": mid - spread, "ask_price1": mid + spread,
                                   "bid_size1": rng.integers(1, 500, rows), "ask_size1": rng.integers(1, 500, rows)}))
    return pd.concat(parts, ignore_index=True)


def test_volatility_per_key_tracks_the_true_one():
    df = _book()
    F = return_features(df, "key", "book")
    assert {"book__bid_price1__rv", "book__mid__rv", "book__wap_1__rv", "book__wap_1__rv_late"} <= set(F.columns)
    assert len(F) == 60
    true = 0.0005 * (1 + F.index.to_numpy() % 5)
    assert np.corrcoef(F["book__mid__rv"], true)[0, 1] > 0.95
    # rows shuffled within the file: the time column puts them back in order
    G = return_features(df.sample(frac=1, random_state=0), "key", "book", time="sec")
    assert np.allclose(G.loc[F.index, "book__mid__rv"], F["book__mid__rv"], rtol=1e-4)


def test_levels_and_off_without_prices():
    df = _book()
    assert book_levels(df.columns, ["bid_price1", "ask_price1"]) == {"wap_1": ("bid_price1", "bid_size1", "ask_price1", "ask_size1")}
    sizes = df[["key", "sec", "bid_size1", "ask_size1"]]
    assert return_features(sizes, "key", "book").shape[1] == 0
    amounts = df[["key", "sec"]].assign(amount=np.random.default_rng(1).lognormal(5, 1, len(df)))
    assert return_features(amounts, "key", "pay").shape[1] == 0   # jumps by far more than 1% a row


def test_levels_under_other_names():
    cols = ["BidPx_L1", "AskPx_L1", "BidQty_L1", "AskQty_L1", "buy_price_0", "sell_price_0", "buy_volume_0", "sell_volume_0"]
    lv = book_levels(cols, ["BidPx_L1", "AskPx_L1", "buy_price_0", "sell_price_0"])
    assert lv == {"wap_l1": ("BidPx_L1", "BidQty_L1", "AskPx_L1", "AskQty_L1"),
                  "wap_0": ("buy_price_0", "buy_volume_0", "sell_price_0", "sell_volume_0")}
    # A price alone, or a bid with no ask, is not a book level.
    assert book_levels(["price", "sale_price", "size", "bid_price", "bid_size"], ["price", "sale_price", "bid_price"]) == {}
    df = _book().rename(columns={"bid_price1": "BidPx_L1", "ask_price1": "AskPx_L1", "bid_size1": "BidQty_L1", "ask_size1": "AskQty_L1"})
    assert "book__wap_l1__rv" in return_features(df, "key", "book").columns
