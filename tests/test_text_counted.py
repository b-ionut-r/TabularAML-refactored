import numpy as np
import pandas as pd

from tabularaml.generate.forge import Context, TextLinearOOF, TextSVD, _fit_counted, _tfidf


def _corpus(seed, n_rows, n_texts, words):
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(words)]
    texts = [" ".join(rng.choice(vocab, rng.integers(1, 40))) for _ in range(n_texts)]
    return pd.Series(np.asarray(texts, dtype=object)[rng.integers(0, n_texts, n_rows)])


def test_counted_fit_matches_fit_on_the_copies():
    for seed, kind, top in [(0, "word", 200_000), (1, "word", 50), (2, "char", 100_000), (3, "char", 40)]:
        t = _corpus(seed, 3000, 120, 300)
        codes, u = pd.factorize(t)
        ref = _tfidf(kind, top).fit(t)
        new = _fit_counted(_tfidf(kind, top), pd.Series(np.asarray(u, dtype=object)), codes)
        assert ref.vocabulary_ == new.vocabulary_ and list(ref.vocabulary_) == list(new.vocabulary_)
        assert ref.idf_.dtype == new.idf_.dtype and np.array_equal(ref.idf_, new.idf_)
        a, b = ref.transform(t), new.transform(t)
        assert (a != b).nnz == 0 and np.array_equal(a.indices, b.indices)


def test_text_specs_unchanged_on_repeated_text():
    t = _corpus(4, 2000, 60, 200)
    df = pd.DataFrame({"txt": t.astype("category")})
    extra = pd.DataFrame({"txt": _corpus(5, 700, 60, 200)})
    ctx = Context("regression", 0, 0)
    ctx.extra_rows = extra
    y = np.random.default_rng(0).normal(size=len(df))
    folds = [(np.where(np.arange(len(df)) % 3 != k)[0], np.where(np.arange(len(df)) % 3 == k)[0]) for k in range(3)]
    v_svd = TextSVD("txt", max_fit_rows=1500).fit(df, None, ctx).transform(df, ctx)
    v_lin = TextLinearOOF(["txt"], 0).fit_transform_oof(df, y, ctx, folds)
    # the same specs fitted the old way: every copy of a text analysed
    from sklearn.decomposition import TruncatedSVD
    from tabularaml.generate.forge import TEXT_CLIP, _text_values
    tt = pd.concat([_text_values(df, "txt", TEXT_CLIP), _text_values(extra, "txt", TEXT_CLIP)], ignore_index=True)
    ts = tt.sample(1500, random_state=0)
    vec = _tfidf("word", 50_000).fit(ts)
    T = vec.transform(ts)
    svd = TruncatedSVD(min(16, T.shape[1] - 1), random_state=0, algorithm="randomized", n_iter=4).fit(T)
    assert np.array_equal(v_svd, svd.transform(vec.transform(_text_values(df, "txt", TEXT_CLIP))).astype(np.float32))
    assert np.isfinite(v_lin).all()
