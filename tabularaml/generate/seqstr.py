"""Structured strings: one token per row whose characters carry the information.

A molecule's SMILES (``*CC(*)c1ccccc1C(=O)O``), a protein or DNA sequence, a part number built from
segments: mostly distinct values, no spaces, so neither the category nor the free-text family reads
them (NeurIPS Open Polymer Prediction 2025 had nothing but this column). Label-free views:

* ``CharGrams``: counts of the most common character 1-3-grams (ring closures, branches, atoms), and
  the length;
* ``CharSVD``: latent components of character 1-4-gram TF-IDF;
* ``MolDescriptors`` / ``MolFingerprint``: when the values parse as SMILES and RDKit is installed, the
  2D descriptors and Morgan count fingerprint that the polymer contest's public notebooks started from.

The sparse linear model on character n-grams is forge's ``TextLinearOOF``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .forge import Spec, TextLinearOOF
from .strs import as_text


def _values(df, c) -> pd.Series:
    t = as_text(df[c])
    return t.where(t != "__NA__", "")


def is_seq_string(s: pd.Series, n_sample: int = 20_000) -> bool:
    """Mostly distinct single tokens of six or more characters, drawn from a varied character set."""
    v = as_text(s.dropna())
    v = v[v != "__NA__"]
    if len(v) < 100:
        return False
    if len(v) > n_sample:
        v = v.sample(n_sample, random_state=0)
    if v.nunique() < max(50, 0.3 * len(v)):
        return False
    ln = v.str.len()
    if float(ln.mean()) < 6 or float(v.str.count(" ").mean()) >= 0.5:
        return False
    # Hashed ids and fixed-width codes (Avazu's device ids, card hashes) are keys, not sequences: a
    # sequence's length varies with its content.
    if float(ln.std()) < 0.1 * float(ln.mean()) or v.str.fullmatch(r"[0-9a-fA-F-]+").mean() >= 0.9:
        return False
    return float(v.map(lambda t: len(set(t))).mean()) >= 5


def _rdkit():
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        return Chem
    except Exception:
        return None


def is_smiles(s: pd.Series) -> bool:
    """Nine in ten sampled values parse as molecules (RDKit installed)."""
    Chem = _rdkit()
    if Chem is None:
        return False
    v = as_text(s.dropna()).drop_duplicates()
    v = v.sample(min(len(v), 300), random_state=0)
    if not len(v) or not v.str.contains(r"[CNOcno]").mean() >= 0.9:
        return False
    return float(np.mean([Chem.MolFromSmiles(t) is not None for t in v])) >= 0.9


class _PerValue(Spec):
    """Computed once per distinct string, then mapped to the rows."""

    def _per_value(self, df, fn) -> np.ndarray:
        t = _values(df, self.parents[0])
        codes, uniq = pd.factorize(t)
        out = fn(list(uniq))
        return out[codes].astype(np.float32)


class CharGrams(_PerValue):
    def __init__(self, col: str, k: int = 64):
        super().__init__([col])
        self.k = k
        self.n_out = k + 1
        self.name = f"chargram__{col}"

    def fit(self, df, y, ctx):
        from sklearn.feature_extraction.text import CountVectorizer
        c = self.parents[0]
        t = _values(df, c)
        extra = getattr(ctx, "extra_rows", None)
        if extra is not None and c in extra:
            t = pd.concat([t, _values(extra, c)], ignore_index=True)
        t = t.drop_duplicates()
        if len(t) > 200_000:
            t = t.sample(200_000, random_state=0)
        self.vec_ = CountVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False, max_features=self.k,
                                    dtype=np.float32).fit(t)
        self.n_out = len(self.vec_.vocabulary_) + 1
        return self

    def transform(self, df, ctx):
        def fn(u):
            C = self.vec_.transform(u).toarray()
            return np.column_stack([C, np.array([len(x) for x in u], dtype=np.float32)])
        return self._per_value(df, fn)


class CharSVD(_PerValue):
    def __init__(self, col: str, n_comp: int = 16):
        super().__init__([col])
        self.n_out = n_comp
        self.name = f"charsvd__{col}"

    def fit(self, df, y, ctx):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        c = self.parents[0]
        t = _values(df, c)
        extra = getattr(ctx, "extra_rows", None)
        if extra is not None and c in extra:
            t = pd.concat([t, _values(extra, c)], ignore_index=True)
        t = t.drop_duplicates()
        if len(t) > 200_000:
            t = t.sample(200_000, random_state=0)
        self.vec_ = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), lowercase=False, min_df=2,
                                    max_features=50_000, sublinear_tf=True, dtype=np.float32).fit(t)
        T = self.vec_.transform(t)
        self.n_out = int(min(self.n_out, max(1, T.shape[1] - 1), max(1, T.shape[0] - 1)))
        self.svd_ = TruncatedSVD(self.n_out, random_state=0, algorithm="randomized", n_iter=4).fit(T)
        return self

    def transform(self, df, ctx):
        return self._per_value(df, lambda u: self.svd_.transform(self.vec_.transform(u)))


class MolDescriptors(_PerValue):
    """Every RDKit 2D descriptor of the molecule (a polymer's '*' ends as dummy atoms)."""

    def __init__(self, col: str):
        from rdkit.Chem import Descriptors
        super().__init__([col])
        self.fns = list(Descriptors.descList)
        self.n_out = len(self.fns)
        self.name = f"moldesc__{col}"

    def out_names(self):
        return [f"{self.name}_{n}" for n, _ in self.fns]

    def transform(self, df, ctx):
        Chem = _rdkit()

        def fn(u):
            out = np.full((len(u), len(self.fns)), np.nan, dtype=np.float64)
            for i, s in enumerate(u):
                m = Chem.MolFromSmiles(s) if s else None
                if m is None:
                    continue
                for j, (_, f) in enumerate(self.fns):
                    try:
                        out[i, j] = f(m)
                    except Exception:
                        pass
            out[~np.isfinite(out)] = np.nan
            return np.clip(out, -1e30, 1e30)
        return self._per_value(df, fn)


class MolFingerprint(_PerValue):
    """Morgan count fingerprint (radius 2) folded to ``bits`` positions."""

    def __init__(self, col: str, bits: int = 256):
        super().__init__([col])
        self.bits = bits
        self.n_out = bits
        self.name = f"molfp__{col}"

    def transform(self, df, ctx):
        Chem = _rdkit()
        from rdkit.Chem import rdFingerprintGenerator
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=self.bits)

        def fn(u):
            out = np.zeros((len(u), self.bits), dtype=np.float32)
            for i, s in enumerate(u):
                m = Chem.MolFromSmiles(s) if s else None
                if m is not None:
                    out[i] = gen.GetCountFingerprintAsNumPy(m)
            return out
        return self._per_value(df, fn)


def seq_specs(cols: Sequence[str], smiles: Sequence[str], n_classes: int):
    """The family's candidates for the structured-string columns."""
    out = []
    for c in cols:
        out += [CharGrams(c), CharSVD(c), TextLinearOOF([c], n_classes, label=f"seq_{c}")]
        if c in smiles:
            out += [MolDescriptors(c), MolFingerprint(c)]
    return out
