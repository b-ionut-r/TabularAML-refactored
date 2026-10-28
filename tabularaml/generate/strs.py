"""String views of columns without fixed-width copies.

``Series.astype(str)`` on a categorical converts its categories to a fixed-width numpy
unicode array first: one long-text level (MCTS's game rules, a keystroke log's pasted text)
makes every level that wide, and a few thousand of them ask for tens or hundreds of GiB.
These helpers map categories through Python strings instead, with the values pandas 2's
``astype(str)`` gives ("nan" for missing) on pandas 2 and 3 alike.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def as_text(s) -> pd.Series:
    """``s.astype(str)`` as an object Series, for categoricals without fixed-width copies."""
    if isinstance(s, pd.Index):
        return pd.Index(as_text(pd.Series(s)).to_numpy(), dtype=object)
    if isinstance(s.dtype, pd.CategoricalDtype):
        cats = np.asarray(s.cat.categories.astype(object).map(str), dtype=object)
        codes = s.cat.codes.to_numpy()
        out = np.full(len(s), "nan", dtype=object)
        ok = codes >= 0
        out[ok] = cats[codes[ok]]
        return pd.Series(out, index=s.index, name=s.name, dtype=object)
    # Missing values as "nan" on every pandas version (pandas 3's astype(str) keeps them missing).
    return s.astype(object).where(s.notna(), "nan").map(str).astype(object)


def as_text_frame(df: pd.DataFrame) -> pd.DataFrame:
    """``df.astype(str)``, column by column through ``as_text``."""
    return pd.DataFrame({c: as_text(df[c]) for c in df.columns}, index=df.index)
