import numpy as np
from sklearn.model_selection import GroupKFold, TimeSeriesSplit


class RotatedGroupKFold:
    """GroupKFold with optional shuffled group-to-fold assignment."""

    def __init__(self, n_splits: int, rotation: int | None = None):
        self.n_splits = n_splits
        self.rotation = rotation

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def rotated(self, generation):
        return RotatedGroupKFold(self.n_splits, rotation=int(generation))

    def split(self, X, y=None, groups=None):
        if self.rotation is None:
            yield from GroupKFold(self.n_splits).split(X, y, groups)
            return
        if groups is None:
            raise ValueError("RotatedGroupKFold requires groups")
        groups_arr = np.asarray(groups)
        unique_groups = np.unique(groups_arr)
        shuffled = unique_groups.copy()
        np.random.RandomState(self.rotation).shuffle(shuffled)
        fold_groups = [set(shuffled[i::self.n_splits]) for i in range(self.n_splits)]
        all_idx = np.arange(len(groups_arr))
        for val_groups in fold_groups:
            val_mask = np.isin(groups_arr, list(val_groups))
            yield all_idx[~val_mask], all_idx[val_mask]


class RotatedTimeSeriesSplit:
    """TimeSeriesSplit with chronological window rotation and future meta split."""

    def __init__(self, base_splitter: TimeSeriesSplit, rotation: int = 0):
        self.base = base_splitter
        self.n_splits = base_splitter.n_splits
        self.gap = getattr(base_splitter, "gap", 0)
        self.test_size = getattr(base_splitter, "test_size", None)
        self.max_train_size = getattr(base_splitter, "max_train_size", None)
        self.rotation = int(rotation)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def rotated(self, generation):
        return RotatedTimeSeriesSplit(self.base, rotation=int(generation))

    def _windows(self, n_units):
        test_size = self.test_size or max(1, n_units // (self.n_splits + 1))
        first_val = n_units - self.n_splits * test_size
        max_shift = max(0, min(test_size - 1, first_val - self.gap - 1))
        shift = self.rotation % (max_shift + 1) if max_shift else 0
        for split_idx in range(self.n_splits):
            val_start = first_val - shift + split_idx * test_size
            val_end = val_start + test_size
            train_end = val_start - self.gap
            if train_end <= 0 or val_start < 0 or val_end > n_units:
                continue
            train_start = max(0, train_end - self.max_train_size) if self.max_train_size else 0
            yield train_start, train_end, val_start, val_end

    def split(self, X, y=None, groups=None):
        if groups is None:
            for tr0, tr1, va0, va1 in self._windows(len(X)):
                yield np.arange(tr0, tr1), np.arange(va0, va1)
            return
        groups_arr = np.asarray(groups)
        periods = np.sort(np.unique(groups_arr))
        for tr0, tr1, va0, va1 in self._windows(len(periods)):
            train_mask = np.isin(groups_arr, periods[tr0:tr1])
            val_mask = np.isin(groups_arr, periods[va0:va1])
            yield np.where(train_mask)[0], np.where(val_mask)[0]

    def split_meta(self, X, y=None, groups=None, frac=0.15, random_state=None):
        units = np.arange(len(X)) if groups is None else np.sort(np.unique(groups))
        n_meta = min(max(1, int(np.ceil(len(units) * frac))), max(1, len(units) - 1))
        meta_start = len(units) - n_meta
        search_end = meta_start - self.gap
        if search_end <= 0:
            raise ValueError(f"Not enough time units for meta-validation with frac={frac} and gap={self.gap}")
        if groups is None:
            return np.arange(search_end), np.arange(meta_start, len(X))
        groups_arr = np.asarray(groups)
        search_mask = np.isin(groups_arr, units[:search_end])
        meta_mask = np.isin(groups_arr, units[meta_start:])
        return np.where(search_mask)[0], np.where(meta_mask)[0]


class PurgedTimeSeriesSplit(RotatedTimeSeriesSplit):
    """Period-level TimeSeriesSplit wrapper used by the UI."""

    def __init__(self, tss: TimeSeriesSplit, unique_periods=None, groups=None, rotation: int = 0):
        super().__init__(tss, rotation=rotation)
        self._periods = unique_periods
        self._groups = groups

    def rotated(self, generation):
        return PurgedTimeSeriesSplit(self.base, self._periods, self._groups, rotation=int(generation))

    def split(self, X, y=None, groups=None):
        yield from super().split(X, y=y, groups=groups if groups is not None else self._groups)

    def split_meta(self, X, y=None, groups=None, frac=0.15, random_state=None):
        return super().split_meta(
            X,
            y=y,
            groups=groups if groups is not None else self._groups,
            frac=frac,
            random_state=random_state,
        )


def normalize_rotatable_splitter(cv):
    if isinstance(cv, TimeSeriesSplit):
        return RotatedTimeSeriesSplit(cv)
    if isinstance(cv, GroupKFold):
        return RotatedGroupKFold(cv.n_splits)
    return cv
