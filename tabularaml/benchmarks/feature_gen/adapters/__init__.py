"""Framework adapters for the cross-framework FE benchmark.

Each adapter wraps a third-party (or in-house) feature-engineering framework
behind a common FEFrameworkAdapter contract: fit_transform(X_train, y_train)
and transform(X_test) must produce DataFrames with identical column sets so
the downstream base learner can be scored on a held-out test set.

The registry is lazy: adapters are imported only when requested by name, so a
user who does not have `autofeat` installed can still run `nofe` + `tabularaml`
without a hard ImportError at benchmark startup.
"""
from __future__ import annotations
from typing import Type

from .base import FEFrameworkAdapter


def _load(name: str) -> Type[FEFrameworkAdapter]:
    if name == "nofe":
        from .nofe_adapter import NoFEAdapter
        return NoFEAdapter
    if name == "tabularaml":
        from .tabularaml_adapter import TabularAMLAdapter
        return TabularAMLAdapter
    if name == "openfe":
        from .openfe_adapter import OpenFEAdapter
        return OpenFEAdapter
    if name == "autofeat":
        from .autofeat_adapter import AutoFeatAdapter
        return AutoFeatAdapter
    if name == "featuretools":
        from .featuretools_adapter import FeaturetoolsAdapter
        return FeaturetoolsAdapter
    raise KeyError(f"Unknown adapter: {name!r}. Known: {sorted(ADAPTER_NAMES)}")


ADAPTER_NAMES = {"nofe", "tabularaml", "openfe", "autofeat", "featuretools"}


def get_adapter_cls(name: str) -> Type[FEFrameworkAdapter]:
    return _load(name)


__all__ = ["FEFrameworkAdapter", "ADAPTER_NAMES", "get_adapter_cls"]
