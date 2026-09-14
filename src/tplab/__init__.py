"""Total Portfolio Lab — institutional risk, ALM and allocation analytics.

Subpackages are organised by what the mathematics is, not by what uses it.
Nothing in this package imports from outside it.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tplab")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
