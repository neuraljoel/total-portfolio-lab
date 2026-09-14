"""Proves the package is installed and importable.

Not a placeholder. This fails if the editable install broke, if a subpackage is
missing from pyproject.toml, or if the wrong interpreter is active — all three
are real, and all three are otherwise silent until something confusing happens
much later.
"""

from __future__ import annotations

import tplab
import tplab.risk
import tplab.stats
import tplab.vol


def test_package_is_installed() -> None:
    assert tplab.__version__ != "0.0.0.dev0", (
        'package not installed — run: pip install -e ".[dev]"'
    )


def test_version_is_wellformed() -> None:
    parts = tplab.__version__.split(".")
    assert len(parts) >= 2
    assert parts[0].isdigit() and parts[1].isdigit()


def test_subpackages_are_importable() -> None:
    """Catches a subpackage that exists on disk but was never packaged."""
    for module in (tplab.stats, tplab.vol, tplab.risk):
        assert module.__doc__, f"{module.__name__} has no docstring"
