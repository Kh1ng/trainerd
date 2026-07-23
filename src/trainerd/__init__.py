"""Reusable training orchestration helper app."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("trainerd")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.3.0"
