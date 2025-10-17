"""Utility helpers for working with array-like data."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike


def ensure_1d_array(values: ArrayLike | Iterable[float]) -> np.ndarray:
    """Convert an array-like object into a one-dimensional :class:`numpy.ndarray`.

    Args:
        values: Sequence or array of numeric values.

    Returns:
        A one-dimensional array with :class:`float` dtype.

    Raises:
        ValueError: If the result cannot be converted to a one-dimensional array.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array.")
    return array


def ensure_2d_array(values: ArrayLike | Iterable[Iterable[float]]) -> np.ndarray:
    """Convert an array-like object into a two-dimensional array."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Expected a two-dimensional array.")
    return array


__all__ = ["ensure_1d_array", "ensure_2d_array"]
