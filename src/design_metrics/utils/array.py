"""Utility helpers for working with array-like data."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.floating[Any]]


def ensure_1d_array(values: ArrayLike | Iterable[float]) -> FloatArray:
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
    return cast(FloatArray, array)


def ensure_2d_array(values: ArrayLike | Iterable[Iterable[float]]) -> FloatArray:
    """Convert an array-like object into a two-dimensional array."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Expected a two-dimensional array.")
    return cast(FloatArray, array)


__all__ = ["FloatArray", "ensure_1d_array", "ensure_2d_array"]
