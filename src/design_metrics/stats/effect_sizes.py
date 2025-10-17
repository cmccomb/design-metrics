"""Effect size calculations for general statistical analysis.

The functions in this module intentionally accept array-like objects and
return plain :class:`float` results to keep them easy to compose with other
statistical tooling across the project.
"""

from __future__ import annotations

from math import sqrt
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from design_metrics.utils.array import ensure_1d_array

FloatArray: TypeAlias = NDArray[np.floating[Any]]


def _validate_samples(sample_a: FloatArray, sample_b: FloatArray) -> None:
    """Validate the provided samples.

    Args:
        sample_a: A one-dimensional numeric array representing the first sample.
        sample_b: A one-dimensional numeric array representing the second sample.

    Raises:
        ValueError: If either array is empty or contains non-finite values.
    """

    if sample_a.size == 0 or sample_b.size == 0:
        raise ValueError("Samples must contain at least one observation.")
    if not (np.isfinite(sample_a).all() and np.isfinite(sample_b).all()):
        raise ValueError("Samples must only contain finite numeric values.")


def _pooled_standard_deviation(sample_a: FloatArray, sample_b: FloatArray) -> float:
    """Return the pooled standard deviation for two samples."""

    variance_a = np.var(sample_a, ddof=1)
    variance_b = np.var(sample_b, ddof=1)
    pooled_variance = (
        ((sample_a.size - 1) * variance_a) + ((sample_b.size - 1) * variance_b)
    ) / (sample_a.size + sample_b.size - 2)
    return sqrt(pooled_variance)


def cohen_d(sample_a: ArrayLike, sample_b: ArrayLike) -> float:
    """Compute Cohen's *d* effect size between two independent samples.

    Args:
        sample_a: First sample of observations.
        sample_b: Second sample of observations.

    Returns:
        The standardized mean difference between the samples.

    Example:
        >>> cohen_d([1, 2, 3], [4, 5, 6])
        -3.0
    """

    array_a = ensure_1d_array(sample_a)
    array_b = ensure_1d_array(sample_b)
    _validate_samples(array_a, array_b)

    pooled_sd = _pooled_standard_deviation(array_a, array_b)
    if pooled_sd == 0:
        raise ValueError("Pooled standard deviation is zero; effect size undefined.")

    mean_difference = np.mean(array_a) - np.mean(array_b)
    return float(mean_difference / pooled_sd)


def hedges_g(sample_a: ArrayLike, sample_b: ArrayLike) -> float:
    """Compute Hedges' *g*, a small-sample corrected effect size.

    The correction factor uses the approach described by Hedges and Olkin (1985).

    Args:
        sample_a: First sample of observations.
        sample_b: Second sample of observations.

    Returns:
        The bias-corrected standardized mean difference.
    """

    array_a = ensure_1d_array(sample_a)
    array_b = ensure_1d_array(sample_b)
    _validate_samples(array_a, array_b)

    degrees_of_freedom = array_a.size + array_b.size - 2
    if degrees_of_freedom <= 0:
        raise ValueError("At least two observations are required in total.")

    correction = 1 - (3 / (4 * degrees_of_freedom - 1))
    return float(cohen_d(array_a, array_b) * correction)


__all__ = ["cohen_d", "hedges_g"]
