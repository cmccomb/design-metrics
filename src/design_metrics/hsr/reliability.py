"""Reliability metrics for scale construction and survey research."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from design_metrics.utils.array import ensure_2d_array


def cronbach_alpha(responses: pd.DataFrame | np.ndarray) -> float:
    """Estimate Cronbach's alpha for a set of Likert-style items.

    Args:
        responses: Two-dimensional matrix where rows correspond to participants
            and columns represent items.

    Returns:
        The Cronbach's alpha reliability estimate.

    Raises:
        ValueError: If fewer than two items are provided or there is no variance.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"q1": [1, 2, 3], "q2": [1, 2, 4], "q3": [2, 3, 4]})
        >>> round(cronbach_alpha(df), 3)
        0.864
    """

    matrix = ensure_2d_array(responses)
    if matrix.shape[1] < 2:
        raise ValueError("Cronbach's alpha requires at least two items.")

    item_variances = matrix.var(axis=0, ddof=1)
    if np.isclose(item_variances.sum(), 0):
        raise ValueError("Items exhibit zero variance; alpha is undefined.")

    total_scores = matrix.sum(axis=1)
    total_variance = total_scores.var(ddof=1)
    item_count = matrix.shape[1]

    return float(
        (item_count / (item_count - 1))
        * (1 - (item_variances.sum() / total_variance))
    )


__all__ = ["cronbach_alpha"]
