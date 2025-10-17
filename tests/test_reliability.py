import numpy as np
import pandas as pd
import pytest

from design_metrics.hsr.reliability import cronbach_alpha


def test_cronbach_alpha_dataframe() -> None:
    responses = pd.DataFrame(
        {
            "item1": [1, 2, 3, 4],
            "item2": [1, 2, 3, 4],
            "item3": [2, 3, 4, 5],
        }
    )

    result = cronbach_alpha(responses)

    assert pytest.approx(result, abs=1e-6) == 1.0


def test_cronbach_alpha_raises_for_single_item() -> None:
    responses = np.array([[1], [2], [3]])

    with pytest.raises(ValueError):
        cronbach_alpha(responses)
