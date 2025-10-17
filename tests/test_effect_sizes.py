import numpy as np
import pytest

from design_metrics.stats.effect_sizes import cohen_d, hedges_g


def test_cohen_d_basic_case() -> None:
    sample_a = np.array([1.0, 2.0, 3.0])
    sample_b = np.array([2.0, 3.0, 4.0])

    result = cohen_d(sample_a, sample_b)

    assert pytest.approx(result, abs=1e-6) == -1.0


def test_hedges_g_bias_correction() -> None:
    sample_a = np.array([1.0, 2.0, 3.0, 4.0])
    sample_b = np.array([2.0, 3.0, 4.0, 5.0])

    result = hedges_g(sample_a, sample_b)

    df = sample_a.size + sample_b.size - 2
    correction = 1 - (3 / (4 * df - 1))
    assert pytest.approx(result, rel=1e-6) == cohen_d(sample_a, sample_b) * correction


def test_cohen_d_raises_on_zero_variance() -> None:
    with pytest.raises(ValueError):
        cohen_d([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])


def test_effect_sizes_available_from_hsr_namespace() -> None:
    import design_metrics.hsr as hsr

    assert hsr.cohen_d is cohen_d
    assert hsr.hedges_g is hedges_g
