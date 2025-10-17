"""Tests for namespace package behavior."""

import types


def test_design_metrics_namespace_package() -> None:
    import design_metrics

    assert isinstance(design_metrics, types.ModuleType)
    assert design_metrics.__spec__ is not None
    assert design_metrics.__spec__.submodule_search_locations is not None
    assert not hasattr(design_metrics, "cohen_d")
