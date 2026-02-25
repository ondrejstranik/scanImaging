"""
Module import and backward compatibility tests.

Verifies that all optimizer modules, the controller, and backward-compatible
aliases can be imported correctly.
"""

import pytest


def test_optimizer_module_import():
    """Test that the ao_optimizers module can be imported."""
    from scanImaging.instrument.ao_optimizers import BaseOptimizer, SequentialOptimizer
    assert BaseOptimizer is not None
    assert SequentialOptimizer is not None


def test_all_optimizer_imports():
    """Test that all 4 optimizer classes can be imported."""
    from scanImaging.instrument.ao_optimizers import (
        BaseOptimizer,
        SequentialOptimizer,
        SPGDOptimizer,
        RandomSearchOptimizer,
    )
    assert BaseOptimizer is not None
    assert SequentialOptimizer is not None
    assert SPGDOptimizer is not None
    assert RandomSearchOptimizer is not None


def test_controller_import():
    """Test that AdaptiveOpticsController imports correctly."""
    from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsController
    assert AdaptiveOpticsController is not None


def test_backward_compatibility_alias():
    """Test that old name AdaptiveOpticsSequencer still works as alias."""
    from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsSequencer
    from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsController
    assert AdaptiveOpticsSequencer is AdaptiveOpticsController


def test_controller_instantiation(controller):
    """Test that controller can be instantiated with virtual devices."""
    assert controller is not None
    assert controller.deformable_mirror is not None
    assert controller.image_provider is not None
    assert controller.verbose is False
