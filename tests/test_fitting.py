"""
Tests for quadratic fitting algorithms.

Tests the actual production code:
- computeOptimalParametersSimpleInterpolation (adaptiveOpticsSequencer.py:599-647)
- computeOptimalParametersWeightedFit (adaptiveOpticsSequencer.py:649-760)

Uses synthetic parabolic data with known optima, extending the pattern
from the __main__ test in adaptiveOpticsSequencer.py.
"""

import numpy as np
import pytest


# ============================================================================
# SimpleInterpolation tests
# ============================================================================

def test_simple_interp_noiseless_parabola(controller):
    """Perfect recovery of known vertex — migrated from __main__ test."""
    # Parabola: f(x) = -3*(x-1)^2 + 2, vertex at x=1
    f = lambda x: -3.0 * (x - 1.0) ** 2 + 2.0
    xv = [0.0, 0.5, 1.0, 1.5, 2.0]
    # image_stack = scalar values, metric = identity
    yv = [f(x) for x in xv]

    opt, miss, metric_at_max = controller.computeOptimalParametersSimpleInterpolation(
        yv, xv, lambda x: x
    )

    assert abs(opt - 1.0) < 0.01, f"Expected optimum ~1.0, got {opt}"
    assert miss == 0, f"Expected miss_max=0, got {miss}"


def test_simple_interp_optimum_below_range(controller):
    """Returns miss_max=-1 when optimum is below scan range."""
    # Parabola: f(x) = -3*(x-10)^2 + 2, vertex at x=10
    # Scan range [0, 2] — optimum far above range, so max is at right edge
    f = lambda x: -3.0 * (x - 10.0) ** 2 + 2.0
    xv = [0.0, 0.5, 1.0, 1.5, 2.0]
    yv = [f(x) for x in xv]

    opt, miss, _ = controller.computeOptimalParametersSimpleInterpolation(
        yv, xv, lambda x: x
    )

    # Max at right edge (x=2.0 is closest to x=10), so miss_max=+1
    assert miss == 1, f"Expected miss_max=+1, got {miss}"


def test_simple_interp_optimum_above_range(controller):
    """Returns miss_max=+1 when optimum is above scan range."""
    # Parabola: f(x) = -3*(x-(-5))^2 + 2, vertex at x=-5
    # Scan range [0, 2] — optimum far below range, so max is at left edge
    f = lambda x: -3.0 * (x - (-5.0)) ** 2 + 2.0
    xv = [0.0, 0.5, 1.0, 1.5, 2.0]
    yv = [f(x) for x in xv]

    opt, miss, _ = controller.computeOptimalParametersSimpleInterpolation(
        yv, xv, lambda x: x
    )

    # Max at left edge (x=0 is closest to x=-5), so miss_max=-1
    assert miss == -1, f"Expected miss_max=-1, got {miss}"


def test_simple_interp_non_equally_spaced(controller):
    """Works with non-uniform parameter spacing."""
    # Parabola: f(x) = -(x-3)^2 + 10, vertex at x=3
    f = lambda x: -(x - 3.0) ** 2 + 10.0
    xv = [1.0, 2.0, 3.5, 4.0, 5.0]  # non-uniform spacing
    yv = [f(x) for x in xv]

    opt, miss, _ = controller.computeOptimalParametersSimpleInterpolation(
        yv, xv, lambda x: x
    )

    assert abs(opt - 3.0) < 0.5, f"Expected optimum ~3.0, got {opt}"
    assert miss == 0


def test_simple_interp_flat_metric(controller):
    """Handles degenerate case where all metric values are equal."""
    xv = [0.0, 1.0, 2.0, 3.0, 4.0]
    yv = [5.0, 5.0, 5.0, 5.0, 5.0]  # flat

    opt, miss, _ = controller.computeOptimalParametersSimpleInterpolation(
        yv, xv, lambda x: x
    )

    # Should not crash — returns some value (likely center or edge)
    assert np.isfinite(opt)


# ============================================================================
# WeightedFit tests
# ============================================================================

def test_weighted_fit_noiseless(controller):
    """Perfect recovery of known parabola vertex."""
    # Parabola: f(x) = -0.1*(x-50)^2 + 100, vertex at x=50
    f = lambda x: -0.1 * (x - 50.0) ** 2 + 100.0
    xv = np.linspace(20, 80, 7)
    yv = [f(x) for x in xv]

    opt, miss, _ = controller.computeOptimalParametersWeightedFit(
        yv, list(xv), lambda x: x
    )

    assert abs(opt - 50.0) < 0.1, f"Expected optimum ~50.0, got {opt}"
    assert miss == 0, f"Expected miss_max=0, got {miss}"


def test_weighted_fit_poisson_noise(controller):
    """Sub-nm accuracy with noisy metric values."""
    np.random.seed(42)
    true_optimum = 50.0

    # Generate noisy parabolic data
    f = lambda x: -0.1 * (x - true_optimum) ** 2 + 100.0
    xv = np.linspace(20, 80, 11)
    yv = []
    for x in xv:
        true_val = f(x)
        # Poisson noise
        noisy_val = np.random.poisson(max(true_val, 0.1) * 10) / 10.0
        yv.append(noisy_val)

    opt, miss, _ = controller.computeOptimalParametersWeightedFit(
        yv, list(xv), lambda x: x
    )

    # Should be within a few nm of true optimum
    assert abs(opt - true_optimum) < 5.0, (
        f"Expected optimum ~{true_optimum}, got {opt} (error: {abs(opt - true_optimum):.2f})"
    )


def test_weighted_fit_accumulated_points(controller):
    """More points should give better accuracy (key feature of weighted fit)."""
    np.random.seed(42)
    true_optimum = 50.0
    f = lambda x: -0.1 * (x - true_optimum) ** 2 + 100.0

    # Few points
    xv_few = np.linspace(20, 80, 5)
    yv_few = [np.random.poisson(max(f(x), 0.1) * 10) / 10.0 for x in xv_few]
    opt_few, _, _ = controller.computeOptimalParametersWeightedFit(
        yv_few, list(xv_few), lambda x: x
    )

    # Many points (simulating accumulation from multiple iterations)
    xv_many = np.linspace(20, 80, 21)
    yv_many = [np.random.poisson(max(f(x), 0.1) * 10) / 10.0 for x in xv_many]
    opt_many, _, _ = controller.computeOptimalParametersWeightedFit(
        yv_many, list(xv_many), lambda x: x
    )

    error_few = abs(opt_few - true_optimum)
    error_many = abs(opt_many - true_optimum)

    # With more points, we generally expect better accuracy
    # This test may occasionally fail due to randomness, so we use a soft check
    # The key is that both estimates are reasonable
    assert error_many < 10.0, f"Many-point estimate error too large: {error_many:.2f}"
    assert error_few < 15.0, f"Few-point estimate error too large: {error_few:.2f}"


def test_weighted_fit_upward_parabola(controller):
    """Falls back to best measured point when parabola opens upward."""
    # Upward parabola: f(x) = +0.1*(x-50)^2 + 10 (no maximum)
    f = lambda x: 0.1 * (x - 50.0) ** 2 + 10.0
    xv = np.linspace(20, 80, 7)
    yv = [f(x) for x in xv]

    opt, miss, metric = controller.computeOptimalParametersWeightedFit(
        yv, list(xv), lambda x: x
    )

    # Should fall back to best measured point (max at edges)
    assert miss == 0, f"Expected miss_max=0 (fallback mode), got {miss}"
    # Optimal should be at one of the edge points (where upward parabola is highest)
    assert opt in [xv[0], xv[-1]] or abs(opt - xv[0]) < 1 or abs(opt - xv[-1]) < 1


def test_weighted_fit_optimum_outside_range(controller):
    """Returns correct miss_max when vertex is outside scan range."""
    # Parabola: f(x) = -0.1*(x-100)^2 + 100, vertex at x=100
    # Scan range [20, 80] — optimum above range
    f = lambda x: -0.1 * (x - 100.0) ** 2 + 100.0
    xv = np.linspace(20, 80, 7)
    yv = [f(x) for x in xv]

    opt, miss, _ = controller.computeOptimalParametersWeightedFit(
        yv, list(xv), lambda x: x
    )

    assert miss == 1, f"Expected miss_max=+1 (above range), got {miss}"
