"""
Tests for the 5 image quality metrics in AdaptiveOpticsController.

Tests the actual production metric functions (adaptiveOpticsSequencer.py:343-370)
using synthetic images with known sharpness properties.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter


METRIC_NAMES = [
    'laplacian_variance',
    'brenner',
    'normalized_variance',
    'tenengrad',
    'gradient_squared',
]


def _make_point_source(size=128, sigma=1.0):
    """Create synthetic image with a Gaussian point source."""
    img = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    img[center, center] = 1000.0
    return gaussian_filter(img, sigma=sigma)


def _make_edge(size=128):
    """Create a sharp vertical edge image."""
    img = np.zeros((size, size), dtype=np.float64)
    img[:, size // 2:] = 100.0
    return img


@pytest.mark.parametrize("metric_name", METRIC_NAMES)
def test_metric_focused_vs_defocused(controller, metric_name):
    """All metrics should return higher value for sharp image than blurred."""
    controller.selected_metric = metric_name
    metric_fn = controller.get_metric_function()

    sharp = _make_point_source(sigma=1.0)
    blurred = _make_point_source(sigma=10.0)

    val_sharp = metric_fn(sharp)
    val_blurred = metric_fn(blurred)

    assert val_sharp > val_blurred, (
        f"{metric_name}: sharp ({val_sharp:.4f}) should be > blurred ({val_blurred:.4f})"
    )


@pytest.mark.parametrize("metric_name", METRIC_NAMES)
def test_metric_positive_values(controller, metric_name):
    """All metrics should return positive values for non-zero images."""
    controller.selected_metric = metric_name
    metric_fn = controller.get_metric_function()

    img = _make_point_source(sigma=2.0)
    val = metric_fn(img)

    assert val > 0, f"{metric_name}: expected positive value, got {val}"


@pytest.mark.parametrize("metric_name", METRIC_NAMES)
def test_metric_zero_image(controller, metric_name):
    """Metrics should handle zero images without crashing (may return 0)."""
    controller.selected_metric = metric_name
    metric_fn = controller.get_metric_function()

    img = np.zeros((64, 64))
    val = metric_fn(img)

    assert np.isfinite(val), f"{metric_name}: returned non-finite value for zero image"


@pytest.mark.parametrize("metric_name", METRIC_NAMES)
def test_metric_monotonic_with_blur(controller, metric_name):
    """Metrics should generally decrease as blur increases."""
    controller.selected_metric = metric_name
    metric_fn = controller.get_metric_function()

    sigmas = [1.0, 3.0, 5.0, 10.0, 20.0]
    values = [metric_fn(_make_point_source(sigma=s)) for s in sigmas]

    # Check overall trend: first value should be > last value
    assert values[0] > values[-1], (
        f"{metric_name}: expected decreasing trend with blur, "
        f"got values {[f'{v:.4f}' for v in values]} for sigmas {sigmas}"
    )


def test_metric_noise_robustness(controller):
    """Brenner gradient should be more robust to Poisson noise than Laplacian variance.

    This tests the relative noise sensitivity documented in METRICS_DERIVATION.md.
    """
    np.random.seed(42)

    base_img = _make_point_source(sigma=2.0)
    n_trials = 50

    def noisy_metric_variance(metric_name, base):
        controller.selected_metric = metric_name
        metric_fn = controller.get_metric_function()
        values = []
        for _ in range(n_trials):
            # Add Poisson noise (simulates low photon count)
            noisy = np.random.poisson(np.maximum(base * 100, 0.01)) / 100.0
            values.append(metric_fn(noisy))
        return np.std(values) / (np.mean(values) + 1e-12)  # coefficient of variation

    cv_laplacian = noisy_metric_variance('laplacian_variance', base_img)
    cv_brenner = noisy_metric_variance('brenner', base_img)

    # Brenner should have lower coefficient of variation (more robust)
    assert cv_brenner < cv_laplacian, (
        f"Brenner CV ({cv_brenner:.4f}) should be < Laplacian CV ({cv_laplacian:.4f})"
    )
