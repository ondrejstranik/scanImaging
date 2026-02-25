"""
Image Quality Metrics for Adaptive Optics Optimization

Standalone module containing all image quality metrics used by AO optimizers.
Each metric takes an image array and returns a scalar quality value (higher = better focus).

See METRICS_DERIVATION.md for mathematical derivations and noise sensitivity analysis.
"""

import numpy as np


def metric_laplacian_variance(img):
    """Laplacian variance - sensitive to focus, but noise-sensitive."""
    from scipy.ndimage import laplace
    return np.mean(laplace(img)**2) / (np.mean(img) + 1e-12)


def metric_brenner(img):
    """Brenner gradient - more robust to Poisson noise."""
    grad_v = np.sum((img[2:, :] - img[:-2, :])**2)
    grad_h = np.sum((img[:, 2:] - img[:, :-2])**2)
    return (grad_v + grad_h) / (np.sum(img) + 1e-12)


def metric_normalized_variance(img):
    """Normalized variance - simple, less sensitive to focus but robust."""
    return np.var(img) / (np.mean(img) + 1e-12)


def metric_tenengrad(img):
    """Tenengrad (Sobel-based) - standard autofocus metric."""
    from scipy.ndimage import sobel
    grad_x = sobel(img, axis=1)
    grad_y = sobel(img, axis=0)
    return np.mean(grad_x**2 + grad_y**2)


def metric_gradient_squared(img):
    """Sum of squared gradients - simpler than Tenengrad."""
    grad_v = np.sum(np.diff(img, axis=0)**2)
    grad_h = np.sum(np.diff(img, axis=1)**2)
    return (grad_v + grad_h) / (np.sum(img) + 1e-12)


METRIC_FUNCTIONS = {
    'laplacian_variance': metric_laplacian_variance,
    'brenner': metric_brenner,
    'normalized_variance': metric_normalized_variance,
    'tenengrad': metric_tenengrad,
    'gradient_squared': metric_gradient_squared,
}


def get_metric_function(name):
    """Get metric function by name.

    Args:
        name: Metric name (one of: laplacian_variance, brenner,
              normalized_variance, tenengrad, gradient_squared)

    Returns:
        callable: Function(image) -> float
    """
    if name not in METRIC_FUNCTIONS:
        raise ValueError(f"Unknown metric '{name}'. Available: {list(METRIC_FUNCTIONS.keys())}")
    return METRIC_FUNCTIONS[name]
