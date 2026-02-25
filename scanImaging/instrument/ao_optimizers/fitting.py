"""
Fitting Functions for Scan-Based AO Optimization

Standalone mathematical functions for finding optimal parameters from
scan data (image stacks + parameter values). Extracted from
AdaptiveOpticsController to allow use by modular optimizers without
controller dependency.

Two methods:
- Simple interpolation: 3-point quadratic around maximum
- Weighted fit: All-points polynomial with Poisson weighting
"""

import numpy as np


def compute_optimal_simple_interpolation(image_stack, parameter_stack, metric_fn,
                                         verbose=False, plot=False):
    """
    Find optimal parameter using 3-point quadratic interpolation around maximum.

    Takes the three points centered on the maximum metric value and fits a
    parabola to find the vertex. Supports non-equally-spaced parameter values.

    Parameters
    ----------
    image_stack : list of np.ndarray
        Images acquired at each scan point
    parameter_stack : list of float
        Parameter values corresponding to each image
    metric_fn : callable
        Function to compute image quality metric (higher = better)
    verbose : bool
        Print diagnostic information
    plot : bool
        Save diagnostic plot to file

    Returns
    -------
    optimal_parameter : float
        Estimated optimal parameter value
    miss_max : int
        -1 if optimum below scan range, +1 if above, 0 if within
    best_metric : float
        Metric value at the best measured point
    metric_values : list of float
        Computed metric values for all images
    """
    if len(image_stack) != len(parameter_stack) or len(image_stack) < 3:
        raise ValueError("Image stack and parameter stack must have the same length (>= 3).")

    # Compute metric for each image
    metric_values = [metric_fn(img) for img in image_stack]
    if verbose:
        print("Metric values during scan:", metric_values)

    # Find maximum and check edge cases
    max_index = np.argmax(metric_values)
    miss_max = 0
    if max_index == 0:
        miss_max = -1
    if max_index == len(metric_values) - 1:
        miss_max = 1
        max_index -= 1  # shift to allow interpolation

    # Diagnostic plot
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(parameter_stack, metric_values, 'o-')
        plt.xlabel('Parameter value')
        plt.ylabel('Metric value')
        plt.title('Optimization Metric vs Parameter')
        plt.grid(True)
        plt.savefig('optimization_metric_plot.png')
        plt.close()

    # 3-point quadratic interpolation (general formula for non-equally spaced points)
    x0, x1, x2 = parameter_stack[max_index - 1], parameter_stack[max_index], parameter_stack[max_index + 1]
    y0, y1, y2 = metric_values[max_index - 1], metric_values[max_index], metric_values[max_index + 1]

    yd1 = (y1 - y0)
    yd2 = (y2 - y1)
    xd1 = (x1 - x0)
    xd2 = (x2 - x1)
    denom = yd1 * xd2 - yd2 * xd1

    if denom == 0:
        optimal_parameter = x1
    else:
        optimal_parameter = x1 + 0.5 * (yd1 * xd2 * xd2 + yd2 * xd1 * xd1) / denom

    return optimal_parameter, miss_max, metric_values[max_index], metric_values


def compute_optimal_weighted_fit(image_stack, parameter_stack, metric_fn,
                                 verbose=False, plot=False):
    """
    Find optimal parameter using all-points weighted polynomial fit.

    More robust to noise than 3-point interpolation by using all measured data.
    Poisson weighting: w proportional to sqrt(metric) (since variance proportional to mean
    for Poisson noise).

    Parameters
    ----------
    image_stack : list of np.ndarray
        All acquired images during parameter scan
    parameter_stack : array-like
        Corresponding parameter values
    metric_fn : callable
        Function to compute image quality metric (higher = better)
    verbose : bool
        Print diagnostic information
    plot : bool
        Save diagnostic plot to file

    Returns
    -------
    optimal_parameter : float
        Estimated optimal parameter value
    miss_max : int
        -1 if optimum below scan range, +1 if above, 0 if within
    optimal_metric : float
        Predicted metric value at optimum (or best measured if parabola invalid)
    fit_coeffs : tuple of float or None
        (a, b, c) polynomial coefficients if fit succeeded, None otherwise
    metric_values : np.ndarray
        Computed metric values for all images
    """
    if len(image_stack) == 0 or len(parameter_stack) == 0:
        raise ValueError("Image stack and parameter stack must have non-zero length.")
    if len(image_stack) != len(parameter_stack):
        raise ValueError("Image stack and parameter stack must have the same length.")

    # Compute metric for each image
    metric_values = np.array([metric_fn(img) for img in image_stack])
    if verbose:
        print(f"Metric values (weighted fit): {metric_values}")

    # Poisson weighting: weight proportional to sqrt(metric)
    weights = np.sqrt(np.maximum(metric_values, 1e-12))

    # Fit quadratic to ALL points with weights
    try:
        coeffs = np.polyfit(parameter_stack, metric_values, deg=2, w=weights)
        a, b, c = coeffs
    except Exception as e:
        if verbose:
            print(f"Polynomial fit failed: {e}, falling back to best measured point")
        max_idx = np.argmax(metric_values)
        return parameter_stack[max_idx], 0, metric_values[max_idx], None, metric_values

    # Check if parabola opens downward (concave) - required for maximum
    if a >= 0:
        if verbose:
            print(f"Parabola opens upward (a={a:.2e}), using best measured point")
        max_idx = np.argmax(metric_values)
        return parameter_stack[max_idx], 0, metric_values[max_idx], None, metric_values

    # Compute vertex of parabola: x = -b / (2a)
    optimal_parameter = -b / (2 * a)
    fit_coeffs = (a, b, c)

    # Check if optimum is within scan range
    miss_max = 0
    min_param = np.min(parameter_stack)
    max_param = np.max(parameter_stack)

    if optimal_parameter < min_param:
        miss_max = -1
        if verbose:
            print(f"Optimum below scan range: {optimal_parameter:.2f} < {min_param:.2f}")
    elif optimal_parameter > max_param:
        miss_max = 1
        if verbose:
            print(f"Optimum above scan range: {optimal_parameter:.2f} > {max_param:.2f}")

    # Predicted optimal metric value at vertex
    optimal_metric = a * optimal_parameter**2 + b * optimal_parameter + c

    # Diagnostic plot if requested
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))

        # Measured points
        plt.scatter(parameter_stack, metric_values, c='blue', s=100,
                    alpha=0.6, label='Measured')

        # Fitted parabola
        param_fine = np.linspace(min_param, max_param, 200)
        metric_fine = a * param_fine**2 + b * param_fine + c
        plt.plot(param_fine, metric_fine, 'r--', linewidth=2,
                 label='Weighted fit')

        # Optimal point
        plt.scatter([optimal_parameter], [optimal_metric], c='red', s=200,
                    marker='*', edgecolors='black', linewidth=2,
                    label=f'Optimum: {optimal_parameter:.2f}')

        plt.xlabel('Parameter value', fontsize=12)
        plt.ylabel('Metric value', fontsize=12)
        plt.title('All-Points Weighted Fit (Poisson weighting)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.savefig('optimization_weighted_fit.png', dpi=150, bbox_inches='tight')
        plt.close()

    return optimal_parameter, miss_max, optimal_metric, fit_coeffs, metric_values
