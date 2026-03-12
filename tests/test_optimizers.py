"""
Tests for optimizer convergence and loop routing.

Tests the actual production optimizer classes:
- SPGDOptimizer (ao_optimizers/spgd_optimizer.py)
- RandomSearchOptimizer (ao_optimizers/random_search_optimizer.py)
- SequentialOptimizer (ao_optimizers/sequential_optimizer.py)

Uses MetricImageProvider (from conftest.py) to create synthetic ground truth
where image quality depends on DM Zernike coefficients.
"""

import numpy as np
import pytest
import time

# Patch time.sleep to speed up tests
import unittest.mock


def _run_generator(gen):
    """Exhaust a generator, collecting all yielded/returned values."""
    results = []
    try:
        while True:
            val = next(gen)
            if val is not None:
                results.append(val)
    except StopIteration:
        pass
    return results


# ============================================================================
# SPGD Optimizer Tests
# ============================================================================

@unittest.mock.patch('time.sleep')
def test_spgd_convergence(mock_sleep, virtual_dm):
    """SPGD should converge toward known optimum on synthetic quadratic."""
    from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsController
    from scanImaging.instrument.ao_optimizers import SPGDOptimizer
    from conftest import MetricImageProvider

    np.random.seed(42)

    # Setup controller with MetricImageProvider
    # Optimum: coefficient 4 = 20 nm
    zernike_indices = [4]
    optimum_values = [20.0]

    provider = MetricImageProvider(virtual_dm, optimum_values, zernike_indices, scale=500.0, noise=0.01)

    ctrl = AdaptiveOpticsController()
    ctrl.deformable_mirror = virtual_dm
    ctrl.image_provider = provider
    ctrl.verbose = False
    ctrl.selected_metric = 'normalized_variance'
    ctrl.spgd_gain = 0.3
    ctrl.spgd_delta = 5.0
    ctrl.spgd_iterations = 30
    ctrl.continuous_scan = False
    ctrl.enable_convergence_detection = False

    # Initialize optimizer
    initial_coeffs = np.zeros(12)
    optimizer = SPGDOptimizer(ctrl)
    optimizer.initialize(initial_coefficients=initial_coeffs, zernike_indices=zernike_indices)

    # Run
    _run_generator(optimizer.run())

    # Check convergence: final coefficient should be closer to optimum than initial
    final_coeffs = optimizer.final_coefficients
    initial_error = abs(0.0 - optimum_values[0])
    final_error = abs(final_coeffs[4] - optimum_values[0])

    assert final_error < initial_error, (
        f"SPGD did not converge: initial_error={initial_error:.1f}, final_error={final_error:.1f}"
    )


@unittest.mock.patch('time.sleep')
def test_spgd_convergence_detection(mock_sleep, virtual_dm):
    """SPGD convergence detection should stop early when improvement is small."""
    from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsController
    from scanImaging.instrument.ao_optimizers import SPGDOptimizer
    from conftest import MetricImageProvider

    np.random.seed(42)

    # Start close to optimum so SPGD converges quickly
    zernike_indices = [4]
    optimum_values = [5.0]  # Close to initial=0

    provider = MetricImageProvider(virtual_dm, optimum_values, zernike_indices, scale=500.0, noise=0.001)

    ctrl = AdaptiveOpticsController()
    ctrl.deformable_mirror = virtual_dm
    ctrl.image_provider = provider
    ctrl.verbose = False
    ctrl.selected_metric = 'normalized_variance'
    ctrl.spgd_gain = 0.3
    ctrl.spgd_delta = 5.0
    ctrl.spgd_iterations = 200  # High limit
    ctrl.continuous_scan = False
    ctrl.enable_convergence_detection = True
    ctrl.convergence_threshold = 0.01  # 1% improvement threshold
    ctrl.convergence_window = 5

    initial_coeffs = np.zeros(12)
    optimizer = SPGDOptimizer(ctrl)
    optimizer.initialize(initial_coefficients=initial_coeffs, zernike_indices=zernike_indices)

    _run_generator(optimizer.run())

    # Convergence detection sets iteration = max_iterations to force exit,
    # so check actual number of steps via metric_history length
    actual_steps = len(optimizer.metric_history)
    assert actual_steps < 200, (
        f"Convergence detection failed: ran {actual_steps} steps (max=200)"
    )


@unittest.mock.patch('time.sleep')
def test_spgd_stop_requested(mock_sleep, virtual_dm):
    """_stop_requested flag should interrupt SPGD cleanly."""
    from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsController
    from scanImaging.instrument.ao_optimizers import SPGDOptimizer
    from conftest import MetricImageProvider

    np.random.seed(42)

    zernike_indices = [4]
    optimum_values = [20.0]
    provider = MetricImageProvider(virtual_dm, optimum_values, zernike_indices)

    ctrl = AdaptiveOpticsController()
    ctrl.deformable_mirror = virtual_dm
    ctrl.image_provider = provider
    ctrl.verbose = False
    ctrl.selected_metric = 'normalized_variance'
    ctrl.spgd_gain = 0.1
    ctrl.spgd_delta = 5.0
    ctrl.spgd_iterations = 100
    ctrl.continuous_scan = False
    ctrl.enable_convergence_detection = False

    initial_coeffs = np.zeros(12)
    optimizer = SPGDOptimizer(ctrl)
    optimizer.initialize(initial_coefficients=initial_coeffs, zernike_indices=zernike_indices)

    # Run a few steps then request stop
    gen = optimizer.run()
    steps = 0
    for _ in gen:
        steps += 1
        if steps >= 5:
            ctrl._stop_requested = True

    # Should have stopped early
    assert optimizer.iteration < 100, f"Stop not honored: ran {optimizer.iteration} iterations"


# ============================================================================
# Random Search Optimizer Tests
# ============================================================================

@unittest.mock.patch('time.sleep')
def test_random_search_improves_metric(mock_sleep, virtual_dm):
    """Random search should improve metric from initial state."""
    from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsController
    from scanImaging.instrument.ao_optimizers import RandomSearchOptimizer
    from conftest import MetricImageProvider

    np.random.seed(42)

    zernike_indices = [4]
    optimum_values = [30.0]
    provider = MetricImageProvider(virtual_dm, optimum_values, zernike_indices, scale=500.0, noise=0.01)

    ctrl = AdaptiveOpticsController()
    ctrl.deformable_mirror = virtual_dm
    ctrl.image_provider = provider
    ctrl.verbose = False
    ctrl.selected_metric = 'normalized_variance'
    ctrl.random_search_iterations = 50
    ctrl.random_search_range = 100.0
    ctrl.continuous_scan = False

    initial_coeffs = np.zeros(12)
    optimizer = RandomSearchOptimizer(ctrl)
    optimizer.initialize(initial_coefficients=initial_coeffs, zernike_indices=zernike_indices)

    _run_generator(optimizer.run())

    # Best metric should be better than initial
    assert optimizer.best_metric >= optimizer.metric_history[0], (
        f"Random search did not improve: initial={optimizer.metric_history[0]:.4f}, "
        f"best={optimizer.best_metric:.4f}"
    )


@unittest.mock.patch('time.sleep')
def test_random_search_stop_requested(mock_sleep, virtual_dm):
    """_stop_requested flag should interrupt random search cleanly."""
    from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsController
    from scanImaging.instrument.ao_optimizers import RandomSearchOptimizer
    from conftest import MetricImageProvider

    np.random.seed(42)

    zernike_indices = [4]
    optimum_values = [20.0]
    provider = MetricImageProvider(virtual_dm, optimum_values, zernike_indices)

    ctrl = AdaptiveOpticsController()
    ctrl.deformable_mirror = virtual_dm
    ctrl.image_provider = provider
    ctrl.verbose = False
    ctrl.selected_metric = 'normalized_variance'
    ctrl.random_search_iterations = 100
    ctrl.random_search_range = 100.0
    ctrl.continuous_scan = False

    initial_coeffs = np.zeros(12)
    optimizer = RandomSearchOptimizer(ctrl)
    optimizer.initialize(initial_coefficients=initial_coeffs, zernike_indices=zernike_indices)

    gen = optimizer.run()
    steps = 0
    for _ in gen:
        steps += 1
        if steps >= 5:
            ctrl._stop_requested = True

    assert optimizer.iteration < 100, f"Stop not honored: ran {optimizer.iteration} iterations"


# ============================================================================
# Sequential Optimizer Tests
# ============================================================================

@unittest.mock.patch('time.sleep')
def test_sequential_simple_interp(mock_sleep, virtual_dm):
    """Sequential optimizer should find optimum for single mode with simple_interpolation."""
    from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsController
    from scanImaging.instrument.ao_optimizers import SequentialOptimizer
    from conftest import MetricImageProvider

    np.random.seed(42)

    zernike_indices = [4]
    optimum_values = [15.0]
    provider = MetricImageProvider(virtual_dm, optimum_values, zernike_indices, scale=500.0, noise=0.01)

    ctrl = AdaptiveOpticsController()
    ctrl.deformable_mirror = virtual_dm
    ctrl.image_provider = provider
    ctrl.verbose = False
    ctrl.print_plot = False
    ctrl.selected_metric = 'normalized_variance'
    ctrl.optim_method = 'simple_interpolation'
    ctrl.optim_iterations = 3
    ctrl.num_steps_per_mode = 5
    ctrl.continuous_scan = False
    ctrl.enable_convergence_detection = False
    ctrl.image_log = False
    ctrl.optim_iterations_per_mode = []

    initial_coeffs = np.zeros(12)
    optimizer = SequentialOptimizer(ctrl)
    optimizer.initialize(
        initial_coefficients=initial_coeffs,
        zernike_indices=zernike_indices,
        scan_amplitudes=[80]
    )

    _run_generator(optimizer.run())

    # Check that final coefficients moved toward optimum
    final = optimizer.final_coefficients
    initial_error = abs(0.0 - optimum_values[0])
    final_error = abs(final[4] - optimum_values[0])

    assert final_error < initial_error, (
        f"Sequential did not converge: initial_error={initial_error:.1f}, final_error={final_error:.1f}"
    )


@unittest.mock.patch('time.sleep')
def test_sequential_weighted_fit(mock_sleep, virtual_dm):
    """Sequential optimizer should find optimum with weighted_fit method."""
    from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsController
    from scanImaging.instrument.ao_optimizers import SequentialOptimizer
    from conftest import MetricImageProvider

    np.random.seed(42)

    zernike_indices = [4]
    optimum_values = [15.0]
    provider = MetricImageProvider(virtual_dm, optimum_values, zernike_indices, scale=500.0, noise=0.01)

    ctrl = AdaptiveOpticsController()
    ctrl.deformable_mirror = virtual_dm
    ctrl.image_provider = provider
    ctrl.verbose = False
    ctrl.print_plot = False
    ctrl.selected_metric = 'normalized_variance'
    ctrl.optim_method = 'weighted_fit'
    ctrl.optim_iterations = 3
    ctrl.num_steps_per_mode = 5
    ctrl.continuous_scan = False
    ctrl.enable_convergence_detection = False
    ctrl.image_log = False
    ctrl.optim_iterations_per_mode = []

    initial_coeffs = np.zeros(12)
    optimizer = SequentialOptimizer(ctrl)
    optimizer.initialize(
        initial_coefficients=initial_coeffs,
        zernike_indices=zernike_indices,
        scan_amplitudes=[80]
    )

    _run_generator(optimizer.run())

    final = optimizer.final_coefficients
    initial_error = abs(0.0 - optimum_values[0])
    final_error = abs(final[4] - optimum_values[0])

    assert final_error < initial_error, (
        f"Sequential (weighted_fit) did not converge: initial_error={initial_error:.1f}, "
        f"final_error={final_error:.1f}"
    )


@unittest.mock.patch('time.sleep')
def test_sequential_multi_mode(mock_sleep, virtual_dm):
    """Sequential optimizer should optimize multiple modes."""
    from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsController
    from scanImaging.instrument.ao_optimizers import SequentialOptimizer
    from conftest import MetricImageProvider

    np.random.seed(42)

    zernike_indices = [4, 5]
    optimum_values = [10.0, -15.0]
    provider = MetricImageProvider(virtual_dm, optimum_values, zernike_indices, scale=1000.0, noise=0.01)

    ctrl = AdaptiveOpticsController()
    ctrl.deformable_mirror = virtual_dm
    ctrl.image_provider = provider
    ctrl.verbose = False
    ctrl.print_plot = False
    ctrl.selected_metric = 'normalized_variance'
    ctrl.optim_method = 'simple_interpolation'
    ctrl.optim_iterations = 2
    ctrl.num_steps_per_mode = 5
    ctrl.continuous_scan = False
    ctrl.enable_convergence_detection = False
    ctrl.image_log = False
    ctrl.optim_iterations_per_mode = []

    initial_coeffs = np.zeros(12)
    optimizer = SequentialOptimizer(ctrl)
    optimizer.initialize(
        initial_coefficients=initial_coeffs,
        zernike_indices=zernike_indices,
        scan_amplitudes=[80, 80]
    )

    _run_generator(optimizer.run())

    # Both modes should have been processed
    assert optimizer.is_complete(), "Sequential optimizer did not complete all modes"
    assert optimizer.final_coefficients is not None


# ============================================================================
# Loop Routing Tests
# ============================================================================

@unittest.mock.patch('time.sleep')
def test_loop_routing_spgd(mock_sleep, virtual_dm):
    """controller.loop() should correctly route to SPGD."""
    from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsController
    from conftest import MetricImageProvider

    np.random.seed(42)

    zernike_indices = [4]
    optimum_values = [10.0]
    provider = MetricImageProvider(virtual_dm, optimum_values, zernike_indices)

    ctrl = AdaptiveOpticsController()
    ctrl.deformable_mirror = virtual_dm
    ctrl.image_provider = provider
    ctrl.verbose = False
    ctrl.selected_metric = 'normalized_variance'
    ctrl.optim_method = 'spgd'
    ctrl.initial_zernike_indices = zernike_indices
    ctrl.zernike_initial_coefficients_nm = [0]
    ctrl.zernike_amplitude_scan_nm = [80]
    ctrl.spgd_gain = 0.1
    ctrl.spgd_delta = 5.0
    ctrl.spgd_iterations = 10
    ctrl.continuous_scan = False
    ctrl.enable_convergence_detection = False
    ctrl.use_current_dm_state = False

    _run_generator(ctrl.loop())

    assert ctrl.final_coefficients is not None, "SPGD loop did not produce final coefficients"


@unittest.mock.patch('time.sleep')
def test_loop_routing_random_search(mock_sleep, virtual_dm):
    """controller.loop() should correctly route to random search."""
    from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsController
    from conftest import MetricImageProvider

    np.random.seed(42)

    zernike_indices = [4]
    optimum_values = [10.0]
    provider = MetricImageProvider(virtual_dm, optimum_values, zernike_indices)

    ctrl = AdaptiveOpticsController()
    ctrl.deformable_mirror = virtual_dm
    ctrl.image_provider = provider
    ctrl.verbose = False
    ctrl.selected_metric = 'normalized_variance'
    ctrl.optim_method = 'random_search'
    ctrl.initial_zernike_indices = zernike_indices
    ctrl.zernike_initial_coefficients_nm = [0]
    ctrl.zernike_amplitude_scan_nm = [80]
    ctrl.random_search_iterations = 10
    ctrl.random_search_range = 50.0
    ctrl.continuous_scan = False
    ctrl.use_current_dm_state = False

    _run_generator(ctrl.loop())

    assert ctrl.final_coefficients is not None, "Random search loop did not produce final coefficients"


@unittest.mock.patch('time.sleep')
def test_loop_routing_sequential(mock_sleep, virtual_dm):
    """controller.loop() should correctly route to sequential optimization."""
    from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsController
    from conftest import MetricImageProvider

    np.random.seed(42)

    zernike_indices = [4]
    optimum_values = [10.0]
    provider = MetricImageProvider(virtual_dm, optimum_values, zernike_indices)

    ctrl = AdaptiveOpticsController()
    ctrl.deformable_mirror = virtual_dm
    ctrl.image_provider = provider
    ctrl.verbose = False
    ctrl.print_plot = False
    ctrl.selected_metric = 'normalized_variance'
    ctrl.optim_method = 'simple_interpolation'
    ctrl.initial_zernike_indices = zernike_indices
    ctrl.zernike_initial_coefficients_nm = [0]
    ctrl.zernike_amplitude_scan_nm = [80]
    ctrl.optim_iterations = 1
    ctrl.num_steps_per_mode = 5
    ctrl.continuous_scan = False
    ctrl.use_current_dm_state = False
    ctrl.image_log = False
    ctrl.optim_iterations_per_mode = []

    _run_generator(ctrl.loop())

    assert ctrl.final_coefficients is not None, "Sequential loop did not produce final coefficients"
