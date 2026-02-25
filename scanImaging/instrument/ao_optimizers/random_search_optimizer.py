"""
Random Search Optimizer

Simple baseline algorithm that randomly samples the coefficient space.
Useful for comparison with more sophisticated methods.

Algorithm:
1. Generate random coefficients within search range
2. Measure metric
3. Keep best coefficients found
4. Repeat for specified iterations
"""

import time
import numpy as np
from .base_optimizer import BaseOptimizer


class RandomSearchOptimizer(BaseOptimizer):
    """
    Random search baseline optimizer for adaptive optics.

    Very robust (no local minima issues) but inefficient compared to
    gradient-based methods. Serves as a baseline to validate that other
    algorithms actually improve performance.
    """

    def __init__(self, controller):
        """
        Initialize random search optimizer.

        Args:
            controller: AdaptiveOpticsController instance
        """
        super().__init__(controller)

        # Random search parameters (from controller)
        self.max_iterations = controller.random_search_iterations
        self.search_range = controller.random_search_range

        # Continuous scan mode
        self.continuous_scan = controller.continuous_scan

        # State
        self.zernike_indices = None
        self.initial_coefficients_full = None
        self.best_coeffs = None
        self.best_metric = None
        self.metric_fn = None

        # History
        self.metric_history = []

    def initialize(self, initial_coefficients, zernike_indices, **kwargs):
        """
        Initialize random search optimizer.

        Args:
            initial_coefficients: np.ndarray of initial Zernike coefficients
            zernike_indices: List of Zernike mode indices to optimize
        """
        self.initial_coefficients_full = initial_coefficients.copy()
        self.best_coeffs = initial_coefficients.copy()
        self.zernike_indices = zernike_indices
        self.metric_fn = self.controller.get_metric_function()
        self.iteration = 0
        self.metric_history = []

        if self.verbose:
            print("\n" + "="*60)
            print("Starting Random Search optimization")
            print("="*60)
            print(f"Iterations: {self.max_iterations}")
            print(f"Search range: +/-{self.search_range} nm")
            print(f"Modes: {self.zernike_indices}")
            print(f"Using metric: {self.controller.selected_metric}")

        # Measure initial metric
        self._apply_coefficients_to_dm(self.best_coeffs)
        time.sleep(0.05)
        img_initial = self._acquire_image()
        self.best_metric = self.metric_fn(img_initial)
        self.metric_history.append(self.best_metric)

        if self.verbose:
            print(f"Initial metric: {self.best_metric:.4f}")
            print(f"Initial coefficients: {self.best_coeffs[self.zernike_indices]}")

    def step(self):
        """
        Perform one random search iteration: sample random point, evaluate, keep if better.

        Yields once to allow GUI updates.

        Returns:
            dict: Progress information
        """
        # Generate random coefficients within search range
        random_coeffs = self.best_coeffs.copy()
        for zernike_index in self.zernike_indices:
            offset = np.random.uniform(-self.search_range, self.search_range)
            random_coeffs[zernike_index] = self.initial_coefficients_full[zernike_index] + offset

        # Measure metric at random point
        # Direct DM call (transient evaluation state)
        self._apply_coefficients_to_dm(random_coeffs)
        time.sleep(0.05)
        img = self._acquire_image()
        metric_value = self.metric_fn(img)

        # Update best if improved
        if metric_value > self.best_metric:
            self.best_metric = metric_value
            self.best_coeffs = random_coeffs.copy()
            if self.verbose:
                print(f"Iteration {self.iteration+1}/{self.max_iterations}: "
                      f"New best metric = {self.best_metric:.4f}")
                print(f"  Best coefficients: {self.best_coeffs[self.zernike_indices]}")

        self.metric_history.append(self.best_metric)

        # Apply best coefficients to DM and notify GUI (standard pattern)
        self.controller._update_dm_and_notify(self.best_coeffs,
                                               iteration=self.iteration,
                                               metric=self.best_metric,
                                               random_search_metric_history=self.metric_history.copy())

        self.iteration += 1

        yield  # Allow interruption

        progress = {
            'coefficients': self.best_coeffs.copy(),
            'metric': self.best_metric,
            'iteration': self.iteration,
            'random_search_metric_history': self.metric_history.copy(),
        }

        return progress

    def is_complete(self):
        """Check if optimization is complete."""
        return self.iteration >= self.max_iterations

    def finalize(self):
        """Finalize random search optimization."""
        self.final_coefficients = self.best_coeffs.copy()

        if self.verbose:
            print(f"\nRandom search optimization complete!")
            print(f"Best metric: {self.best_metric:.4f}")
            print(f"Best coefficients: {self.best_coeffs[:min(12, len(self.best_coeffs))]}")
            if len(self.metric_history) > 1:
                improvement = (self.best_metric - self.metric_history[0]) / (abs(self.metric_history[0]) + 1e-12) * 100
                print(f"Improvement: {improvement:.1f}%")

        # Final notification with complete history
        self.controller._update_dm_and_notify(self.best_coeffs,
                                               final_coefficients=self.best_coeffs.copy(),
                                               random_search_metric_history=self.metric_history.copy(),
                                               iteration=len(self.metric_history) - 1)

        # Stop continuous scan mode if enabled
        if self.continuous_scan:
            if self.verbose:
                print("Stopping continuous acquisition mode...")
            self.controller.image_provider.stopContinuousMode()

    def run(self):
        """
        Main random search loop with proper cleanup.

        Yields:
            dict: Progress information at each step
        """
        try:
            if self.continuous_scan:
                self.controller.image_provider.startContinuousMode()

            yield from super().run()
        finally:
            self.finalize()
