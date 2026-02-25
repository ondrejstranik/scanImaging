"""
SPGD (Stochastic Parallel Gradient Descent) Optimizer

Optimizes ALL Zernike modes simultaneously using stochastic perturbations.
Very robust to Poisson noise and mode coupling.

Algorithm:
1. Apply random perturbation +δ to all coefficients
2. Measure metric J+
3. Apply perturbation -δ
4. Measure metric J-
5. Estimate gradient: ∇J ≈ (J+ - J-) / (2δ) × δ
6. Update coefficients: c += gain × ∇J
"""

import time
import numpy as np
from .base_optimizer import BaseOptimizer


class SPGDOptimizer(BaseOptimizer):
    """
    SPGD optimizer for adaptive optics.

    Optimizes all Zernike modes simultaneously using stochastic perturbations.
    Best for deep tissue imaging with large aberrations and low SNR.
    """

    def __init__(self, controller):
        """
        Initialize SPGD optimizer.

        Args:
            controller: AdaptiveOpticsController instance
        """
        super().__init__(controller)

        # SPGD parameters (from controller)
        self.gain = controller.spgd_gain
        self.delta = controller.spgd_delta
        self.max_iterations = controller.spgd_iterations

        # Convergence detection
        self.enable_convergence_detection = controller.enable_convergence_detection
        self.convergence_threshold = controller.convergence_threshold
        self.convergence_window = controller.convergence_window

        # Continuous scan mode
        self.continuous_scan = controller.continuous_scan

        # State
        self.zernike_indices = None
        self.coeffs = None
        self.metric_fn = None

        # History
        self.metric_history = []
        self.coeff_history = []

    def initialize(self, initial_coefficients, zernike_indices, **kwargs):
        """
        Initialize SPGD optimizer.

        Args:
            initial_coefficients: np.ndarray of initial Zernike coefficients
            zernike_indices: List of Zernike mode indices to optimize
        """
        self.coeffs = initial_coefficients.copy()
        self.zernike_indices = zernike_indices
        self.metric_fn = self.controller.get_metric_function()
        self.iteration = 0
        self.metric_history = []
        self.coeff_history = []

        if self.verbose:
            print("\n" + "="*60)
            print("Starting SPGD optimization")
            print("="*60)
            print(f"Gain: {self.gain}")
            print(f"Delta: {self.delta} nm")
            print(f"Iterations: {self.max_iterations}")
            print(f"Modes: {self.zernike_indices}")
            print(f"Using metric: {self.controller.selected_metric}")

    def step(self):
        """
        Perform one SPGD iteration: perturb +/-, measure, estimate gradient, update.

        Yields twice (once per perturbation measurement) to allow GUI updates.

        Returns:
            dict: Progress information
        """
        # Generate random perturbation for optimized modes only
        perturbation = np.zeros_like(self.coeffs)
        for zernike_index in self.zernike_indices:
            perturbation[zernike_index] = np.random.randn() * self.delta

        # Measure metric with positive perturbation
        # Direct DM call (transient state, no GUI notification)
        coeffs_plus = self.coeffs + perturbation
        self._apply_coefficients_to_dm(coeffs_plus)
        time.sleep(0.05)  # Shorter settling for SPGD
        img_plus = self._acquire_image()
        metric_plus = self.metric_fn(img_plus)

        yield  # Allow interruption

        if self.controller._stop_requested:
            return None

        # Measure metric with negative perturbation
        # Direct DM call (transient state, no GUI notification)
        coeffs_minus = self.coeffs - perturbation
        self._apply_coefficients_to_dm(coeffs_minus)
        time.sleep(0.05)
        img_minus = self._acquire_image()
        metric_minus = self.metric_fn(img_minus)

        yield  # Allow interruption

        if self.controller._stop_requested:
            return None

        # Estimate gradient and update coefficients
        gradient_estimate = (metric_plus - metric_minus) / (2 * self.delta)
        self.coeffs += self.gain * gradient_estimate * perturbation

        # Track metrics
        avg_metric = (metric_plus + metric_minus) / 2
        self.metric_history.append(avg_metric)
        self.coeff_history.append(self.coeffs.copy())

        # Apply updated coefficients and notify GUI (standard pattern)
        self.controller._update_dm_and_notify(self.coeffs,
                                               iteration=self.iteration,
                                               metric=avg_metric,
                                               spgd_metric_history=self.metric_history.copy())

        # Print progress every 5 iterations
        if self.verbose and (self.iteration % 5 == 0 or self.iteration == self.max_iterations - 1):
            print(f"SPGD iter {self.iteration+1}/{self.max_iterations}: metric={avg_metric:.4f}")
            print(f"  Coeffs: {self.coeffs[self.zernike_indices]}")

        # Check convergence
        converged = False
        if self.enable_convergence_detection and len(self.metric_history) >= self.convergence_window + 1:
            recent_metrics = self.metric_history[-self.convergence_window-1:]
            relative_improvement = (recent_metrics[-1] - recent_metrics[0]) / (abs(recent_metrics[0]) + 1e-12)

            if relative_improvement < self.convergence_threshold:
                if self.verbose:
                    print(f"SPGD converged after {self.iteration+1} iterations!")
                    print(f"Improvement {relative_improvement:.4f} < threshold {self.convergence_threshold:.4f}")
                converged = True

        self.iteration += 1

        # Force completion if converged
        if converged:
            self.iteration = self.max_iterations

        progress = {
            'coefficients': self.coeffs.copy(),
            'metric': avg_metric,
            'iteration': self.iteration,
            'spgd_metric_history': self.metric_history.copy(),
        }

        return progress

    def is_complete(self):
        """Check if optimization is complete."""
        return self.iteration >= self.max_iterations

    def finalize(self):
        """Finalize SPGD optimization."""
        self.final_coefficients = self.coeffs.copy()

        if self.verbose:
            print(f"\nSPGD optimization complete!")
            print(f"Final coefficients: {self.coeffs[:min(12, len(self.coeffs))]}")
            if self.metric_history:
                print(f"Final metric: {self.metric_history[-1]:.4f}")

        # Final notification with complete metric history
        try:
            self._notify_progress(
                current_coefficients=self.coeffs.copy(),
                final_coefficients=self.coeffs.copy(),
                spgd_metric_history=self.metric_history.copy(),
                iteration=len(self.metric_history) - 1
            )
        except Exception:
            pass

        # Stop continuous scan mode if enabled
        if self.continuous_scan:
            if self.verbose:
                print("Stopping continuous acquisition mode...")
            self.controller.image_provider.stopContinuousMode()

    def run(self):
        """
        Main SPGD loop with proper cleanup.

        Yields:
            dict: Progress information at each step
        """
        try:
            if self.continuous_scan:
                self.controller.image_provider.startContinuousMode()

            yield from super().run()
        finally:
            self.finalize()
