"""
Sequential Scan-Based Optimizer

Implements the original sequential hill-climbing algorithm that:
1. Optimizes one Zernike mode at a time (hierarchical)
2. Scans around current value with fixed number of points
3. Fits parabola to find optimum
4. Adapts scan range if optimum is outside range
5. Supports simple_interpolation (3 points) or weighted_fit (accumulated)
"""

import numpy as np
from pathlib import Path
from .base_optimizer import BaseOptimizer


class SequentialOptimizer(BaseOptimizer):
    """
    Sequential scan-based optimizer for adaptive optics.

    Optimizes Zernike modes one at a time using scan patterns and parabolic fitting.
    Supports both simple interpolation (current scan only) and weighted fit (accumulated data).
    """

    def __init__(self, controller):
        """
        Initialize sequential optimizer.

        Args:
            controller: AdaptiveOpticsController instance
        """
        super().__init__(controller)

        # Algorithm parameters (from controller)
        self.optim_method = controller.optim_method  # 'simple_interpolation' or 'weighted_fit'
        self.num_steps_per_mode = controller.num_steps_per_mode
        self.optim_iterations = controller.optim_iterations
        self.optim_iterations_per_mode = controller.optim_iterations_per_mode

        # Convergence detection
        self.enable_convergence_detection = controller.enable_convergence_detection
        self.convergence_threshold = controller.convergence_threshold
        self.convergence_window = controller.convergence_window

        # Image logging
        self.image_log = controller.image_log
        self.recorded_image_folder = controller.recorded_image_folder
        self.print_plot = controller.print_plot

        # Continuous scan mode
        self.continuous_scan = controller.continuous_scan

        # Optimization state
        self.zernike_indices = None
        self.current_coefficients = None
        self.initial_coefficients = None
        self.scan_amplitudes = None

        # Per-mode state
        self.current_mode_idx = 0
        self.current_mode_iteration = 0
        self.mode_iterations = None

        # Accumulated data for weighted_fit
        self.accumulated_images = []
        self.accumulated_params = []
        self.metric_history = []

        # Results storage
        self.imageSet = []
        self.parameter_set = []
        self.opt_v_set = []

    def initialize(self, initial_coefficients, zernike_indices, scan_amplitudes, **kwargs):
        """
        Initialize the optimizer with problem parameters.

        Args:
            initial_coefficients: np.ndarray of initial Zernike coefficients
            zernike_indices: List of Zernike mode indices to optimize
            scan_amplitudes: List of scan amplitude (nm) for each mode
            **kwargs: Additional parameters (ignored for now)
        """
        self.initial_coefficients = initial_coefficients.copy()
        self.current_coefficients = initial_coefficients.copy()
        self.zernike_indices = zernike_indices
        self.scan_amplitudes = scan_amplitudes

        # Build per-mode iteration counts
        self.mode_iterations = self._build_iterations_dict()

        if self.verbose:
            print("Sequential Optimizer Initialized")
            print(f"Optimizing indices: {self.zernike_indices}")
            print(f"Initial coefficients (nm): {initial_coefficients[zernike_indices]}")
            print(f"Scan amplitudes (nm): {self.scan_amplitudes}")
            if self.optim_iterations_per_mode:
                print(f"Per-mode iterations: {self.mode_iterations}")

        # Create image log folder if needed
        if self.image_log:
            p = Path(self.recorded_image_folder)
            p.mkdir(parents=True, exist_ok=True)

        # Start continuous scan mode if enabled
        if self.continuous_scan:
            self.controller.image_provider.startContinuousMode()

        # Reset state
        self.current_mode_idx = 0
        self.current_mode_iteration = 0
        self.iteration = 0

    def _build_iterations_dict(self):
        """
        Build dictionary mapping Zernike index to iteration count.

        Returns:
            dict: {zernike_index: num_iterations}
        """
        if not self.optim_iterations_per_mode:
            return {idx: self.optim_iterations for idx in self.zernike_indices}

        iterations_dict = {}
        for idx in self.zernike_indices:
            iterations_dict[idx] = self.optim_iterations_per_mode.get(idx, self.optim_iterations)
        return iterations_dict

    def step(self):
        """
        Perform one optimization step (scan + optimize one mode for one iteration).

        Yields at each image acquisition to allow GUI updates.

        Returns:
            dict: Progress information
        """
        # Get current mode
        zernike_index = self.zernike_indices[self.current_mode_idx]
        mode_idx = self.current_mode_idx
        max_iterations = self.mode_iterations[zernike_index]

        if self.current_mode_iteration == 0:
            # Starting new mode
            if self.verbose:
                print(f"\n=== Optimizing mode {zernike_index} with {max_iterations} iterations ===")
            # Reset mode-specific state
            self.metric_history = []
            self.accumulated_images = []
            self.accumulated_params = []

        # Prepare scan parameters
        scan_amplitude = self.scan_amplitudes[mode_idx]
        num_scan_points = self.num_steps_per_mode
        scan_values = np.linspace(
            self.initial_coefficients[zernike_index] - scan_amplitude/2,
            self.initial_coefficients[zernike_index] + scan_amplitude/2,
            num_scan_points
        )

        # Acquire images at scan points
        image_stack = []
        parameter_stack = []
        for val in scan_values:
            self.current_coefficients[zernike_index] = val

            # Acquire image and update DM
            image = self.controller._updateImageAndDM(
                current_coefficients=self.current_coefficients,
                zernike_index=zernike_index,
                val=val
            )
            image_stack.append(image)
            parameter_stack.append(val)

            yield  # Allow interruption

            if self.controller._stop_requested:
                if self.verbose:
                    print("Stop requested - exiting AO loop")
                return None

        # Accumulate data for weighted_fit
        if self.optim_method == 'weighted_fit':
            self.accumulated_images.extend(image_stack)
            self.accumulated_params.extend(parameter_stack)

        # Find optimal parameter with range extension if needed
        miss_max = 1
        while num_scan_points < 100 and abs(miss_max) != 0:
            metric_fn = self.controller.get_metric_function()

            # Select optimization algorithm
            if self.optim_method == 'weighted_fit':
                optimal_value, miss_max, opt_v = self.controller.computeOptimalParametersWeightedFit(
                    self.accumulated_images,
                    self.accumulated_params,
                    optim_metric=metric_fn,
                    plot=self.print_plot
                )
                range_check_params = self.accumulated_params
            else:  # 'simple_interpolation'
                optimal_value, miss_max, opt_v = self.controller.computeOptimalParametersSimpleInterpolation(
                    image_stack,
                    parameter_stack,
                    optim_metric=metric_fn,
                    plot=self.print_plot
                )
                range_check_params = parameter_stack

            # Adjust scan range if optimum is outside
            if miss_max != 0:
                num_scan_points += 1
                scan_amplitude *= 1.2
            else:
                scan_amplitude *= 0.7

            # Acquire additional sample if needed
            added_value = None
            if miss_max < 0:
                added_value = range_check_params[0] - scan_amplitude/2
                if optimal_value < range_check_params[0]:
                    added_value = optimal_value - scan_amplitude/2
                parameter_stack = [added_value] + parameter_stack
            elif miss_max > 0:
                added_value = range_check_params[-1] + scan_amplitude/2
                if optimal_value > range_check_params[-1]:
                    added_value = optimal_value + scan_amplitude/2
                parameter_stack = parameter_stack + [added_value]

            if added_value is not None:
                self.current_coefficients[zernike_index] = added_value
                image = self.controller._updateImageAndDM(
                    current_coefficients=self.current_coefficients,
                    zernike_index=zernike_index,
                    val=added_value
                )

                # Add to stacks
                if miss_max < 0:
                    image_stack = [image] + image_stack
                    if self.optim_method == 'weighted_fit':
                        self.accumulated_images.insert(0, image)
                        self.accumulated_params.insert(0, added_value)
                else:
                    image_stack = image_stack + [image]
                    if self.optim_method == 'weighted_fit':
                        self.accumulated_images.append(image)
                        self.accumulated_params.append(added_value)

                yield  # Allow interruption

                if self.controller._stop_requested:
                    if self.verbose:
                        print("Stop requested - exiting AO loop")
                    return None

        # Apply optimal value
        self.initial_coefficients[zernike_index] = optimal_value
        self.current_coefficients[zernike_index] = optimal_value
        self._apply_coefficients_to_dm(self.current_coefficients)

        # Notify dependents
        self._notify_progress(coefficients=self.initial_coefficients.copy())

        if self.verbose:
            print(f"Optimized Zernike index {zernike_index} to value {optimal_value:.2f} nm "
                  f"(iteration {self.current_mode_iteration+1}/{max_iterations}), metric={opt_v:.4f}")

        # Track convergence
        self.metric_history.append(opt_v)

        # Check convergence
        converged = False
        if self.enable_convergence_detection and len(self.metric_history) >= self.convergence_window + 1:
            recent_metrics = self.metric_history[-self.convergence_window-1:]
            relative_improvement = (recent_metrics[-1] - recent_metrics[0]) / (abs(recent_metrics[0]) + 1e-12)

            if relative_improvement < self.convergence_threshold:
                if self.verbose:
                    print(f"Converged! Improvement {relative_improvement:.4f} < "
                          f"threshold {self.convergence_threshold:.4f}")
                    print(f"Stopping after {self.current_mode_iteration+1}/{max_iterations} "
                          f"iterations for mode {zernike_index}")
                converged = True

        # Update iteration counters
        self.current_mode_iteration += 1
        self.iteration += 1

        # Check if this mode is done
        mode_complete = converged or (self.current_mode_iteration >= max_iterations)

        if mode_complete:
            # Get final image for this mode
            image = self._acquire_image()
            self.imageSet.append(image)
            self.parameter_set.append(self.initial_coefficients.copy())
            self.opt_v_set.append(opt_v)

            # Final update for this mode
            self.controller._updateImageAndDM(
                current_coefficients=self.initial_coefficients,
                zernike_index=0,
                val=0
            )

            # Notify final coefficients
            try:
                self._notify_progress(final_coefficients=self.initial_coefficients.copy())
            except Exception as e:
                if self.verbose:
                    print(f"Error notifying dependents after mode optimization: {e}")

            # Move to next mode
            self.current_mode_idx += 1
            self.current_mode_iteration = 0

        # Return progress information
        progress = {
            'coefficients': self.initial_coefficients.copy(),
            'metric': opt_v,
            'iteration': self.iteration,
            'mode_idx': self.current_mode_idx,
            'zernike_index': zernike_index,
            'mode_iteration': self.current_mode_iteration,
        }

        return progress

    def is_complete(self):
        """
        Check if optimization is complete.

        Returns:
            bool: True if all modes have been optimized
        """
        return self.current_mode_idx >= len(self.zernike_indices)

    def finalize(self):
        """
        Finalize optimization and save results.
        """
        # Store final coefficients
        self.final_coefficients = self.initial_coefficients.copy()

        if self.verbose:
            print(f"\nSequential optimization complete!")
            print(f"Final coefficients: {self.final_coefficients[:min(12, len(self.final_coefficients))]}")

        # Save results if logging enabled
        if self.image_log:
            np.save(self.recorded_image_folder + '/' + 'imageSet', self.imageSet)
            np.save(self.recorded_image_folder + '/' + 'parameterSet', self.parameter_set)
            np.save(self.recorded_image_folder + '/' + 'optimalValuesSet', self.opt_v_set)

        # Stop continuous scan mode if enabled
        if self.continuous_scan:
            if self.verbose:
                print("Stopping continuous acquisition mode...")
            self.controller.image_provider.stopContinuousMode()

    def run(self):
        """
        Main optimization loop with proper cleanup.

        Yields:
            dict: Progress information at each step
        """
        try:
            # Run base class loop
            yield from super().run()
        finally:
            # Ensure cleanup happens
            self.finalize()
