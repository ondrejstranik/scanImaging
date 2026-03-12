"""
Base Optimizer Abstract Class

Defines the interface that all AO optimization algorithms must implement.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizer(ABC):
    """
    Abstract base class for adaptive optics optimization algorithms.

    All optimizer implementations must inherit from this class and implement
    the required abstract methods.

    The optimizer takes a reference to the controller to access hardware
    (deformable_mirror, image_provider) and utility methods (metric computation, etc.).
    """

    def __init__(self, controller):
        """
        Initialize the optimizer with reference to the controller.

        Args:
            controller: AdaptiveOpticsController instance providing access to:
                - deformable_mirror: DM hardware interface
                - image_provider: Camera/image acquisition interface
                - image_processor: Optional image processing pipeline
                - get_metric_function(): Method to get metric computation function
                - acquire_and_process_image(): Centralized image acquisition
                - verbose: Debug output flag
                - _stop_requested: Flag to check for user interruption
        """
        self.controller = controller
        self.verbose = controller.verbose

        # Optimization state
        self.iteration = 0
        self.completed = False

        # Results
        self.final_coefficients = None
        self.optimization_history = []

    @abstractmethod
    def initialize(self, initial_coefficients, zernike_indices, **kwargs):
        """
        Initialize the optimizer with problem parameters.

        Args:
            initial_coefficients: np.ndarray of initial Zernike coefficients
            zernike_indices: List of Zernike mode indices to optimize
            **kwargs: Algorithm-specific parameters
        """
        pass

    @abstractmethod
    def step(self):
        """
        Perform one optimization step.

        This method should:
        1. Generate next sample point(s)
        2. Acquire image(s) via self.controller.acquire_and_process_image()
        3. Compute metric(s)
        4. Update internal state
        5. Apply improved coefficients to DM
        6. Return progress information dict

        Returns:
            dict: Progress information containing:
                - 'coefficients': Current best coefficients
                - 'metric': Current best metric value
                - 'iteration': Current iteration number
                - Algorithm-specific additional fields

        Yields:
            Should yield periodically to allow GUI updates and interruption checks
        """
        pass

    @abstractmethod
    def is_complete(self):
        """
        Check if optimization is complete.

        Returns:
            bool: True if optimization should stop, False otherwise
        """
        pass

    def run(self):
        """
        Main optimization loop - yields progress at each step.

        This is a generator that can be called with `yield from optimizer.run()`
        from the controller's loop() method.

        Yields:
            dict: Progress information at each step
        """
        if self.verbose:
            print(f"Starting {self.__class__.__name__} optimization")

        while not self.is_complete():
            # Check for user stop request
            if self.controller._stop_requested:
                if self.verbose:
                    print("Stop requested - exiting optimization")
                return

            # Perform optimization step (yields internally for interruption)
            progress = yield from self.step()

            # Record progress
            if progress is not None:
                self.optimization_history.append(progress)

        # Mark as completed
        self.completed = True
        self.final_coefficients = progress.get('coefficients') if progress else None

        if self.verbose:
            print(f"Optimization complete after {self.iteration} iterations")
            if self.final_coefficients is not None:
                print(f"Final coefficients: {self.final_coefficients[:min(12, len(self.final_coefficients))]}")

    def get_results(self):
        """
        Get final optimization results.

        Returns:
            dict: Results containing:
                - 'final_coefficients': Optimized Zernike coefficients
                - 'optimization_history': List of progress dicts from each step
                - 'completed': Whether optimization finished normally
        """
        return {
            'final_coefficients': self.final_coefficients,
            'optimization_history': self.optimization_history,
            'completed': self.completed
        }

    # Utility methods that subclasses can use

    def _apply_coefficients_to_dm(self, coefficients):
        """
        Apply coefficients to deformable mirror.

        Args:
            coefficients: np.ndarray of Zernike coefficients
        """
        self.controller.deformable_mirror.set_phase_map_from_zernike(coefficients)
        self.controller.deformable_mirror.display_surface()

    def _acquire_image(self):
        """
        Acquire and process image using controller's pipeline.

        Returns:
            np.ndarray: Processed image ready for metric computation
        """
        return self.controller.acquire_and_process_image()

    def _compute_metric(self, image):
        """
        Compute image quality metric.

        Args:
            image: np.ndarray image

        Returns:
            float: Metric value (higher is better)
        """
        metric_fn = self.controller.get_metric_function()
        return metric_fn(image)

    def _notify_progress(self, **params):
        """
        Notify GUI/dependents of progress update.

        Args:
            **params: Parameters to pass to dependents (e.g., coefficients, metric, etc.)
        """
        try:
            self.controller._notify_dependents(params=params)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to notify dependents: {e}")
