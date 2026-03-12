"""
Shared pytest fixtures for AO testing framework.

Provides reusable mock objects and controller instances for testing
metrics, fitting algorithms, and optimizers without hardware.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure scanImaging is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


class MockImageProvider:
    """
    Mock image provider for testing.

    Supports three modes:
    1. Fixed image: returns the same image every call (set via set_image)
    2. Function-based: calls a function to generate each image (set via set_image_function)
    3. Default: returns random 100x100 image
    """

    def __init__(self):
        self._image = None
        self._image_fn = None
        self.call_count = 0

    def set_image(self, img):
        """Set a fixed image to return on every getImage() call."""
        self._image = img
        self._image_fn = None

    def set_image_function(self, fn):
        """Set a function that generates images. Called on every getImage()."""
        self._image_fn = fn
        self._image = None

    def getImage(self):
        self.call_count += 1
        if self._image_fn is not None:
            return self._image_fn()
        if self._image is not None:
            return self._image.copy()
        return np.random.rand(100, 100)

    def get_image(self):
        return self.getImage()

    def startContinuousMode(self):
        pass

    def stopContinuousMode(self):
        pass


class MetricImageProvider:
    """
    Image provider where image quality depends on DM Zernike coefficients.

    Produces images with a Gaussian point source whose width (blur) depends on
    the distance from the known optimum in Zernike coefficient space:
    - At optimum: sharp point source (sigma_min) → high metric value
    - Far from optimum: broad blur (sigma_max) → low metric value

    This works correctly with all 5 production metrics (they all increase with sharpness).
    """

    def __init__(self, dm, optimum_coeffs, zernike_indices, scale=1000.0, noise=0.1):
        self.dm = dm
        self.optimum = np.asarray(optimum_coeffs, dtype=float)
        self.indices = list(zernike_indices)
        self.scale = scale
        self.noise = noise
        self.call_count = 0
        self._size = 64
        self._sigma_min = 1.0   # Sharp PSF at optimum
        self._sigma_max = 15.0  # Broad blur far from optimum

    def getImage(self):
        from scipy.ndimage import gaussian_filter
        self.call_count += 1
        coeffs = self.dm.get_current_coefficients()
        if coeffs is None:
            coeffs = np.zeros(max(self.indices) + 1)

        # Quadratic error from optimum
        error = 0.0
        for i, idx in enumerate(self.indices):
            if idx < len(coeffs) and i < len(self.optimum):
                error += (coeffs[idx] - self.optimum[i]) ** 2

        # Map error to PSF width: sigma increases with error
        sigma = self._sigma_min + (self._sigma_max - self._sigma_min) * (1.0 - np.exp(-error / self.scale))

        # Create point source and blur it
        img = np.zeros((self._size, self._size), dtype=np.float64)
        center = self._size // 2
        img[center, center] = 1000.0
        img = gaussian_filter(img, sigma=sigma)

        # Add small noise
        if self.noise > 0:
            img += np.abs(np.random.randn(self._size, self._size) * self.noise)

        return np.clip(img, 0, None)

    def get_image(self):
        return self.getImage()

    def startContinuousMode(self):
        pass

    def stopContinuousMode(self):
        pass


@pytest.fixture
def virtual_dm():
    """Connected VirtualDMBmc instance."""
    from scanImaging.instrument.virtual.virtualDMBmc import VirtualDMBmc
    dm = VirtualDMBmc()
    dm.connect()
    yield dm
    dm.disconnect()


@pytest.fixture
def mock_image_provider():
    """MockImageProvider instance."""
    return MockImageProvider()


@pytest.fixture
def controller(virtual_dm, mock_image_provider):
    """
    AdaptiveOpticsController with VirtualDMBmc + MockImageProvider.

    Ready for testing metrics, fitting, and optimization without hardware or GUI.
    """
    from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsController

    ctrl = AdaptiveOpticsController()
    ctrl.deformable_mirror = virtual_dm
    ctrl.image_provider = mock_image_provider
    ctrl.verbose = False
    ctrl.print_plot = False

    # Default optimization parameters
    ctrl.initial_zernike_indices = [4, 3, 5]
    ctrl.zernike_initial_coefficients_nm = [0, 0, 0]
    ctrl.zernike_amplitude_scan_nm = [80, 60, 60]
    ctrl.optim_iterations = 1
    ctrl.num_steps_per_mode = 5

    return ctrl
