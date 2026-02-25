"""
Tests for Zernike polynomial utilities and PSF generation.

Tests:
- zernike_phase_map() from algorithm/SimpleZernike.py
- generatePSF() from virtual/virtualISM.py
- add_microscopy_noise() from virtual/virtualISM.py
"""

import numpy as np
import pytest


def test_zernike_zero_coefficients():
    """All-zero coefficients should produce a zero (or near-zero) phase map."""
    from scanImaging.algorithm.SimpleZernike import zernike_phase_map

    coeffs = [0, 0, 0, 0, 0, 0]
    phase_map = zernike_phase_map(coeffs, 32, 32)

    assert phase_map.shape == (32, 32)
    assert np.allclose(phase_map, 0.0, atol=1e-10), (
        f"Expected zero phase map, got max={np.max(np.abs(phase_map))}"
    )


def test_zernike_output_shape():
    """Output shape should match requested grid dimensions."""
    from scanImaging.algorithm.SimpleZernike import zernike_phase_map

    coeffs = [0, 0, 0, 0, 100.0]  # Some defocus
    phase_map = zernike_phase_map(coeffs, 64, 48)

    assert phase_map.shape == (48, 64), f"Expected (48, 64), got {phase_map.shape}"


def test_zernike_defocus_symmetry():
    """Defocus (j=4) should produce a radially symmetric pattern."""
    from scanImaging.algorithm.SimpleZernike import zernike_phase_map

    # Only defocus (j=4, OSA index)
    # Use odd-sized grid so center pixel aligns exactly with mathematical center
    coeffs = [0, 0, 0, 0, 100.0]
    size = 65
    phase_map = zernike_phase_map(coeffs, size, size)

    # Check symmetry: value at (center+d, center) should equal (center-d, center)
    center = size // 2
    for d in range(1, min(10, center)):
        val_plus = phase_map[center, center + d]
        val_minus = phase_map[center, center - d]
        if val_plus != 0 or val_minus != 0:  # Skip if outside aperture
            assert abs(val_plus - val_minus) < 1e-6, (
                f"Defocus not symmetric at offset {d}: {val_plus} vs {val_minus}"
            )


def test_zernike_nonzero_coefficients():
    """Non-zero coefficients should produce non-zero phase map inside aperture."""
    from scanImaging.algorithm.SimpleZernike import zernike_phase_map

    coeffs = [0, 0, 0, 0, 100.0, 50.0]  # defocus + astigmatism
    phase_map = zernike_phase_map(coeffs, 32, 32)

    assert np.max(np.abs(phase_map)) > 0, "Expected non-zero phase map"


def test_generate_psf_shape():
    """PSF output should have correct shape."""
    from scanImaging.instrument.virtual.virtualISM import generatePSF

    size = 64
    aperture_mask = np.zeros((size, size))
    rr, cc = np.ogrid[:size, :size]
    center = size // 2
    radius = 20
    aperture_mask[(rr - center) ** 2 + (cc - center) ** 2 <= radius ** 2] = 1.0

    aperture_phase = np.zeros((size, size))
    psf = generatePSF(aperture_mask, aperture_phase, 520.0)

    assert psf.shape == (size, size), f"Expected ({size}, {size}), got {psf.shape}"


def test_generate_psf_zero_aberration():
    """Zero aberration should produce a PSF with peak near center."""
    from scanImaging.instrument.virtual.virtualISM import generatePSF

    size = 64
    aperture_mask = np.zeros((size, size))
    rr, cc = np.ogrid[:size, :size]
    center = size // 2
    radius = 15
    aperture_mask[(rr - center) ** 2 + (cc - center) ** 2 <= radius ** 2] = 1.0

    aperture_phase = np.zeros((size, size))
    psf = generatePSF(aperture_mask, aperture_phase, 520.0)

    # Peak should be near center
    peak_idx = np.unravel_index(np.argmax(psf), psf.shape)
    assert abs(peak_idx[0] - center) <= 1, f"PSF peak row {peak_idx[0]} far from center {center}"
    assert abs(peak_idx[1] - center) <= 1, f"PSF peak col {peak_idx[1]} far from center {center}"


def test_add_noise_preserves_shape():
    """Noise function should not change array shape."""
    from scanImaging.instrument.virtual.virtualISM import add_microscopy_noise

    img = np.random.rand(64, 64).astype(np.float64)
    original_shape = img.shape

    # add_microscopy_noise modifies in-place and returns None
    add_microscopy_noise(img, mean_photons=100, read_noise_e=2, background=0.05)

    assert img.shape == original_shape
