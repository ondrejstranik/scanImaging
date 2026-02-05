import numpy as np
from zernike import RZern

def zernike_phase_map(osa_coefficients, grid_x, grid_y, effective_aperture_radius=1.0):
    """
    Generate a wavefront phase map from OSA Zernike coefficients.

    osa_coefficients : list (OSA j indices start at 0)
    grid_x, grid_y : ints -> produce a square grid Nx by Ny
    effective_aperture_radius : radius in normalized units (<=0 means full aperture -> 1.0)
    """
    # ensure aperture radius in normalized units
    if effective_aperture_radius <= 0:
        effective_aperture_radius = 1.0

    # --- Create Zernike basis up to needed radial order ---
    max_j = len(osa_coefficients)
    n = 0
    while (n + 1) * (n + 2) // 2 < max_j:
        n += 1
    cart = RZern(n)

    # --- Create Cartesian grid on [-1, 1] x [-1, 1] ---
    x = np.linspace(-1.0, 1.0, grid_x)
    y = np.linspace(-1.0, 1.0, grid_y)
    xv, yv = np.meshgrid(x, y)

    # Precompute polar grid/mask internally
    cart.make_cart_grid(xv, yv)

    # Build coefficient vector in the package's OSA order (1-based -> index j-1)
    c = np.zeros(cart.nk, dtype=float)
    for j in range(len(osa_coefficients)):
        if j-1 < c.size and j >= 1:
            c[j-1] = osa_coefficients[j]

    # Evaluate the wavefront on the grid
    phase_map = cart.eval_grid(c, matrix=True)

    # Mask points outside the circular pupil (set to 0 or np.nan as desired)
    rho = np.sqrt(xv**2 + yv**2)
    mask = rho <= effective_aperture_radius
    phase_map = np.where(mask, phase_map, 0.0)

    return phase_map


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    coeff_nm    = [0,0,0,0, 120.0, 0.0]
    #rho = np.sqrt(xv**2 + yv**2)
    #Phi_masked = np.where(rho <= 1.0, Phi, np.nan)
    phi_map = zernike_phase_map(coeff_nm, 20, 20)
    # --- Plot in nanometers ---
    plt.figure(figsize=(6,5))
    im = plt.imshow(phi_map, origin='lower', extent=(-1,1,-1,1), cmap='RdBu')
    plt.colorbar(im, label='Wavefront [nm]')
    plt.title('Wavefront from OSA Coefficients (RZern)')
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()
