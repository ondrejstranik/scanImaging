import numpy as np
from zernike import RZern

def zernike_phase_map(osa_coefficients,grid_x,grid_y,effective_aperture_radius=1.0):
    """
    Generate a wavefront phase map from OSA Zernike coefficients.

    Parameters:
    -----------
    osa_coefficients : list or array-like
        List of Zernike coefficients in OSA indexing (j=1,2,...).
    grid_x : 2D array
        X-coordinates of the grid (should be normalized to unit circle).
    grid_y : 2D array
        Y-coordinates of the grid (should be normalized to unit circle).

    Returns:
    --------
    phase_map : 2D array
        Wavefront phase map evaluated on the provided grid.
    """
    # --- Inputs: OSA indices and coefficients in nm ---
    osa_indices = [i for i in range(len(osa_coefficients))]  # OSA j indices are 1-based
    coeff_nm    = osa_coefficients

    # --- Create Zernike basis up to needed radial order ---
    max_j = max(osa_indices)
    # Estimate max radial order n needed for given max_j using OSA indexing rules
    n = 0
    while (n+1)*(n+2)//2 < max_j:
        n += 1
    cart = RZern(n)  # max radial order n

    # --- Create Cartesian grid on [-1, 1] x [-1, 1] ---
    x = np.linspace(-1.0, 1.0, grid_x)
    y = np.linspace(-1.0, 1.0, grid_y)
    xv, yv = np.meshgrid(x, y)

    # Precompute polar grid/mask internally
    cart.make_cart_grid(xv, yv)  # matches docs usage

    # --- Build coefficient vector in the package's OSA order ---
    c = np.zeros(cart.nk, dtype=float)  # number of supported modes
    for j, coeff in zip(osa_indices, coeff_nm):
        c[j - 1] = coeff

    # --- Evaluate the wavefront on the grid ---
    phase_map = cart.eval_grid(c, matrix=True)  # returns a 2D array on the same xv,yv grid

    return phase_map


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    coeff_nm    = [0,0,0,0,120.0, -50.0, 30.0]
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
