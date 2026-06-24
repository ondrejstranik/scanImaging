"""
End-to-end smoke test of AdaptiveOpticsController (the engine behind main_virtual.py).

Runs each of the 4 optimization algorithms through the real loop() router against a
metric that improves as DM coefficients approach a known optimum, then exercises a
save/load coefficient round-trip. Headless (no GUI), no hardware.

Run: .venv/bin/python scanImaging/tests/smoke_main_virtual.py
"""
import sys
import os
import tempfile
import numpy as np

# Make the conftest helpers + package importable
sys.path.insert(0, os.path.dirname(__file__))
from conftest import MetricImageProvider

from scanImaging.instrument.virtual.virtualDMBmc import VirtualDMBmc
from scanImaging.instrument.adaptiveOpticsSequencer import AdaptiveOpticsController


def metric_at(ctrl, n=15):
    """Mean metric value at the DM's current coefficients, averaged over n noisy frames."""
    fn = ctrl.get_metric_function()
    return float(np.mean([fn(ctrl.image_provider.getImage()) for _ in range(n)]))


def run_algorithm(method):
    indices = [4, 3, 5]
    optimum = [60.0, -40.0, 50.0]   # known best coefficients (nm), inside the scan basin

    dm = VirtualDMBmc()
    dm.connect()
    # scale large enough that the sharpness landscape varies across the full scan range
    provider = MetricImageProvider(dm, optimum_coeffs=optimum,
                                   zernike_indices=indices, scale=20000.0, noise=0.02)

    ctrl = AdaptiveOpticsController()
    ctrl.deformable_mirror = dm
    ctrl.image_provider = provider
    ctrl.verbose = False
    ctrl.print_plot = False
    ctrl.initial_zernike_indices = indices
    ctrl.zernike_initial_coefficients_nm = [0, 0, 0]
    ctrl.zernike_amplitude_scan_nm = [120, 120, 120]
    ctrl.optim_iterations = 2
    ctrl.num_steps_per_mode = 7
    ctrl.optim_method = method

    # SPGD / random-search params
    ctrl.spgd_gain = 0.05
    ctrl.spgd_delta = 15
    ctrl.spgd_iterations = 60
    ctrl.random_search_iterations = 60
    ctrl.random_search_range = 150

    # baseline at zero correction
    dm.set_phase_map_from_zernike(np.zeros(max(indices) + 1))
    dm.display_surface()
    m_before = metric_at(ctrl)

    # drive the generator to completion (this is what the GUI loop does)
    steps = 0
    for _ in ctrl.loop():
        steps += 1
        if steps > 100000:
            raise RuntimeError(f"{method}: loop did not terminate")

    m_after = metric_at(ctrl)
    improved = m_after > m_before
    print(f"  {method:20s} steps={steps:5d}  metric {m_before:.4g} -> {m_after:.4g}  "
          f"{'IMPROVED' if improved else 'no improvement'}")
    dm.disconnect()
    return ctrl, improved


def test_save_load(ctrl):
    path = os.path.join(tempfile.gettempdir(), "smoke_coeffs.npz")
    ctrl.save_coefficients(path)
    saved = np.asarray(ctrl.final_coefficients, dtype=float).copy()

    ctrl2 = AdaptiveOpticsController()
    dm2 = VirtualDMBmc(); dm2.connect()
    ctrl2.deformable_mirror = dm2
    # load_coefficients returns the loaded data and updates zernike_initial_coefficients_nm;
    # it does not set final_coefficients (that is by design — matches GUI usage).
    result = ctrl2.load_coefficients(path)
    loaded = np.asarray(result['coefficients'], dtype=float)
    ok = np.allclose(saved, loaded)
    print(f"  save/load round-trip: {'OK' if ok else 'MISMATCH'} "
          f"(max diff {np.max(np.abs(saved - loaded)):.3e})")
    dm2.disconnect()
    os.remove(path)
    return ok


def main():
    print("=== main_virtual.py engine smoke test (headless) ===")
    print("Running all 4 algorithms through loop():")
    results = {}
    last_ctrl = None
    for method in ['simple_interpolation', 'weighted_fit', 'spgd', 'random_search']:
        ctrl, improved = run_algorithm(method)
        results[method] = improved
        last_ctrl = ctrl

    print("Save/load round-trip:")
    save_ok = test_save_load(last_ctrl)

    print("\n=== Summary ===")
    all_ran = all(v is not None for v in results.values())
    for m, imp in results.items():
        print(f"  {m:20s} ran=YES improved={imp}")
    # All four must run to completion; metric improvement is expected but noise-tolerant.
    improved_count = sum(1 for v in results.values() if v)
    print(f"  algorithms improved metric: {improved_count}/4")
    print(f"  save/load: {'OK' if save_ok else 'FAIL'}")

    ok = all_ran and save_ok and improved_count >= 3
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
