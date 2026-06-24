"""
Verification harness for the virtual FLIM lifetime simulation.

Confirms that the VirtualBHScanner produces photon microtimes drawn from the
intended bi-exponential decay (+ Gaussian IRF) in the B&H reverse start-stop
convention, and that the BHScannerProcessor inversion recovers a forward-running
decay. This is the ground-truth check for future lifetime-fitting algorithms.

Checks:
  1. Mean photon arrival time matches the analytic mixture mean (no IRF).
  2. Gaussian IRF shifts the mean arrival time by the IRF offset.
  3. Processor-style inversion yields a forward (decaying) histogram.
  4. Two-region sample (foreground/background) recovers distinct lifetimes,
     driven through VirtualISM's region mapping.
  5. A bi-exponential curve fit recovers the assigned lifetimes (informational).

Run: .venv/bin/python scanImaging/tests/verify_flim_lifetime.py
"""
import sys
import numpy as np

from scanImaging.instrument.virtual.virtualScannerBH import VirtualBHScanner
from scanImaging.instrument.virtual.virtualISM import VirtualISM

N = 300_000          # photons per test
RTOL = 0.03          # 3% tolerance on mean arrival time at this N


def raw_to_delay_ns(raw, scanner):
    """Replicate the processor inversion: raw microtime -> physical delay [ns]."""
    span = float(scanner.timeRange[1] - scanner.timeRange[0])
    photon_delay_bins = scanner.timeSize - raw.astype('int64')   # reverse start-stop
    return photon_delay_bins * span / scanner.timeSize


def mixture_mean(frac, tau1, tau2, irf_offset=0.0):
    """Analytic mean arrival time of the photon-fraction mixture."""
    return frac * tau1 + (1.0 - frac) * tau2 + irf_offset


def make_scanner():
    sc = VirtualBHScanner(name='flimTest')
    return sc


def test_mean_arrival_no_irf():
    sc = make_scanner()
    frac, tau1, tau2 = 0.7, 2.8, 0.6
    sc.setLifetimeUniform(tau1, tau2, frac)
    sc.setIRF(0.0, 0.0)
    raw = sc.generatePhotonMicroTimes(np.zeros(N, dtype=int))  # all at pixel 0 (foreground)
    delay = raw_to_delay_ns(raw, sc)
    measured, expected = delay.mean(), mixture_mean(frac, tau1, tau2)
    ok = abs(measured - expected) / expected < RTOL
    print(f"  [1] mean arrival (no IRF): measured={measured:.3f} ns  "
          f"expected={expected:.3f} ns  {'OK' if ok else 'FAIL'}")
    return ok


def test_irf_shift():
    sc = make_scanner()
    frac, tau1, tau2, offset = 0.7, 2.8, 0.6, 1.5
    sc.setLifetimeUniform(tau1, tau2, frac)
    sc.setIRF(0.15, offset)
    raw = sc.generatePhotonMicroTimes(np.zeros(N, dtype=int))
    delay = raw_to_delay_ns(raw, sc)
    measured, expected = delay.mean(), mixture_mean(frac, tau1, tau2, offset)
    ok = abs(measured - expected) / expected < RTOL
    print(f"  [2] IRF offset shift:      measured={measured:.3f} ns  "
          f"expected={expected:.3f} ns  {'OK' if ok else 'FAIL'}")
    return ok


def test_forward_decay():
    sc = make_scanner()
    sc.setLifetimeUniform(2.8, 0.6, 0.7)
    sc.setIRF(0.0, 0.0)
    raw = sc.generatePhotonMicroTimes(np.zeros(N, dtype=int))
    delay = raw_to_delay_ns(raw, sc)
    span = float(sc.timeRange[1] - sc.timeRange[0])
    counts, _ = np.histogram(delay, bins=32, range=(0, span))
    # forward-running decay: early bins (after the rise) dominate late bins
    ok = counts[0] > counts[-1] and counts[:3].sum() > counts[-3:].sum()
    print(f"  [3] forward decay shape:   counts[0]={counts[0]} > counts[-1]={counts[-1]}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def test_two_regions():
    """Drive VirtualISM region mapping: bright square = foreground, rest = background."""
    sc = make_scanner()
    ism = VirtualISM(name='flimISM')
    ism.virtualScanner = sc
    # synthetic structure: a bright square in the centre
    H, W = sc.imageSize
    base = np.zeros((H, W), dtype=float)
    base[H//4:3*H//4, W//4:3*W//4] = 1.0
    ism.base_image = base
    ism.flim.update({
        'enabled': True, 'regionThreshold': 0.5,
        'fg_tau1': 3.2, 'fg_tau2': 0.8, 'fg_frac': 0.8,
        'bg_tau1': 1.0, 'bg_tau2': 0.2, 'bg_frac': 0.5,
        'irf_sigma': 0.0, 'irf_offset': 0.0,
    })
    ism._applyFlimToScanner()

    # linear indices (first copy of the doubled extended layout)
    def lin(y, x):
        return y * sc.scanSize[1] + x
    fg_idx = np.full(N, lin(H//2, W//2), dtype=int)        # centre = foreground
    bg_idx = np.full(N, lin(2, 2), dtype=int)              # corner = background

    fg_mean = raw_to_delay_ns(sc.generatePhotonMicroTimes(fg_idx), sc).mean()
    bg_mean = raw_to_delay_ns(sc.generatePhotonMicroTimes(bg_idx), sc).mean()
    fg_exp = mixture_mean(0.8, 3.2, 0.8)
    bg_exp = mixture_mean(0.5, 1.0, 0.2)
    ok_fg = abs(fg_mean - fg_exp) / fg_exp < RTOL
    ok_bg = abs(bg_mean - bg_exp) / bg_exp < RTOL
    ok_sep = fg_mean > bg_mean
    print(f"  [4] region map: fg mean={fg_mean:.3f} (exp {fg_exp:.3f}), "
          f"bg mean={bg_mean:.3f} (exp {bg_exp:.3f}), fg>bg={ok_sep}  "
          f"{'OK' if (ok_fg and ok_bg and ok_sep) else 'FAIL'}")
    return ok_fg and ok_bg and ok_sep


def test_biexp_fit():
    """Informational: fit a bi-exponential to the recovered decay."""
    try:
        from scipy.optimize import curve_fit
    except Exception as e:
        print(f"  [5] bi-exp fit: scipy unavailable ({e}) - skipped")
        return True
    sc = make_scanner()
    frac, tau1, tau2 = 0.7, 2.8, 0.6
    sc.setLifetimeUniform(tau1, tau2, frac)
    sc.setIRF(0.0, 0.0)
    delay = raw_to_delay_ns(sc.generatePhotonMicroTimes(np.zeros(N, dtype=int)), sc)
    span = float(sc.timeRange[1] - sc.timeRange[0])
    counts, edges = np.histogram(delay, bins=128, range=(0, span))
    t = 0.5 * (edges[:-1] + edges[1:])

    def model(t, A, f, t1, t2):
        return A * (f / t1 * np.exp(-t / t1) + (1 - f) / t2 * np.exp(-t / t2))

    try:
        p0 = [counts.sum() * span / 128, 0.6, 2.5, 0.5]
        popt, _ = curve_fit(model, t, counts, p0=p0, maxfev=20000,
                            bounds=([0, 0, 0.05, 0.05], [np.inf, 1, 20, 20]))
        f_fit, t1_fit, t2_fit = popt[1], max(popt[2], popt[3]), min(popt[2], popt[3])
        print(f"  [5] bi-exp fit: tau1={t1_fit:.2f} (true {tau1}), "
              f"tau2={t2_fit:.2f} (true {tau2}), frac={f_fit:.2f} (true {frac}) [informational]")
    except Exception as e:
        print(f"  [5] bi-exp fit: fit did not converge ({e}) [informational]")
    return True


def main():
    print("=== Virtual FLIM lifetime simulation verification ===")
    results = [
        test_mean_arrival_no_irf(),
        test_irf_shift(),
        test_forward_decay(),
        test_two_regions(),
        test_biexp_fit(),
    ]
    ok = all(results)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}  ({sum(results)}/{len(results)} checks)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
