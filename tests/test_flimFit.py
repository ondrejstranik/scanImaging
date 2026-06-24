''' tests for the rough single-exponential FLIM lifetime fit '''
import numpy as np
import pytest

from scanImaging.algorithm.flimFit import rough_single_exp_fit
from scanImaging.algorithm.flimData import FlimData


def _simulate_histogram(seed=0, n=200_000, nbins=128):
    ''' simulate a whole-image FLIM histogram with the virtual scanner's decay model '''
    np.random.seed(seed)
    from scanImaging.instrument.virtual.virtualScannerBH import VirtualBHScanner
    sc = VirtualBHScanner(name='fitTest')
    sc.setLifetimeUniform(2.8, 0.6, 0.7)   # long=2.8 ns, short=0.6 ns, 70% comp1
    sc.setIRF(0.15, 0.0)
    raw = sc.generatePhotonMicroTimes(np.zeros(n, dtype=int))
    span = float(sc.timeRange[1] - sc.timeRange[0])
    delay = (sc.timeSize - raw.astype('int64')) * span / sc.timeSize  # processor inversion
    counts, edges = np.histogram(delay, bins=nbins, range=(0, span))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts, span


def test_rough_single_exp_in_expected_range():
    t, y, _ = _simulate_histogram()
    res = rough_single_exp_fit(t, y, peak_offset_ns=0.6)
    assert res['success'], res['message']
    # single-exp fit of the tail is biased towards the long component (2.8 ns)
    assert 1.5 < res['tau_ns'] < 4.0, f"tau={res['tau_ns']}"
    assert res['t_start_ns'] > res['t_peak_ns']


def test_rough_fit_via_flimdata_method():
    t, y, span = _simulate_histogram(seed=1)
    nbins = y.size
    cube = np.zeros((nbins, 1, 1, 1), dtype=float)
    cube[:, 0, 0, 0] = y
    fd = FlimData(cube, timeRange=np.array([0, span]))
    res = fd.roughLifetimeFit(peak_offset_ns=0.6)
    assert res['success'], res['message']
    assert 1.5 < res['tau_ns'] < 4.0, f"tau={res['tau_ns']}"


def test_rough_fit_rejects_bad_input():
    res = rough_single_exp_fit([0, 1, 2], [1, 2])   # mismatched lengths
    assert not res['success']
    assert 'equal length' in res['message']
