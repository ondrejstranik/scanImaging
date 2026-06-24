'''
Rough lifetime fitting for FLIM data.

This module provides a *coarse* sanity-check fit only: it fits a single
exponential to the whole-image time histogram (summed over all pixels), after
excluding the region around the IRF peak. Its purpose is to confirm the signal
falls into the expected range of tau values - NOT to do pixel-resolved analysis.

Pixel-resolved / multi-exponential FLIM fitting is handed off to the separate
Julia package at ~/projects/microscopy/flim_analysis/ (see project notes).
'''
import numpy as np


def rough_single_exp_fit(time_axis, histogram, peak_offset_ns=0.3,
                         fit_window_ns=None, fit_background=True):
    '''Rough single-exponential lifetime estimate from a whole-image FLIM histogram.

    The fit region starts at (histogram peak + peak_offset_ns) to skip the IRF
    rise/peak, and a single exponential (optionally + constant background) is fit to
    the remaining decay tail. For a multi-exponential sample this returns an estimate
    biased towards the longer/dominant component - good enough to check the lifetime
    is in the expected ballpark.

    Args:
        time_axis: 1D array of time-bin centres [ns]
        histogram: 1D array of counts (summed over all pixels/channels), same length
        peak_offset_ns: how far past the peak to start the fit (excludes the IRF) [ns]
        fit_window_ns: if set, only fit up to (t_start + fit_window_ns) [ns]
        fit_background: if True, fit a constant background term as well

    Returns:
        dict with keys:
            tau_ns, tau_stderr_ns, amplitude, background,
            t_peak_ns, t_start_ns, t_fit, y_fit, success, message
    '''
    from scipy.optimize import curve_fit

    time_axis = np.asarray(time_axis, dtype=float)
    histogram = np.asarray(histogram, dtype=float)

    result = {
        'tau_ns': np.nan, 'tau_stderr_ns': np.nan,
        'amplitude': np.nan, 'background': np.nan,
        't_peak_ns': np.nan, 't_start_ns': np.nan,
        't_fit': None, 'y_fit': None,
        'success': False, 'message': '',
    }

    if time_axis.size != histogram.size or time_axis.size < 4:
        result['message'] = 'time_axis and histogram must be 1D arrays of equal length (>=4)'
        return result

    # locate the IRF peak and start the fit just after it
    peak_idx = int(np.argmax(histogram))
    t_peak = time_axis[peak_idx]
    t_start = t_peak + peak_offset_ns
    result['t_peak_ns'] = t_peak
    result['t_start_ns'] = t_start

    mask = time_axis >= t_start
    if fit_window_ns is not None:
        mask &= time_axis <= (t_start + fit_window_ns)
    t = time_axis[mask]
    y = histogram[mask]

    if t.size < 3 or y.max() <= 0:
        result['message'] = 'not enough points past the IRF peak to fit'
        return result

    t0 = t[0]
    # initial guesses
    bg0 = max(float(y.min()), 0.0) if fit_background else 0.0
    A0 = max(float(y[0] - bg0), 1.0)
    # mean of a shifted exponential equals tau -> use as tau seed
    w = np.clip(y - bg0, 0, None)
    tau0 = float(np.sum((t - t0) * w) / np.sum(w)) if np.sum(w) > 0 else 1.0
    tau0 = min(max(tau0, 0.05), 50.0)

    try:
        if fit_background:
            def model(tt, A, tau, bg):
                return A * np.exp(-(tt - t0) / tau) + bg
            p0 = [A0, tau0, bg0]
            bounds = ([0, 0.01, 0], [np.inf, 100.0, np.inf])
            popt, pcov = curve_fit(model, t, y, p0=p0, bounds=bounds, maxfev=20000)
            A, tau, bg = popt
        else:
            def model(tt, A, tau):
                return A * np.exp(-(tt - t0) / tau)
            p0 = [A0, tau0]
            bounds = ([0, 0.01], [np.inf, 100.0])
            popt, pcov = curve_fit(model, t, y, p0=p0, bounds=bounds, maxfev=20000)
            A, tau = popt
            bg = 0.0

        perr = np.sqrt(np.diag(pcov))
        tau_idx = 1
        result.update({
            'tau_ns': float(tau),
            'tau_stderr_ns': float(perr[tau_idx]) if np.all(np.isfinite(perr)) else np.nan,
            'amplitude': float(A),
            'background': float(bg),
            't_fit': t,
            'y_fit': model(t, *popt),
            'success': True,
            'message': 'ok',
        })
    except Exception as e:
        result['message'] = f'fit failed: {e}'

    return result
