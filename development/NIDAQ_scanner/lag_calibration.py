import numpy as np


def estimate_even_odd_shift(frame: np.ndarray) -> float:
    """
    Estimate horizontal displacement of odd lines relative to even lines.

    Cross-correlates the column-averaged even sub-image with the column-averaged
    odd sub-image. This is robust to row-to-row content variation because averaging
    projects out y-variation, leaving only the x-profile shift.

    Returns:
        float: positive = odd appears LEFT of even (under-correction of odd, or
               over-correction of even). Total displacement in pixels.
               Correct with: scanner_lag_samples_odd += round(result * samples_per_pixel)

    Note: measures the SUM of even rightward and odd leftward displacements.
          Best used after global lag is approximately set, to fix the differential.
    """
    ny, nx = frame.shape
    if ny < 4:
        return 0.0

    even_avg = frame[0::2, :].mean(axis=0).astype(np.float64)
    odd_avg  = frame[1::2, :].mean(axis=0).astype(np.float64)
    even_avg -= even_avg.mean()
    odd_avg  -= odd_avg.mean()

    if even_avg.std() < 1e-6 or odd_avg.std() < 1e-6:
        return 0.0

    corr = np.correlate(even_avg, odd_avg, mode='full')
    peak_idx = int(np.argmax(corr))
    shift = float(peak_idx - (nx - 1))

    # Parabolic sub-pixel refinement
    if 0 < peak_idx < len(corr) - 1:
        y0, y1, y2 = corr[peak_idx - 1], corr[peak_idx], corr[peak_idx + 1]
        denom = 2 * y1 - y0 - y2
        if abs(denom) > 1e-10:
            shift += (y2 - y0) / (2 * denom)

    return float(shift)


def reconstruct_from_raw(
    raw_lines: np.ndarray,
    nx: int,
    bidirectional: bool = True,
    global_lag: int = 0,
    even_lag: int = 0,
    odd_lag: int = 0,
    xy_scale: float = 1.0,
) -> np.ndarray:
    """
    Reconstruct a frame from raw per-line detector data at sample-level precision.

    Args:
        raw_lines: (ny, max_raw_samples) array — unmodified detector samples per line,
                   aligned to the line trigger, before any lag correction or reversal.
        nx: output pixels per line
        bidirectional: True if odd lines were scanned in reverse direction
        global_lag, even_lag, odd_lag: sample offsets (same meaning as online params)
        xy_scale: horizontal zoom applied after reconstruction

    Returns:
        (ny, nx) float32 frame
    """
    ny, max_raw = raw_lines.shape
    blocking = max_raw // nx
    if blocking < 1:
        raise ValueError(f"raw_lines has too few samples ({max_raw}) for nx={nx}")
    usable = blocking * nx
    frame = np.zeros((ny, nx), dtype=np.float32)

    for i in range(ny):
        lag = global_lag + (even_lag if i % 2 == 0 else odd_lag)
        raw = raw_lines[i]

        if lag >= 0:
            end = lag + usable
            if end <= max_raw:
                line = raw[lag:end].copy()
            else:
                available = max(0, max_raw - lag)
                line = np.zeros(usable, dtype=np.float32)
                if available > 0:
                    line[:available] = raw[lag:lag + available]
        else:
            skip = -lag
            line = np.zeros(usable, dtype=np.float32)
            valid = min(usable - skip, max_raw)
            if valid > 0:
                line[skip:skip + valid] = raw[:valid]

        if bidirectional and i % 2 == 1:
            line = line[::-1]

        frame[i] = line.reshape(nx, blocking).mean(axis=1)

    if abs(xy_scale - 1.0) > 1e-6:
        from scipy.ndimage import zoom as nd_zoom
        frame = nd_zoom(frame, [1.0, xy_scale], order=1)
        if frame.shape[1] > nx:
            frame = frame[:, :nx]
        elif frame.shape[1] < nx:
            frame = np.pad(frame, ((0, 0), (0, nx - frame.shape[1])))

    return frame


def shift_rows(
    frame: np.ndarray,
    global_px: float = 0.0,
    even_px: float = 0.0,
    odd_px: float = 0.0,
) -> np.ndarray:
    """
    Pixel-level row shifts: fallback when raw line data is not available.

    Mimics the effect of changing lag parameters on the reconstructed frame:
    - Even rows shift LEFT by (global_px + even_px) pixels
    - Odd rows shift RIGHT by (global_px + odd_px) pixels
    (This matches the display effect of increasing lag in the online tool.)

    Positive global_px, even_px, odd_px all correspond to "more correction applied".
    """
    from scipy.ndimage import shift as nd_shift
    out = np.empty_like(frame, dtype=np.float32)
    for i in range(frame.shape[0]):
        row = frame[i].astype(np.float32)
        if i % 2 == 0:
            s = -(global_px + even_px)   # left = negative scipy shift
        else:
            s = +(global_px + odd_px)    # right = positive scipy shift
        if abs(s) < 1e-9:
            out[i] = row
        else:
            out[i] = nd_shift(row, s, mode='constant', cval=0.0, order=1)
    return out
