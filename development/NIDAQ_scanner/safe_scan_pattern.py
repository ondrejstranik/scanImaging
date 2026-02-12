import numpy as np
from typing import Tuple

# add hard coded Galvo limits to prevent overdriving
GALVO_V_MIN = -10.0
GALVO_V_MAX = 10.0
# galvo limits: specs for Thorlabs GVS002 galvo scanner sine wave with 250 Hz frequency at full (+/-10 V) amplitude
# vmax=2 pi f A = 2 * pi * 250 Hz * 10 V = 15.707 V/ms
# amax=(2 pi f)^2 A = (2 * pi * 250 Hz)^2 * 10 V = 24,674 V/ms²
GALVO_MAX_DERIVATIVE_V_PER_S = 15707  # max allowed slew rate (V/s)
GALVO_MAX_SECOND_DERIVATIVE_V_PER_S_SQUARED = 2.4674e7  # max allowed acceleration (V/s²) (10 V/ms²)
GALVO_MAX_FREQUENCY_HZ = 250.0         # max allowed frequency (Hz)


def stop_outputs(device_channels: str) -> None:
    """Set outputs to zero (both channels) and clean up."""
    try:
        import nidaqmx
        with nidaqmx.Task() as t:
            t.ao_channels.add_ao_voltage_chan(device_channels, min_val=-10.0, max_val=10.0)
            # many devices accept writing a single sample per channel to set output
            t.write([0.0, 0.0], auto_start=True)
    except Exception as exc:
        print("Warning: could not explicitly write zeros to device:", exc)
    print("Outputs set to 0 V (best-effort).")

def set_voltage(device_channels: str,x,y) -> None:
    if x>10.0 or x<-10.0 or y>10.0 or y<-10.0:
        return
    try:
        import nidaqmx
        with nidaqmx.Task() as t:
            t.ao_channels.add_ao_voltage_chan(device_channels, min_val=-10.0, max_val=10.0)
            # many devices accept writing a single sample per channel to set output
            t.write([x, y], auto_start=True)
    except Exception as exc:
        print("Warning: could not explicitly write to device:", exc)
    print(f"Outputs set to {x},{y} V.")

def periodic_gradient(x, dt):
    return (np.roll(x, -1) - np.roll(x, 1)) / (2 * dt)

def periodic_second_gradient(x, dt):
    return (np.roll(x, -1) - 2*x + np.roll(x, 1)) / (dt**2)


def compute_pattern_metrics(pattern: np.ndarray, rate: float):
    """
    Compute summary statistics for a two-channel pattern, treating the pattern
    as periodic (wrap-around) for derivative and crossing calculations.

    Frequency estimation:
      - threshold = (min+max)/2
      - find all sample indices where the signal crosses the threshold (either direction),
        including a crossing between the last and first sample.
      - median distance between consecutive crossings ~= half-period (samples)
      - period_samples = median_diff * 2
      - freq = rate / period_samples
      - cap freq at frame_rate = rate / N_samples (pattern repetition rate)

    Derivative:
      - compute discrete differences including the wrap (last -> first),
        then scale by sample rate to get V/s and take the max absolute value.
    """

    a = np.asarray(pattern, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("pattern must be shape (N, 2)")

    N = a.shape[0]
    frame_rate = float(rate) / float(max(1, N))  # pattern repetition frequency (Hz)
    res = {}

    for idx, axis_name in enumerate(("X", "Y")):
        axis = a[:, idx].ravel()
        if axis.size == 0:
            res[f"{axis_name}_min"] = 0.0
            res[f"{axis_name}_max"] = 0.0
            res[f"{axis_name}_peak_to_peak"] = 0.0
            res[f"{axis_name}_frequency_hz"] = 0.0
            res[f"{axis_name}_max_abs_derivative"] = 0.0
            res[f"{axis_name}_max_abs_second_derivative"] = 0.0
            continue

        vmin = float(axis.min())
        vmax = float(axis.max())
        res[f"{axis_name}_min"] = vmin
        res[f"{axis_name}_max"] = vmax
        res[f"{axis_name}_peak_to_peak"] = float(vmax - vmin)

        # Numerical derivative (V/s) including wrap-around (last -> first)
        if axis.size >= 2:
            # differences for samples 0..N-2
            dv_linear = np.diff(axis)
            # wrap difference between last and first
            dv_wrap = axis[0] - axis[-1]
            dv_all = np.concatenate((dv_linear, np.atleast_1d(dv_wrap)))
            deriv = np.abs(dv_all) * float(rate)  # deltaV / deltaT where deltaT=1/rate
            res[f"{axis_name}_max_abs_derivative"] = float(np.nanmax(deriv))
        else:
            res[f"{axis_name}_max_abs_derivative"] = 0.0
        # Numerical second derivative (V/s²) including wrap-around
        if axis.size >= 3:  
            d2v_linear = np.diff(axis, n=2)
            # wrap second differences
            d2v_wrap_1 = axis[1] - 2 * axis[0] + axis[-1]
            d2v_wrap_2 = axis[0] - 2 * axis[-1] + axis[-2]
            d2v_all = np.concatenate((d2v_linear, np.atleast_1d(d2v_wrap_1), np.atleast_1d(d2v_wrap_2)))
            second_deriv = np.abs(d2v_all) * (float(rate) ** 2)  # delta²V / deltaT² where deltaT=1/rate
            res[f"{axis_name}_max_abs_second_derivative"] = float(np.nanmax(second_deriv))
        else:
            res[f"{axis_name}_max_abs_second_derivative"] = 0.0

        # Frequency via threshold crossings (robust, circular)
        if vmin == vmax or axis.size < 3:
            res[f"{axis_name}_frequency_hz"] = 0.0
        else:
            thresh = 0.5 * (vmin + vmax)
            s = axis - thresh

            # robust sign array (treat zeros consistently)
            sign = np.sign(s)
            # treat zeros: replace 0 with previous non-zero sign to avoid spurious gaps
            zero_idx = np.where(sign == 0)[0]
            if zero_idx.size:
                # fill zeros by propagating nearest non-zero neighbor (forward fill then backward)
                sign_ffill = sign.copy()
                for i in range(1, sign_ffill.size):
                    if sign_ffill[i] == 0:
                        sign_ffill[i] = sign_ffill[i - 1]
                # backward pass for leading zeros
                for i in range(sign_ffill.size - 2, -1, -1):
                    if sign_ffill[i] == 0:
                        sign_ffill[i] = sign_ffill[i + 1]
                # if still zeros (all zeros), then no crossings
                if np.all(sign_ffill == 0):
                    res[f"{axis_name}_frequency_hz"] = 0.0
                    continue
                sign = sign_ffill

            # detect indices where sign changes between consecutive samples, include wrap
            # sign[i] != sign[(i+1)%N] -> crossing between i and i+1 (report i+1)
            changes = np.nonzero(sign != np.roll(sign, -1))[0]
            if changes.size == 0:
                res[f"{axis_name}_frequency_hz"] = 0.0
            else:
                # crossing indices (report the index of the sample after the change)
                crossings = (changes + 1) % N
                crossings = np.unique(crossings)  # unique and sorted
                if crossings.size < 2:
                    res[f"{axis_name}_frequency_hz"] = 0.0
                else:
                    # compute circular diffs between consecutive crossings
                    diffs = np.diff(crossings)
                    # include wrap gap from last -> first
                    wrap_gap = (crossings[0] + N) - crossings[-1]
                    all_gaps = np.concatenate((diffs.astype(float), np.atleast_1d(float(wrap_gap))))
                    # median gap in samples is median distance between consecutive crossings
                    median_crossing_gap = float(np.median(all_gaps))
                    if median_crossing_gap <= 0:
                        res[f"{axis_name}_frequency_hz"] = 0.0
                    else:
                        # consecutive crossings are ~ half-period -> full period = median_crossing_gap * 2
                        period_samples = median_crossing_gap * 2.0
                        freq = float(rate) / float(period_samples)
                        # cap to Nyquist (prevent unrealistic > Nyquist values)
                        freq = min(freq, float(rate) / 2.0)
                        res[f"{axis_name}_frequency_hz"] = float(freq)

    return res


def print_pattern_metrics(pattern: np.ndarray, rate: float):
    m = compute_pattern_metrics(pattern, rate)
    # pretty print
    print("Pattern metrics:")
    print(f"  X: min={m['X_min']:.4g}  max={m['X_max']:.4g}  p2p={m['X_peak_to_peak']:.4g}  "
          f"freq={m['X_frequency_hz']:.3f} Hz  max|dV/dt|={m['X_max_abs_derivative']:.4g} V/s  max|d²V/dt²|={m['X_max_abs_second_derivative']:.4g} V/s²")
    print(f"  Y: min={m['Y_min']:.4g}  max={m['Y_max']:.4g}  p2p={m['Y_peak_to_peak']:.4g}  "
          f"freq={m['Y_frequency_hz']:.3f} Hz  max|dV/dt|={m['Y_max_abs_derivative']:.4g} V/s  max|d²V/dt²|={m['Y_max_abs_second_derivative']:.4g} V/s²")
    print(f"time to complete pattern: {len(pattern)/rate:.3f} s")
    return m

def check_if_exceeds_limits(pattern: np.ndarray, rate: float) -> bool:
    """Check whether the pattern exceeds galvo limits; return True if it does."""
    m = compute_pattern_metrics(pattern, rate)
    exceeds = False

    for axis_name in ("X", "Y"):
        vmin = m[f"{axis_name}_min"]
        vmax = m[f"{axis_name}_max"]
        max_deriv = m[f"{axis_name}_max_abs_derivative"]
        freq = m[f"{axis_name}_frequency_hz"]

        if vmin < GALVO_V_MIN or vmax > GALVO_V_MAX:
            print(f"Error: {axis_name} channel exceeds voltage limits ({vmin:.2f} V .. {vmax:.2f} V).")
            exceeds = True
        if max_deriv > GALVO_MAX_DERIVATIVE_V_PER_S:
            print(f"Error: {axis_name} channel exceeds max derivative limit ({max_deriv:.2f} V/s).")
            exceeds = True
        if freq > GALVO_MAX_FREQUENCY_HZ:
            print(f"Error: {axis_name} channel exceeds max frequency limit ({freq:.2f} Hz).")
            exceeds = True
        if m[f"{axis_name}_max_abs_second_derivative"] > GALVO_MAX_SECOND_DERIVATIVE_V_PER_S_SQUARED:
            print(f"Error: {axis_name} channel exceeds max second derivative limit ({m[f'{axis_name}_max_abs_second_derivative']:.2f} V/s²).")
            exceeds = True  

    return exceeds

def quintic_hermite(t: np.ndarray, T: float, x0: float, x1: float, v0: float, v1: float) -> np.ndarray:
    """
    Quintic polynomial on [0,T] that satisfies:
      x(0) = x0, x(T) = x1
      x'(0) = v0, x'(T) = v1
      x''(0) = x''(T) = 0

    Parameters
    - t: sample times (0 <= t < T)
    - T: duration of the interval
    - x0, x1: endpoint positions
    - v0, v1: endpoint velocities

    Returns:
      x(t) evaluated at the times in `t`.
    """
    if T == 0:
        return np.full_like(t, x0, dtype=float)

    s = t / T

    a0 = x0
    a1 = v0 * T
    a2 = 0.0
    a3 = 10.0 * (x1 - x0) - (6.0 * v0 + 4.0 * v1) * T
    a4 = -15.0 * (x1 - x0) + (8.0 * v0 + 7.0 * v1) * T
    a5 = 6.0 * (x1 - x0) - (3.0 * v0 + 3.0 * v1) * T

    return a0 + a1 * s + a3 * s**3 + a4 * s**4 + a5 * s**5


def smooth_step_cos(s: np.ndarray) -> np.ndarray:
    """
    Smooth cosine step from 0 -> 1 with zero slope at both ends.

    Input s expected in [0,1].
    """
    return 0.5 * (1.0 - np.cos(np.pi * s))


def _safe_num_points(n: int) -> int:
    """Ensure we always return at least one sample (prevents empty arrays)."""
    return max(1, int(n))


#fs = 100_000
# 50kHz to 150 kHz -- above 150kHz not needed. Safe for DAQ (USB-6421) 200 kS/s 
#nx = 128 to 2048
#ny = 64  to 1024
#line_rate = 10 to 20
#x_amp = 1.5 to 2.5
#y_amp = 1.5 to 2.5
#x_flyback_frac = 0.15 to 0.25
#bidirectional = False (first tests)


def safe_scan_pattern(
    fs: float = 100_000,
    ny: int = 128,
    line_rate: float = 20.0,
    x_amp: float = 2.0,
    y_amp: float = 2.0,
    x_flyback_frac: float = 0.25,
    y_flyback_frac: float = 0.05,
    bidirectional: bool = True,
    split_xframefly:bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 'safe' scan trajectory for galvanometer mirrors.

    Behavior summary:
    - Each scan line is composed of an active segment (constant-velocity sweep)
      and a flyback segment that transitions smoothly to the next line.
    - X axis: constant-velocity active scan, smooth quintic flyback.
    - Y axis: holds value during active scan, smoothly steps during flyback.

    Parameters:
    - fs: sampling frequency (Hz)
    - nx: number of samples per active line (kept for API compatibility; not used directly)
    - ny: number of scan lines per frame (vertical lines)
    - line_rate: number of lines per second (Hz)
    - x_amp/y_amp: amplitude for X and Y (voltages). Scan covers [-amp, +amp].
    - x_flyback_frac: fraction of line period reserved for X flyback
    - y_flyback_frac: fraction of frame period reserved for Y flyback at frame end
    - bidirectional: if True alternate left-to-right / right-to-left scanning
    - split_xframefly: if True, in bidirectional mode split the final line flyback
      into an X flyback segment and a constant segment to fill the Y flyback time.

    Returns:
    - t: time vector (seconds)
    - x: X axis waveform (voltage)
    - y: Y axis waveform (voltage)
    """
    # Derived timings
    T_line = 1.0 / float(line_rate)
    T_fly = float(x_flyback_frac) * T_line
    T_act = T_line - T_fly
    if T_act <= 0.0:
        raise ValueError("x_flyback_frac too large; active segment duration <= 0")

    # Y flyback should be relative to the entire frame duration (ny * line time)
    T_frame = ny * T_line
    T_yfly = float(y_flyback_frac) * T_frame

    # sample counts (ensure at least one sample)
    n_fly = _safe_num_points(int(round(fs * T_fly)))
    n_act = _safe_num_points(int(round(fs * T_act)))
    n_yfly = _safe_num_points(int(round(fs * T_yfly)))

    # constant X velocity during active scan (magnitude)
    vx = 2.0 * x_amp / T_act

    # Y positions for each line (equally spaced)
    y_vals = np.linspace(-y_amp, y_amp, ny)

    # This ensures that the trigger pulses at the start of each line and frame are 
    # included in the pattern, even if the flyback time is very short. 
    # The line trigger will be high for the first few samples of the active segment, 
    # and the frame trigger will be high for the first few samples of the first line.
    # If hardware allows very short pulses, set LINE_PULSE_SAMPLES and FRAME_PULSE_SAMPLES to 1 for minimal trigger duration.
    LINE_PULSE_SAMPLES  = max(1, int(0.00005 * fs))   # 50 µs
    FRAME_PULSE_SAMPLES = max(1, int(0.0001  * fs))   # 100 µs


    x_segments = []
    y_segments = []
    pixel_segments = []
    line_segments = []
    frame_segments = []

    for i in range(ny):
        y0 = float(y_vals[i])
        # next Y; wrap to first element for final step so frame is continuous
        y1 = float(y_vals[i + 1]) if i < ny - 1 else float(y_vals[0])

        # Active segment: linear sweep across X
        t_act = np.linspace(0.0, T_act, n_act, endpoint=False, dtype=float)
        if not bidirectional:
            x_act = -x_amp + vx * t_act
        else:
            # alternate sweep direction per line
            if (i % 2) == 0:
                x_act = -x_amp + vx * t_act
            else:
                x_act = +x_amp - vx * t_act
        y_act = np.full_like(x_act, y0, dtype=float)
        pixel_act = np.ones(n_act, dtype=np.uint8)
        line_act = np.zeros(n_act, dtype=np.uint8)
        frame_act = np.zeros(n_act, dtype=np.uint8)

        # Line trigger at first active pixel
        line_act[:LINE_PULSE_SAMPLES] = 1
        if i == 0:
            frame_act[:FRAME_PULSE_SAMPLES] = 1


        # Flyback segment: smooth transition to next X start and next Y
        # For the final line we may use T_yfly (frame-level Y flyback)
        fly_T = T_fly if i < ny - 1 else max(T_fly, T_yfly)
        fly_n = n_fly if i < ny - 1 else max(n_fly, n_yfly)
        t_fly = np.linspace(0.0, fly_T, fly_n, endpoint=False, dtype=float)

        if not bidirectional:
            # unidirectional case: always fly from +A -> -A
            x_f0, x_f1 = +x_amp, -x_amp
            v0 = vx
            v1 = vx
        else:
            # bidirectional: choose endpoints and endpoint velocities
            if (i % 2) == 0:
                # current active swept left->right; next active will sweep right->left
                x_f0 = +x_amp
                # if not last line, next start is +x_amp (since next active will start at +A)
                x_f1 = +x_amp if i < ny - 1 else -x_amp
                v0 = vx
                # decide final velocity sign: next active may start moving left (-vx) unless final wrap
                v1 = -vx if i < ny - 1 else vx
            else:
                # current active swept right->left
                x_f0 = -x_amp
                x_f1 = -x_amp
                v0 = -vx
                v1 = vx

        # Use quintic for the flyback to get zero endpoint accelerations (smooth)
        if  not bidirectional or not (split_xframefly or (i < ny - 1)):
            x_fly = quintic_hermite(t_fly, fly_T, x_f0, x_f1, v0, v1)
        else:
            t_xfly = np.linspace(0.0, T_fly, n_fly, endpoint=False, dtype=float)
            x_fly = quintic_hermite(t_xfly, T_fly, x_f0, x_f1, v0, v1)
            # now add constant segment at max/min X to fill remaining Y flyback time
            n_const = fly_n - n_fly
            if n_const > 0:
                ind=np.argmax(abs(x_fly))
                xmax = x_fly[ind]
                x_const = np.full(n_const, xmax, dtype=float)
                x_fly = np.concatenate([x_fly[:ind], x_const, x_fly[ind:]])


        # Y smoothly steps between lines during flyback
        s = np.linspace(0.0, 1.0, fly_n, endpoint=False, dtype=float)
        y_fly = y0 + (y1 - y0) * smooth_step_cos(s)

        #add gating signals (1 during active, 0 during flyback)
        pixel_fly = np.zeros(fly_n, dtype=np.uint8)
        line_fly = np.zeros(fly_n, dtype=np.uint8)
        frame_fly = np.zeros(fly_n, dtype=np.uint8)

        # Concatenate active + flyback for this line
        x_segments.append(np.concatenate([x_act, x_fly]))
        y_segments.append(np.concatenate([y_act, y_fly]))
        pixel_segments.append(np.concatenate([pixel_act, pixel_fly]))
        line_segments.append(np.concatenate([line_act, line_fly]))
        frame_segments.append(np.concatenate([frame_act, frame_fly]))

    # Final concatenation across all lines
    x = np.concatenate(x_segments)
    y = np.concatenate(y_segments)
    pixel_gate = np.concatenate(pixel_segments)
    line_trig = np.concatenate(line_segments)
    frame_trig = np.concatenate(frame_segments)

    t = np.arange(len(x), dtype=float) / float(fs)

    return t, x, y, pixel_gate, line_trig, frame_trig



# Quick demo when run as script ------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t, x, y, pixel_gate, line_trig, frame_trig = safe_scan_pattern(
        fs=100_000,
        nx=256,
        ny=10,
        line_rate=20,
        x_amp=2.0,
        y_amp=2.0,
        x_flyback_frac=0.1,
        y_flyback_frac=0.5/11,
        bidirectional=True,
    )

    plt.figure(figsize=(10, 4))
    plt.plot(t, x, label="X")
    plt.plot(t, y, label="Y", alpha=0.7)
    #add also the line triggers
    plt.plot(t, line_trig * 3.0 - 2.0, label="Line Trigger", linestyle='--', color='gray')
    plt.plot(t, frame_trig * 3.0 - 2.0, label="Frame Trigger", linestyle='--', color='black')
    #add also the pixel gate
    plt.plot(t, pixel_gate * 3.0 - 2.0, label="Pixel Gate", linestyle='--', color='orange')
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title("Safe Galvo Scan (quintic flyback)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.plot(x, y, linewidth=0.5)
    plt.axis("equal")
    plt.xlabel("X [V]")
    plt.ylabel("Y [V]")
    plt.title("XY Trajectory")
    plt.tight_layout()
    plt.show()

