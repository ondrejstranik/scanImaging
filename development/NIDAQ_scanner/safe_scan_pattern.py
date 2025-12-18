import numpy as np
from typing import Tuple


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
    nx: int = 256,
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
        line_act[0] = 1
        if i == 0:
            frame_act[0] = 1


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

