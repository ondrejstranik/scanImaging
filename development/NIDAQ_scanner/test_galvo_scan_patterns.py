#!/usr/bin/env python3
"""
Galvo/AO pattern streamer.

Usage:
 - Run the script and follow prompts.
 - Press keys while streaming:
     's' = square (dwell corners) 
     'c' = circle
     'x' = cross
     'r' = raster (sharp)
     'l' = raster (smooth)
     'q' = quit (stop and zero outputs)

Notes:
 - Configure `V_RANGE`, `RATE`, `POINTS`, `AO_CHANNELS` at top to match your DAQ.
 - Requires `nidaqmx` and `keyboard` packages.
"""
from __future__ import annotations
import time
import numpy as np
from safe_scan_pattern import safe_scan_pattern, stop_outputs, periodic_gradient, periodic_second_gradient, check_if_exceeds_limits, print_pattern_metrics,set_voltage



# User parameters (tweak to your hardware / needs)
V_RANGE = 5.0                  # ± voltage range visible in UI (not enforced by device)
RATE = 10000                    # samples per second for AO streaming
#RATE = 100000  # Sample rate in Hz (100 kHz typical for galvo microscopy)
POINTS = 500                   # default points per pattern (per loop)
AO_CHANNELS = "Dev1/ao0:1"     # channel string (two channels expected)
SLEEP_POLL = 0.05              # poll interval while waiting for key press
# add hard coded Galvo limits to prevent overdriving
GALVO_V_MIN = -10.0
GALVO_V_MAX = 10.0
# galvo limits: specs for Thorlabs GVS002 galvo scanner sine wave with 250 Hz frequency at full (+/-10 V) amplitude
# vmax=2 pi f A = 2 * pi * 250 Hz * 10 V = 15.707 V/ms
# amax=(2 pi f)^2 A = (2 * pi * 250 Hz)^2 * 10 V = 24,674 V/ms²
GALVO_MAX_DERIVATIVE_V_PER_S = 15707  # max allowed slew rate (V/s)
GALVO_MAX_SECOND_DERIVATIVE_V_PER_S_SQUARED = 2.4674e7  # max allowed acceleration (V/s²) (10 V/ms²)
GALVO_MAX_FREQUENCY_HZ = 250.0         # max allowed frequency (Hz)
# GVS002 RECOMMENDED LIMITS (adjust if needed)
#SAFE_MAX_SCAN_RATE = 150        # Hz at full amplitude
#SAFE_MAX_DVDT = 5_000           # V/ms = 5 V/ms
#V_PER_DEG = 1.0                 # typical GVS002 value (~0.8–1.0 V/degree)
#SAFE_MAX_ANGULAR_VEL = 5.0      # degrees/ms

def generate_safe_raster_pattern(
    rate: int,
    fov_voltage: float,
    pixels_x: int,
    pixels_y: int,
    line_rate: float,
    flyback_frac: float,
    flyback_frame_frac: float,
    bidirectional: bool = True,
) -> np.ndarray:    
    t,x,y, pixel_gate, line_trig, frame_trig = safe_scan_pattern(
        fs=int(rate),
        ny=pixels_y,
        line_rate=line_rate,
        x_amp=fov_voltage / 2,
        y_amp=fov_voltage / 2,
        x_flyback_frac=flyback_frac,
        y_flyback_frac=flyback_frame_frac,
        bidirectional=bidirectional,
    )
    return np.column_stack((x, y))



# Local imports defer to runtime so file can be inspected without nidaq or GUI
def make_circle(radius: float = V_RANGE / 2, points: int = POINTS) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, points, endpoint=False)
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))

# make simple sine wave in x or y or diagonal one or diagonal 2 for testing
def make_sine_wave(size: float = V_RANGE, points: int = POINTS, direction: str = 'x') -> np.ndarray:
    t = np.linspace(0, 1, points, endpoint=False)
    if direction == 'x':
        x = size * np.sin(2 * np.pi * t)
        y = np.zeros_like(x)
    elif direction == 'y':
        y = size * np.sin(2 * np.pi * t)
        x = np.zeros_like(y)
    elif direction == 'diag1':
        x = size * np.sin(2 * np.pi * t)
        y = size * np.sin(2 * np.pi * t)
    elif direction == 'diag2':
        x = size * np.sin(2 * np.pi * t)
        y = -size * np.sin(2 * np.pi * t)
    else:
        raise ValueError("Invalid direction; choose from 'x', 'y', 'diag1', 'diag2'")
    return np.column_stack((x, y))


def make_dwell_square(size: float = V_RANGE, edge_points: int = POINTS, dwell_points: int | None = None) -> np.ndarray:
    if dwell_points is None:
        dwell_points = max(1, edge_points // 10)
    half = size / 2.0
    corners = [(+half, +half), (-half, +half), (-half, -half), (+half, -half)]

    x = []
    y = []
    for i in range(4):
        cx, cy = corners[i]
        x.extend([cx] * dwell_points)
        y.extend([cy] * dwell_points)

        nx, ny = corners[(i + 1) % 4]
        x_edge = np.linspace(cx, nx, edge_points, endpoint=False)
        y_edge = np.linspace(cy, ny, edge_points, endpoint=False)
        x.extend(x_edge)
        y.extend(y_edge)

    total = len(x)
    t = np.linspace(0, 1, total)
    t_uniform = np.linspace(0, 1, edge_points)
    x_u = np.interp(t_uniform, t, x)
    y_u = np.interp(t_uniform, t, y)
    return np.column_stack((x_u, y_u))


def make_smooth_cross(size: float = V_RANGE, points: int = POINTS) -> np.ndarray:
    half = size / 2.0
    t = np.linspace(0, 2 * np.pi, points, endpoint=False)
    x = half * np.sin(2 * t)   # horizontal strokes
    y = half * np.sin(t)       # vertical stroke
    return np.column_stack((x, y))


def make_raster(samples_per_line: int = 512, lines_per_frame: int = 512, v_range: float = V_RANGE) -> np.ndarray:
    x_line = np.linspace(-v_range / 2, v_range / 2, samples_per_line)
    x = np.tile(x_line, lines_per_frame)
    y_line = np.linspace(-v_range / 2, v_range / 2, lines_per_frame)
    y = np.repeat(y_line, samples_per_line)
    return np.column_stack((x, y))


def make_raster_sine(samples_per_line: int = 512, lines_per_frame: int = 128,
                     v_range: float = V_RANGE, turnaround_points: int = 100) -> np.ndarray:
    half = v_range / 2.0
    x = []
    y = []
    y_vals = np.linspace(-half, half, lines_per_frame)
    base_line = np.linspace(-half, half, samples_per_line)

    for yi in y_vals:
        x.extend(base_line)
        y.extend([yi] * samples_per_line)
        t = np.linspace(0, 1, turnaround_points)
        # smooth turnaround from last to first (sinusoidal)
        x_back = base_line[-1] + (base_line[0] - base_line[-1]) * 0.5 * (1 - np.cos(np.pi * t))
        x.extend(x_back)
        y.extend([yi] * turnaround_points)

    return np.column_stack((np.array(x), np.array(y)))





def run_pattern(pattern: np.ndarray, rate: int = RATE, device_channels: str = AO_CHANNELS) -> None:
    """
    Stream one pattern continuously to the two AO channels until this function returns.
    The function blocks while the pattern is active; caller should arrange re-calling
    to switch patterns.
    """
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, RegenerationMode

    if pattern.ndim != 2 or pattern.shape[1] != 2:
        raise ValueError("pattern must be shape (N, 2)")

    samples = int(len(pattern))
    # Ensure contiguous (samples, channels) float array
    data = np.asarray(pattern, dtype=float).reshape(samples, 2)

    with nidaqmx.Task() as ao_task:
        ao_task.ao_channels.add_ao_voltage_chan(device_channels, min_val=-10.0, max_val=10.0)
        ao_task.timing.cfg_samp_clk_timing(rate, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=samples)
        # allow regeneration for continuous replay; write expects shape (samples, channels)
        try:
            ao_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
        except Exception:
            # some backends may not expose regen_mode; ignore if not supported
            pass

        # write and start streaming
        # ensure a float64 contiguous array
        data = np.ascontiguousarray(data, dtype=float)

        # prefer list-of-channels layout (channels, samples)
        try:
            channels_data = data.T.tolist()   # -> [[ch0_samples...], [ch1_samples...]]
            ao_task.write(channels_data, auto_start=True)
        except RuntimeError:
            # fallback: try writing numpy array transposed (some versions accept array)
            try:
                ao_task.write(data.T, auto_start=True)
            except Exception as exc:
                # last-resort: try the original layout to surface the real error
                print("DAQ write failed; debug info:",
                      "data.shape=", data.shape, "dtype=", data.dtype, "contiguous=", data.flags['C_CONTIGUOUS'])
                raise

        print(f"Streaming pattern (samples={samples}, rate={rate} Hz). Press keys to change pattern or 'q' to quit.")

        # polling loop: the caller will break the outer loop and re-call run_pattern with a new pattern
        try:
            # we simply stay alive until the caller decides to stop streaming by returning control
            # in this script we will break by keyboard events detected in main loop
            while True:
                time.sleep(SLEEP_POLL)
                # keep the loop alive; actual pattern switching is handled by terminating this task
                # The outer logic will break by returning from run_pattern (we don't return here)
                # to allow key-driven switching we rely on the caller to interrupt and re-call
        except KeyboardInterrupt:
            # user interrupt; clean up and return
            print("Keyboard interrupt received inside streaming; stopping pattern.")
            ao_task.stop()


def plot_pattern(x, y, interval_ms=10):
    """Show a simple animated preview of the (x,y) pattern using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except Exception as exc:
        print("Matplotlib not available:", exc)
        return

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size == 0 or y.size == 0:
        print("Empty pattern, nothing to display.")
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x, y, color="lightgray", lw=1, label="Path")
    dot, = ax.plot([], [], "ro", markersize=6)
    ax.set_xlim(-V_RANGE, V_RANGE)
    ax.set_ylim(-V_RANGE, V_RANGE)
    ax.set_aspect("equal")
    ax.set_title("Galvo Scan Simulation")
    ax.grid(True)

    def update(frame):
        xf = float(x[frame % x.size])
        yf = float(y[frame % y.size])
        dot.set_data([xf], [yf])
        return (dot,)

    ani = FuncAnimation(fig, update, frames=range(len(x)), interval=interval_ms, blit=True)
    plt.legend()
    plt.show()

def plot_pattern_xy(pattern: np.ndarray,rate: float):
    """Plot the given pattern using matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("Matplotlib not available:", exc)
        return
        # plot x and y channels separately
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 1, figsize=(8, 6))
    axs[0].plot(pattern[:, 0], label="X channel")
    axs[0].set_title("X Channel Waveform")
    axs[0].set_ylabel("Voltage (V)")
    axs[0].grid(True)
    axs[0].legend()
    axs[1].plot(pattern[:, 1], label="Y channel", color="orange")
    axs[1].set_title("Y Channel Waveform")
    axs[1].set_ylabel("Voltage (V)")
    axs[1].set_xlabel("Sample Index")
    axs[1].grid(True)
    axs[1].legend()
    # add a plot with the x and y derivatives
    axs[2].plot(periodic_gradient(pattern[:, 0], 1/rate), label="X derivative", color="blue")
    axs[2].plot(periodic_gradient(pattern[:, 1], 1/rate), label="Y derivative", color="green")
    axs[2].set_title("Derivatives (X and Y)")
    axs[2].set_ylabel("Voltage derivatives (V/s)")
    axs[2].set_xlabel("Sample Index")
    axs[2].grid(True)
    axs[2].legend()
    # add also a plot with the acceleration
    axs[3].plot(periodic_second_gradient(pattern[:, 0], 1/rate), label="X acceleration", color="cyan")
    axs[3].plot(periodic_second_gradient(pattern[:, 1], 1/rate), label="Y acceleration", color="magenta")
    axs[3].set_title("Accelerations (X and Y)")
    axs[3].set_ylabel("Voltage acceleration (V/s²)")
    axs[3].set_xlabel("Sample Index")
    axs[3].grid(True)
    axs[3].legend()
    plt.tight_layout()
    plt.show()



def main():
    # check dependencies but don't hard-fail: plotting should work without DAQ
    try:
        import nidaqmx  # quick check for dependency
        has_nidaqmx = True
    except Exception:
        print("nidaqmx not available: streaming disabled (plot/display still works).")
        has_nidaqmx = False

    try:
        import keyboard
    except Exception:
        print("keyboard module not available; interactive switching disabled.")
        keyboard = None
    # input actual voltage range and sample rate
    vrange=V_RANGE
    rate=RATE
    try:
        v_input = input(f"Enter voltage range for display (default ±{V_RANGE} V): ").strip()
        if v_input:
            tmp = float(v_input)
            if tmp > 0 and tmp <= 5.0:
                vrange = tmp
        r_input = input(f"Enter sample rate in Hz (default {RATE}): ").strip()
        if r_input:
            tmp = int(r_input)
            if tmp > 0 and tmp <= 200000:
                rate = tmp
    except Exception as exc:
        print("Invalid input, using defaults:", exc)   
    
    # Precompute patterns (same as before)
    pattern_map = {
        'square': make_dwell_square(size=vrange, edge_points=POINTS),
        'circle': make_circle(radius=vrange / 2, points=POINTS),
        'cross': make_smooth_cross(size=vrange, points=POINTS),
        'sineX': make_sine_wave(size=vrange, points=POINTS, direction='x'),
        'sineY': make_sine_wave(size=vrange, points=POINTS, direction='y'),
        'sineDiag1': make_sine_wave(size=vrange, points=POINTS, direction='diag1'),
        'sineDiag2': make_sine_wave(size=vrange, points=POINTS, direction='diag2'),
        'rasterU': generate_safe_raster_pattern(
    rate=rate,
    fov_voltage=vrange,
    pixels_x=512,
    pixels_y=512,
    line_rate=250,  # 100 lines per second
    flyback_frac=0.7,
    flyback_frame_frac=1.0/512,
    bidirectional=False
    ),
    'rasterB': generate_safe_raster_pattern(
    rate=rate,
    fov_voltage=vrange,
    pixels_x=512,
    pixels_y=512,# 512
    line_rate=300,#250
    flyback_frac=0.1,
    flyback_frame_frac=20/512, #1.5/512
    bidirectional=True
    )
    }
# make_raster(samples_per_line=100, lines_per_frame=100, v_range=V_RANGE),
    print("Parameters:")
    print(f"  AO channels: {AO_CHANNELS}")
    print(f"  Voltage range visual: ±{V_RANGE} V, pattern range {vrange} V")
    print(f"  Sample rate: {RATE} sps")
    print(f"  Points per pattern (default): {POINTS}")
    print()
    print("Controls while streaming:")
    print("  s = square, c = circle, x = cross, u = unidirectional raster, b = bidirectional raster, l = smooth raster, q = quit")
    # startup prompt: allow display mode without nidaqmx
    choice = input("Press Enter to start streaming (if available), 'q' to quit, or 'd' to display patterns: ").strip().lower()
    if choice.startswith('q'):
        print("Exiting without generating any waveform.")
        return
    # choose initial pattern
    current = 'circle'
    while choice.startswith('d'):
        # interactive display selection (works even without nidaqmx)
        print("Select pattern to display: 's'=square, 'c'=circle, 'x'=cross, 'u'=raster unidirectional, 'b'=raster bidirectional, 'WX'=sine wave in X, 'WY'=sine wave in Y, 'WD1'=sine wave diagonal 1, 'WD2'=sine wave diagonal 2")
        sel = input("Choice: ").strip().lower()
        if sel == 's':
            patt = pattern_map['square']
        elif sel == 'c':
            patt = pattern_map['circle']
        elif sel == 'x':
            patt = pattern_map['cross']
        elif sel == 'u':
            patt = pattern_map['rasterU']
        elif sel == 'b':
            patt = pattern_map['rasterB']
        elif sel == 'wx':
            patt = pattern_map['sineX']
        elif sel == 'wy':
            patt = pattern_map['sineY']
        elif sel == 'wd1':
            patt = pattern_map['sineDiag1']
        elif sel == 'wd2':
            patt = pattern_map['sineDiag2']
        else:
            print("Unknown selection, showing circle.")
            patt = pattern_map['circle']

        x, y = patt.T
        print_pattern_metrics(patt, rate)
        plot_pattern_xy(patt,rate)
        plot_pattern(x, y, interval_ms=10*1000.0 / rate)
        if check_if_exceeds_limits(patt, rate):
            print("Pattern exceeds galvo limits. Do not stream this pattern to hardware.")
        # after display, ask whether to continue to streaming
        current= sel if sel in pattern_map else 'circle'
        choice = input("Display finished. Press Enter to continue to streaming (if available), 'd' to display another waveform, or 'q' to quit: ").strip().lower()
        if choice.startswith('q'):
            print("Exiting without streaming.")
            return

    # if user continues to streaming but nidaqmx missing -> inform and exit
    if not has_nidaqmx:
        print("nidaqmx is not installed or device not available; streaming disabled.")
        print("You can use the 'd' display option on a machine with matplotlib installed.")
        return

    choice = input("Do you want to set a specific voltage value (y/N): ").strip().lower()
    while choice.startswith('y'):
        xvolt = float(input("input x voltage: ").strip())
        yvolt = float(input("input y voltage: ").strip())
        print("Setting voltage value")
        set_voltage(AO_CHANNELS,xvolt,yvolt)
        choice = input("Do you want to set a specific voltage value (y/N): ").strip().lower()

    input("Press Enter to start streaming (or Ctrl-C to cancel)...")


    try:
        while current is not None:
            pattern = pattern_map.get(current)
            if pattern is None:
                print(f"Unknown pattern key: {current}; defaulting to circle.")
                pattern = pattern_map['circle']

            # start a streaming task in a subprocess-like fashion by running it and watching keys
            # We'll create a Task here and watch for keyboard input to decide what to do next.
            import nidaqmx
            from nidaqmx.constants import AcquisitionType, RegenerationMode

            samples = len(pattern)
            data = np.asarray(pattern, dtype=float).reshape(samples, 2).T

            # show metrics for the pattern about to be streamed
            print_pattern_metrics(pattern, rate)
            if check_if_exceeds_limits(pattern, rate):
                print("Pattern exceeds galvo limits; not streaming. Choose a different pattern.")
                current = None
                continue
            with nidaqmx.Task() as ao_task:
                ao_task.ao_channels.add_ao_voltage_chan(AO_CHANNELS, min_val=-10.0, max_val=10.0)
                ao_task.timing.cfg_samp_clk_timing(rate, sample_mode=AcquisitionType.CONTINUOUS, samps_per_chan=samples)
                try:
                    ao_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
                except Exception:
                    pass
                ao_task.write(np.ascontiguousarray(data), auto_start=True)


                # inner loop polls keyboard to decide whether to switch pattern
                next_pattern = None
                try:
                    while True:
                        time.sleep(SLEEP_POLL)
                        if keyboard:
                            if keyboard.is_pressed('s'):
                                next_pattern = 'square'
                                break
                            if keyboard.is_pressed('c'):
                                next_pattern = 'circle'
                                break
                            if keyboard.is_pressed('x'):
                                next_pattern = 'cross'
                                break
                            if keyboard.is_pressed('u'):
                                next_pattern = 'rasterU'
                                break
                            if keyboard.is_pressed('b'):
                                next_pattern = 'rasterB'
                                break
                            if keyboard.is_pressed('q'):
                                next_pattern = None
                                break
                        # If no keyboard module: fallback to manual prompt
                except KeyboardInterrupt:
                    # allow Ctrl-C to stop streaming and exit
                    print("Interrupted by user.")
                    next_pattern = None

                # stop and close task, then update current for next iteration
                try:
                    ao_task.stop()
                except Exception:
                    pass

                current = next_pattern
    finally:
        # Ensure outputs are set to zero (best-effort) on exit
        stop_outputs(AO_CHANNELS)
        print("Exited cleanly.")


if __name__ == "__main__":
    main()


