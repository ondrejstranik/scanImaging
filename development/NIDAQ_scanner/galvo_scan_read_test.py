# **NI USB‑6421**: only **two AO channels** → we’ll drive the GVS002 galvos in **single‑ended mode** (AO0 = X+, AO1 = Y+), with X− and Y− tied to the galvo controller’s signal ground.
# **Thorlabs GVS002**: ±10 V input for full ±12.5° mechanical deflection, ~0.8 V/° scaling. We’ll pick safe amplitudes within its bandwidth.
# **Thorlabs DET36A/M**: biased Si photodiode, BNC output, DC‑coupled, 0–10 V into high‑impedance input. We’ll read it on AI0.

import time
import threading
import queue
import numpy as np

from safe_scan_pattern import print_pattern_metrics, safe_scan_pattern, stop_outputs,check_if_exceeds_limits

# ----------------------------
# Device and scan parameters
# ----------------------------
DEV = "Dev1"

# Detector input (Thorlabs DET36A/M)
ai_channel = f"{DEV}/ai0"
ai_vmin, ai_vmax = 0.0, 13.0  # DET36A/M output range into high-Z

# Threading



V_RANGE = 5.0                  # ± voltage range visible in UI (not enforced by device)
# rate limit of DAQ AI/AO: 250 kS/s.
RATE = 200000                    # samples per second for AO streaming
#RATE = 100000  # Sample rate in Hz (100 kHz typical for galvo microscopy)
POINTS = 500                   # default points per pattern (per loop)
AO_CHANNELS = f"{DEV}/ao0:1"     # channel string (two channels expected)
SAMPLE_CLOCK=f"{DEV}/ao/SampleClock"
TRIGGER_START=f"/{DEV}/ao/StartTrigger"
TRIGGER_LINES=f"{DEV}/port0/line0:2"  # DO lines for line/frame/dwell triggers
TRIGGER_PORT=f"{DEV}/port0"              # port for DO triggers
SLEEP_POLL = 0.05              # poll interval while waiting for key press


specs_unidirectional={"rate":RATE,
    "fov_voltage":4,
    "pixels_x":512,
    "pixels_y":512,
    "line_rate":250,  # 100 lines per second
    "flyback_frac":0.7,
    "flyback_frame_frac":1.0/512,
    "bidirectional":False}

specs_bidirectional={
    "rate":RATE,
    "fov_voltage":4,
    "pixels_x":512,# we need at least ~8 samples per pixel. 
    "pixels_y":512,# 512
    "line_rate":100,#250
    "flyback_frac":0.1,
    "flyback_frame_frac":20/512, #1.5/512
    "bidirectional":True}

def setup_and_check_pattern(pattern_specs=specs_bidirectional):
    rate=int(pattern_specs["rate"])
    samples_per_line = round(rate / pattern_specs["line_rate"])
    ny=pattern_specs["pixels_y"]
    # Note: pixel_gate is high during active pixel period, line_trig is high at start of each line, frame_trig is high at start of first line of each frame
    t,x,y, pixel_gate, line_trig, frame_trig = safe_scan_pattern(
        fs=rate,
        ny=pattern_specs["pixels_y"],
        line_rate=pattern_specs["line_rate"],
        x_amp=pattern_specs["fov_voltage"] / 2,
        y_amp=pattern_specs["fov_voltage"] / 2,
        x_flyback_frac=pattern_specs["flyback_frac"],
        y_flyback_frac=pattern_specs["flyback_frame_frac"],
        bidirectional=pattern_specs["bidirectional"],
    )
    ao_data=np.asarray(np.column_stack((x, y)), dtype=float).reshape(-1,2)
    # pack the three digital outputs into bits of a single byte: bit 0 = pixel_gate, bit 1 = line_trig, bit 2 = frame_trig
#    do_data=np.vstack((line_trig, frame_trig, pixel_gate)).T.astype(np.uint8)
    do_data = (
    (pixel_gate << 0) |
    (line_trig   << 1) |
    (frame_trig  << 2)
    ).astype(np.uint8)

    # Analyze the pixel_gate to find where the signal goes from low to high (start of active pixel period) and from high to low (end of active pixel period). This will help us determine the actual pixel indices and line start indices.
    # diff calculates the differences between consecutive elements. A transition from 0 to 1 will give +1, and a transition from 1 to 0 will give -1. We prepend a 0 to align the indices correctly.
    dg = np.diff(pixel_gate.astype(int), prepend=0)
    
    line_starts = np.where(dg == +1)[0]
    # shift indices by -1, but ommit all negative elements (first one).
    before_line_starts=line_starts - 1
    before_line_starts=before_line_starts[before_line_starts >= 0]
    line_ends   = np.where(dg == -1)[0]
    before_line_ends=line_ends-1
    before_line_ends=before_line_ends[before_line_ends >= 0]
    dg=np.diff(line_trig.astype(int), prepend=0)
    line_trigger_line_start=np.where(dg == +1)[0]
    # Sanity check: at line starts the pixel_gate should go from 0 to 1. This means before each line start index, the pixel_gate should be 0.
    if np.any(pixel_gate[line_starts]==0) or np.any(pixel_gate[before_line_starts]==1):
        print(pixel_gate[pixel_gate[line_starts]==0])
        print(pixel_gate[before_line_starts][pixel_gate[before_line_starts]==1])
        raise ValueError("Inconsistent pixel gate behavior detected line_starts.")
    if np.any(pixel_gate[line_ends]==1) or np.any(pixel_gate[before_line_ends]==0):
        raise ValueError("Inconsistent pixel gate behavior detected line_ends.")
    # all line trigger start signals should coincide with line starts
    if not np.array_equal(line_trigger_line_start, line_starts):
        print("Line trigger indices:", line_trig)
        raise ValueError("Line trigger does not coincide with line starts.")
    if len(line_starts) != ny:
        raise ValueError(f"Number of line starts ({len(line_starts)}) does not match expected number of lines ({ny}).")
    if len(line_ends) != ny:
        raise ValueError(f"Number of line ends ({len(line_ends)}) does not match expected number of lines ({ny}).")
    
    # each line starts at line_start[n] (including this point) and ends and line_end[n] (excluding this point).
    line_samples = line_ends - line_starts
    complete_line_samples=np.diff(line_starts, prepend=0,append=len(pixel_gate))
    # the maximum read 
    max_line_read_samples = np.max(complete_line_samples)
    max_line_samples = np.max(line_samples)
    


    scan_layout = {
    "line_start_indices": line_starts,
    "line_samples": line_samples,
    "line_end_indices": line_ends,
    "max_line_samples": max_line_samples,
    "complete_line_samples": complete_line_samples,
    "max_line_read_samples": max_line_read_samples
    }
    samples_per_pixel = max_line_samples / pattern_specs["pixels_x"]

    if check_if_exceeds_limits(np.column_stack((x, y)), rate):
        print_pattern_metrics(ao_data, rate)
        raise ValueError("Pattern exceeds device limits.")
    if samples_per_pixel < 3:
        raise ValueError(f"Pattern has only {samples_per_pixel} samples per pixel, which is less than the minimum of 3.")
    print_pattern_metrics(ao_data, rate)
    return ao_data, do_data, scan_layout


class NiDaqAOHandler:
    def __init__(self):
        self.aochannels=AO_CHANNELS
        self.aichannels=ai_channel
        self.min_voltage=-10.0
        self.max_voltage=10.0
        self.total_samples=0
        self.ao_task=None
        self.do_task=None
    
    def start(self,ao_data,do_data):
        pass

    def stop(self):
        pass


# ----------------------------
# Threads
# ----------------------------
def galvo_and_triggers_thread(ao_data,do_data,stop_event,pattern_specs=specs_bidirectional):
    rate=int(pattern_specs["rate"])
    total_samples = ao_data.shape[0]
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, LineGrouping, Edge, RegenerationMode
    try:
        with nidaqmx.Task() as ao_task, nidaqmx.Task() as do_task:
            # AO: two channels for X+ and Y+
            ao_task.ao_channels.add_ao_voltage_chan(AO_CHANNELS, min_val=-10.0, max_val=10.0)
            ao_task.timing.cfg_samp_clk_timing(rate, 
                sample_mode=AcquisitionType.CONTINUOUS, 
                samps_per_chan=total_samples)
            try:
                ao_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
            except Exception:
                pass

            # DO: three trigger lines, clocked by AO sample clock
            do_task.do_channels.add_do_chan(
            TRIGGER_PORT,
            line_grouping=LineGrouping.CHAN_FOR_ALL_LINES
            )

 #           do_task.do_channels.add_do_chan(TRIGGER_LINES, line_grouping=LineGrouping.CHAN_PER_LINE)
            do_task.timing.cfg_samp_clk_timing(
                rate=rate,
                source=SAMPLE_CLOCK,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=do_data.shape[0]
            )
            do_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                trigger_source=TRIGGER_START, trigger_edge=Edge.RISING
            )
            ao_task.write(np.ascontiguousarray(ao_data), auto_start=False)
            do_write = np.ascontiguousarray(do_data, dtype=np.uint8)
            do_task.write(do_write, auto_start=False)

            do_task.start()
            ao_task.start()

            print(f"[AO/DO] Running")
            try:
                while not stop_event.is_set():
                    time.sleep(0.1)
            finally:
                ao_task.stop()
                do_task.stop()
                print("[AO/DO] Stopped.")
    finally:
        stop_outputs(AO_CHANNELS)

def detector_thread(scan_layout,stop_event,image_queue, pattern_specs=specs_bidirectional):
    nx=pattern_specs["pixels_x"]
    ny=pattern_specs["pixels_y"]
    bidirectional=pattern_specs["bidirectional"]
    max_line_samples=scan_layout["max_line_samples"]
    rate=pattern_specs["rate"]    
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, Edge
    with nidaqmx.Task() as ai_task:
        ai_task.ai_channels.add_ai_voltage_chan(ai_channel, min_val=ai_vmin, max_val=ai_vmax)
        ai_task.timing.cfg_samp_clk_timing(
            rate=rate,
            source=SAMPLE_CLOCK,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=2*scan_layout["max_line_read_samples"]
        )
        ai_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            trigger_source=TRIGGER_START, trigger_edge=Edge.RISING
        )

        ai_task.start()
        print("[AI] Acquisition started.")
        line_idx = 0
        frame = np.zeros((ny, nx), dtype=np.float32)
        blocking=max_line_samples//nx #samples per pixel, rounded down. 
        if blocking <= 0:
            raise RuntimeError("Invalid blocking factor: increase sampling rate or reduce pixels_x.")
        if max_line_samples % nx != 0:
            print("Warning: pixel binning truncates samples")
        try:
            while not stop_event.is_set():
                samples_per_read = scan_layout["complete_line_samples"][line_idx]
                line_samples=scan_layout["line_samples"][line_idx]
                data = ai_task.read(number_of_samples_per_channel=samples_per_read, timeout=5.0)
                if samples_per_read<line_samples:
                    raise RuntimeError(f"Read slice out of range: {samples_per_read} vs {line_samples}")
                data = np.asarray(data, dtype=np.float32)
                line=data[:line_samples]
                if len(line) < max_line_samples:
                    padded = np.full(max_line_samples, 0, dtype=np.float32)
                    padded[:len(line)] = line
                    line = padded
                line = np.nanmean(
                    line[:nx * blocking].reshape(nx, blocking),
                    axis=1
                )
                if bidirectional and (line_idx % 2 == 1):
                    line = line[::-1]
                frame[line_idx,:] = line
                line_idx += 1
                if line_idx >= ny:
                    try:
                        if image_queue.full():
                            image_queue.get_nowait()
                        image_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass
                    line_idx = 0
        finally:
            ai_task.stop()
            print("[AI] Acquisition stopped.")

def display_thread(stop_event,image_queue,pattern_specs=specs_bidirectional):
    import matplotlib.pyplot as plt
    nx=pattern_specs["pixels_x"]
    ny=pattern_specs["pixels_y"]
    plt.ion()
    fig, ax = plt.subplots()
    img = ax.imshow(
        np.zeros((ny, nx)),
        cmap="gray",
        vmin=0,
        vmax=ai_vmax,
        origin="upper",
        aspect="auto"
    )
    plt.title("Live Confocal Image")
    plt.colorbar(img, ax=ax)

    frames = 0
    t0 = time.time()

    while not stop_event.is_set():
        try:
            frame = image_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        img.set_data(frame)
        img.set_clim(frame.min(), frame.max())
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        frames += 1
        if time.time() - t0 >= 1.0:
            print(f"[Display] FPS ~ {frames}")
            frames = 0
            t0 = time.time()

    plt.close(fig)

def display_loop(stop_event,image_queue,pattern_specs=specs_bidirectional):
    import matplotlib.pyplot as plt
    nx=pattern_specs["pixels_x"]
    ny=pattern_specs["pixels_y"]
    plt.ion()
    fig, ax = plt.subplots()
    img = ax.imshow(
        np.zeros((ny, nx)),
        cmap="gray",
        vmin=0,
        vmax=ai_vmax,
        origin="upper",
        aspect="auto"
    )
    plt.title("Live Confocal Image")
    plt.colorbar(img, ax=ax)

    frames = 0
    t0 = time.time()
    while not stop_event.is_set():
        try:
            frame = image_queue.get(timeout=0.1)
            img.set_data(frame)
            img.set_clim(frame.min(), frame.max())
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)
            frames += 1
            if time.time() - t0 >= 1.0:
                print(f"[Display] FPS ~ {frames}")
                frames = 0
                t0 = time.time()
        except queue.Empty:
            plt.pause(0.001)
    plt.ioff()
    plt.close(fig)



# ----------------------------
# Main
# ----------------------------
def run():
    ao_data, do_data, scan_layout = setup_and_check_pattern(specs_bidirectional)
    stop_event = threading.Event()
    image_queue = queue.Queue(maxsize=3)
    t_galvo = threading.Thread(
        target=galvo_and_triggers_thread,
        args=(ao_data, do_data,stop_event, specs_bidirectional),
        daemon=True
    )
    t_ai = threading.Thread(
        target=detector_thread,
        args=(scan_layout,stop_event,image_queue, specs_bidirectional),
        daemon=True
    )


    t_galvo.start()
    time.sleep(0.1)
    t_ai.start()

    try:
        display_loop(stop_event,image_queue)

    except KeyboardInterrupt:
        stop_event.set()
    
    t_galvo.join()
    t_ai.join()
    print("All threads stopped.")

def test():
    ao_data, do_data, pixel_gate, scan_layout = setup_and_check_pattern(specs_bidirectional)
    print("AO data shape:", ao_data.shape)
    print("DO data shape:", do_data.shape)
    print("Pixel gate shape:", pixel_gate.shape)
    print("Scan layout:", scan_layout["line_samples"], "samples per line")
    print("Scan layout:", scan_layout["max_line_samples"], "max samples per line")
    print("Scan layout:", scan_layout["complete_line_samples"], "complete line samples (including flyback)")

if __name__ == "__main__":
    test()
    #run()