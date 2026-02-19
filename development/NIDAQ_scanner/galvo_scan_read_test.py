# **NI USB‑6421**: only **two AO channels** → we’ll drive the GVS002 galvos in **single‑ended mode** (AO0 = X+, AO1 = Y+), with X− and Y− tied to the galvo controller’s signal ground.
# **Thorlabs GVS002**: ±10 V input for full ±12.5° mechanical deflection, ~0.8 V/° scaling. We’ll pick safe amplitudes within its bandwidth.
# **Thorlabs DET36A/M**: biased Si photodiode, BNC output, DC‑coupled, 0–10 V into high‑impedance input. We’ll read it on AI0.

import time
import threading
import queue
import numpy as np
from collections import deque
from safe_scan_pattern import print_pattern_metrics, safe_scan_pattern, stop_outputs,check_if_exceeds_limits

# ----------------------------
# Device and scan parameters
# ----------------------------
#DEV = "Dev1"

# Detector input (Thorlabs DET36A/M)
#ai_channel = f"{DEV}/ai0"
#ai_vmin, ai_vmax = 0.0, 13.0  # DET36A/M output range into high-Z

# Threading



#V_RANGE = 5.0                  # ± voltage range visible in UI (not enforced by device)
# rate limit of DAQ AI/AO: 250 kS/s.
#RATE = 200000                    # samples per second for AO streaming
#RATE = 100000  # Sample rate in Hz (100 kHz typical for galvo microscopy)
#POINTS = 500                   # default points per pattern (per loop)
#AO_CHANNELS = f"{DEV}/ao0:1"     # channel string (two channels expected)
#SAMPLE_CLOCK=f"{DEV}/ao/SampleClock"
#TRIGGER_START=f"/{DEV}/ao/StartTrigger"
#TRIGGER_LINES=f"{DEV}/port0/line0:2"  # DO lines for line/frame/dwell triggers
#TRIGGER_PORT=f"{DEV}/port0"              # port for DO triggers
#SLEEP_POLL = 0.05              # poll interval while waiting for key press


class GalvoAoWrapper:
    def __init__(self, task,channel):
        self.task = task
        self.data=None
        self.channel=channel

    def write(self, ao_data):
        tmp = np.asarray(ao_data, dtype=np.float64,order='C')
        self.data = np.ascontiguousarray(tmp.T.copy())
        self.task.write(self.data, auto_start=False)

    def start(self):
        self.task.start()

    def stop(self):
        self.task.stop()
        self.task.close()
        if self.channel is not None:
            stop_outputs(self.channel)

class AiWrapper:
    def __init__(self, task):
        self.task = task
    # needed to mock scan
    def initialize(self,scan_layout):
        pass

    def start(self):
        self.task.start()

    def read(self, n,timeout):
        return self.task.read(number_of_samples_per_channel=n, timeout=timeout)

    def stop(self):
        self.task.stop()
        self.task.close()

class DoWrapper:
    def __init__(self, task):
        self.task = task
        self.data=None

    def write_from_gates(self,pixel_gate,line_trig,frame_trig):
        do_data = (
        (pixel_gate << 0) |
        (line_trig   << 1) |
        (frame_trig  << 2)
        ).astype(np.uint8)
        self.write(do_data)

    def write(self, do_data):
        self.data = do_data.copy()
        self.task.write(self.data, auto_start=False)

    def start(self):
        self.task.start()

    def stop(self):
        self.task.stop()
        self.task.close()



# Important Note: the ai and do tasks need to be armed (started) before the ao task. Otherwise the tasks get out of sync.
# The AO task defines the sample clock and therefore it has to be generated first, but started last.
# Only if the AO task start the sample clock, the other tasks will start.
class NIDaqTaskFactory:
    def __init__(self, device="Dev1"):
        import nidaqmx
        self.nidaqmx = nidaqmx
        self.device = device
        self._sample_clock = None
        self._ao_task = None
        self.device_voltage_range=10.0 # ± voltage range visible in UI (not enforced by device)
        self.device_max_sample_rate=250000 
        self.device_sample_rate=200000 
        self.ao_channels="ao0:1"
        self.do_port="port0/line0:2"
        #        DET36A/M BNC center  → Pin 19 (AI 0) Label 0
        #        DET36A/M BNC shield  → Pin 90  (AI GND) Label triangle with tick
        self.ai_channel="ai0"

    def set_rate(self,rate):
        if rate>self.device_max_sample_rate:
            raise RuntimeError("To large input for smaple rate on this device!")
        self.device_sample_rate=rate

    def create_ao_wrapper(self, samps_per_chan):
        # samps_per_chan=total_samples=ao_data.shape[0]
        from nidaqmx.constants import AcquisitionType,RegenerationMode
        ao_channels=f"{self.device}/{self.ao_channels}"
        ao_task = self.nidaqmx.Task()
        ao_task.ao_channels.add_ao_voltage_chan(ao_channels, min_val=-self.device_voltage_range, max_val=self.device_voltage_range)
        ao_task.timing.cfg_samp_clk_timing(self.device_sample_rate, 
                sample_mode=AcquisitionType.CONTINUOUS, 
                samps_per_chan=samps_per_chan)
        try:
            ao_task.out_stream.regen_mode = RegenerationMode.ALLOW_REGENERATION
        except Exception:
            pass

        # sample clock is needed for later reference.
        self._sample_clock = ao_task.timing.samp_clk_term
        self._ao_task = ao_task

        return GalvoAoWrapper(ao_task,ao_channels)

    def create_do_wrapper(self, samps_per_chan):
        if self._sample_clock is None:
            raise RuntimeError("AO must be created first")
        from nidaqmx.constants import AcquisitionType,Edge,LineGrouping
        do_task = self.nidaqmx.Task()
        do_task.do_channels.add_do_chan(
            f"{self.device}/{self.do_port}",
            line_grouping=LineGrouping.CHAN_FOR_ALL_LINES
            )
        do_task.timing.cfg_samp_clk_timing(
                rate=self.device_sample_rate,
                source=self._sample_clock,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=samps_per_chan
            )
        return DoWrapper(do_task)

    def create_ai_wrapper(self, ai_vmin,ai_vmax,samps_per_chan):
        # samps_per_chan=2*scan_layout["max_line_read_samples"]
        if self._sample_clock is None:
            raise RuntimeError("AO must be created first")
        from nidaqmx.constants import AcquisitionType,Edge
        ai_task = self.nidaqmx.Task()
        ai_task.ai_channels.add_ai_voltage_chan(f"{self.device}/{self.ai_channel}",
                                                 min_val=ai_vmin, max_val=ai_vmax,
                                                 terminal_config=self.nidaqmx.constants.TerminalConfiguration.RSE
                                                 )
        ai_task.timing.cfg_samp_clk_timing(
            rate=self.device_sample_rate,
            source=self._sample_clock,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=samps_per_chan
        )
        
        return AiWrapper(ai_task)


class FakeGalvoAoWrapper:
    def __init__(self):
        self.task = None
        self.data=None
        self.channel=None
        self.ao_data=None

    def write(self, ao_data):
        self.ao_data=ao_data

    def start(self):
        pass

    def stop(self):
        pass


class FakeAiWrapper:
    def __init__(self):
        self.line_length=10000
        self.bidirectional=True
        self.line_flyback=10
        self.frame_flyback=500
        self.nLines=512

    def initialize(self,scan_layout):
        self.line_length=scan_layout["line_samples"][0]
        self.bidirectional=True
        self.line_flyback=scan_layout["complete_line_samples"][0]-scan_layout["line_samples"][0]
        self.complete_line_size=scan_layout["complete_line_samples"][0]
        if self.line_flyback<0:
            self.line_flyback=0
        self.frame_flyback=scan_layout["complete_line_samples"][-1]-scan_layout["line_samples"][-1]-self.line_flyback
        if self.frame_flyback<0:
            self.frame_flyback=0
        self.nLines=len(scan_layout["line_samples"])
        self.sample_size=scan_layout["pattern_size"]
        self.rate=scan_layout["rate"]
        self._set_image()

    def _set_image(self):
        from PIL import Image
        from pathlib import Path
        base_path = Path(__file__).resolve().parent
        p = rf"{base_path}/../../scanImaging/instrument/virtual/images/radial-sine-144.png"
        img = Image.open(p).convert("L")
        img = img.resize((int(self.line_length), int(self.nLines)), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        if arr.max() != 0:
            arr /= arr.max()
#        arr*=0.0
#        arr[:,len(arr)//2-10:len(arr)//2+10]=1.0
        self.image=arr
        if self.bidirectional:
            for n in range(arr.shape[0]):
                if n%2==1:
                    arr[n,:]=arr[n,::-1]
        arr=np.pad(arr,((0,0),(0,self.line_flyback)),mode='constant')
        print(arr.shape)
        if arr.shape[1]!=self.complete_line_size:
            raise RuntimeError(f"Wrong line size {arr.shape[1]}!={self.complete_line_size}")
        self.frame_image=arr
        self.framedata=np.concatenate((arr.flatten(),np.zeros(self.frame_flyback)))
        if len(self.framedata)!=self.sample_size:
            raise RuntimeError(f"Not matching fake sample size {len(self.framedata)}!={self.sample_size}")
        self.framedataindex=0



    def start(self):
        pass

    def read(self, n,timeout):
        nn=len(self.framedata)
        enddata=(self.framedataindex+n)
        if enddata>nn:
            enddata=enddata%nn
            outdata=np.concatenate((self.framedata[self.framedataindex:nn],self.framedata[0:enddata]))
        else:
            outdata=self.framedata[self.framedataindex:enddata]
        self.framedataindex=enddata
        time.sleep(n/self.rate)
        return outdata

    def stop(self):
        pass

class FakeDoWrapper:
    def __init__(self):
        self.task = None
        self.data=None

    def write_from_gates(self,pixel_gate,line_trig,frame_trig):
        pass

    def write(self, do_data):
        pass

    def start(self):
        pass

    def stop(self):
        pass



# Important Note: the ai and do tasks need to be armed (started) before the ao task. Otherwise the tasks get out of sync.
# The AO task defines the sample clock and therefore it has to be generated first, but started last.
# Only if the AO task start the sample clock, the other tasks will start.
class FakeNIDaqTaskFactory:
    def __init__(self, device="Dev1"):
        self.nidaqmx = None
        self.device = device
        self._sample_clock = None
        self._ao_task = None
        self.device_voltage_range=10.0
        self.device_max_sample_rate=250000 
        self.device_sample_rate=200000 
        self.ao_channels="ao0:1"
        self.do_port="port0/line0:2"
        self.ai_channel="ai0"

    def set_rate(self,rate):
        if rate>self.device_max_sample_rate:
            raise RuntimeError("To large input for smaple rate on this device!")
        self.device_sample_rate=rate

    def create_ao_wrapper(self, samps_per_chan):
        return FakeGalvoAoWrapper()

    def create_do_wrapper(self, samps_per_chan):
        return FakeDoWrapper()

    def create_ai_wrapper(self, ai_vmin,ai_vmax,samps_per_chan):
        return FakeAiWrapper()






specs_unidirectional={"rate":200000,
    "fov_voltage":1.5,
    "pixels_x":512,
    "pixels_y":512,
    "line_rate":250,  # 100 lines per second
    "flyback_frac":0.7,
    "flyback_frame_frac":1.0/512,
    "bidirectional":False}

specs_bidirectional={
    "rate":100000,
    "fov_voltage":1.5,
    "pixels_x":100,# we need at least ~8 samples per pixel. 
    "pixels_y":100,# 512
    "line_rate":300,#250
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
    ).astype(np.uint32)

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
    complete_line_samples=np.diff(line_starts,append=len(pixel_gate))
    # the maximum read 
    max_line_read_samples = np.max(complete_line_samples)
    max_line_samples = np.max(line_samples)
    


    scan_layout = {
    "line_start_indices": line_starts,
    "line_samples": line_samples,
    "line_end_indices": line_ends,
    "max_line_samples": max_line_samples,
    "complete_line_samples": complete_line_samples,
    "max_line_read_samples": max_line_read_samples,
    "pattern_size": len(pixel_gate),
    "pixel_gate": pixel_gate,
    "rate": rate
    }

    if np.sum(complete_line_samples)!=len(pixel_gate):
        raise RuntimeError("Wrong array size calculations")



    samples_per_pixel = max_line_samples / pattern_specs["pixels_x"]

    if check_if_exceeds_limits(np.column_stack((x, y)), rate):
        print_pattern_metrics(ao_data, rate)
        raise ValueError("Pattern exceeds device limits.")
    if samples_per_pixel < 3:
        raise ValueError(f"Pattern has only {samples_per_pixel} samples per pixel, which is less than the minimum of 3.")
    print_pattern_metrics(ao_data, rate)
    return ao_data, do_data, scan_layout



# ----------------------------
# Threads
# ----------------------------
def consume_from_fifo(fifo,n):
    out = [fifo.popleft() for _ in range(n)]
    return np.asarray(out, dtype=np.float32)

# Do not call ai_task.start() since thread handling is non-deterministic.
def detector_thread(ai_task,scan_layout,stop_event,image_queue, pattern_specs=specs_bidirectional):
    nx=pattern_specs["pixels_x"]
    ny=pattern_specs["pixels_y"]
    bidirectional=pattern_specs["bidirectional"]
    max_line_samples=scan_layout["max_line_samples"]
    line_idx = 0
    frame = np.zeros((ny, nx), dtype=np.float32)
    blocking=max_line_samples//nx #samples per pixel, rounded down. 
    if blocking <= 0:
        raise RuntimeError("Invalid blocking factor: increase sampling rate or reduce pixels_x.")
    if max_line_samples % nx != 0:
        print("Warning: pixel binning truncates samples")
    # add a software fifo
    fifo=deque()
    READ_CHUNK=4096
    MAX_FIFO=10*READ_CHUNK
    while not stop_event.is_set():
        samples_per_read = scan_layout["complete_line_samples"][line_idx]
        line_samples=scan_layout["line_samples"][line_idx]
        while len(fifo)<samples_per_read:
            fifo.extend(ai_task.read(READ_CHUNK, timeout=5.0))
        data = consume_from_fifo(fifo,samples_per_read)
        if samples_per_read<line_samples:
            raise RuntimeError(f"Read slice out of range: {samples_per_read} vs {line_samples}")
        if blocking * nx > line_samples:
            raise RuntimeError("Blocking exceeds available samples")
        data = np.asarray(data, dtype=np.float32)
        # 1) extract the relevant data.
        line=data[:line_samples]
        # This is irrelevant in most cases (just for safety)
        if len(line) < max_line_samples:
            padded = np.full(max_line_samples, np.nan, dtype=np.float32)
            padded[:len(line)] = line
            line = padded
        # 2) reordering in case of bidirectional data
        if bidirectional and (line_idx % 2 == 1):
            line = line[::-1]
        # 3) blocking, This needs to be the last step!
        line = np.nanmean(
            line[:nx * blocking].reshape(nx, blocking),
            axis=1
        )
        frame[line_idx,:] = line
        line_idx += 1
        if line_idx >= ny:
            try:
                if image_queue.full():
                    image_queue.get_nowait()
                image_queue.put_nowait(frame.copy())
            except queue.Full:
                print("AI thread: full image queue.")
                pass
            line_idx = 0
        if len(fifo)>MAX_FIFO:
             print("[WARN] FIFO overflow, dropping samples")
             while len(fifo) > MAX_FIFO // 2:
                fifo.popleft()
    print("[AI] Detector thread exiting")

class Display:
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt=plt
        
    def initialize_display(self,pattern_specs):
        self.nx=pattern_specs["pixels_x"]
        self.ny=pattern_specs["pixels_y"]
        self.plt.ion()
        self.fig, self.ax = self.plt.subplots()
        self.img = self.ax.imshow(
            np.zeros((self.ny, self.nx)),
            cmap="gray",
            origin="upper",
            aspect="auto"
        )
        self.plt.title("Live Confocal Image")
        self.plt.colorbar(self.img, ax=self.ax)



    def display_loop(self,stop_event,image_queue):
        frames = 0
        t0 = time.time()
        while not stop_event.is_set():
            try:
                frame = image_queue.get(timeout=0.5)
                self.img.set_data(frame)
                self.img.set_clim(frame.min(), frame.max())
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
                self.plt.pause(0.001)
                frames += 1
                if frames > 100000:
                    frames=0
                    t0=time.time()
                elapsed=time.time() - t0
                if int(elapsed)%5==4:
                    print(f"[Display] FPS ~ {float(frames)/elapsed}")
            except queue.Empty:
                self.plt.pause(0.001)

    def close(self):
        self.plt.ioff()
        self.plt.close(self.fig)

def show_image(img_data):
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plt.subplots()
    img = ax.imshow(
        img_data,
        cmap="gray",
        origin="upper",
        aspect="auto"
    )
    plt.title("Live Confocal Image")
    plt.colorbar(img, ax=ax)
    plt.pause(20)
    plt.ioff()
    plt.close(fig)



# ----------------------------
# Main
# ----------------------------
def run():
    ai_vmin, ai_vmax = 0.0, 10.0  # DET36A/M output range into high-Z
    ao_data, do_data, scan_layout = setup_and_check_pattern(specs_bidirectional)
    rate=int(specs_bidirectional["rate"])
    total_scan_samples = ao_data.shape[0]
    if total_scan_samples!= do_data.shape[0]:
        raise RuntimeError("No match of scan and trigger samples!")
    # make the reading channel larger than needed for safety
    samps_per_read_chan=10*4096   #2*scan_layout["max_line_read_samples"]
    stop_event = threading.Event()
    image_queue = queue.Queue(maxsize=3)
    display=Display()
    display.initialize_display(specs_bidirectional)
    tasksfactory=NIDaqTaskFactory()
    tasksfactory.set_rate(rate)
    ao_task=tasksfactory.create_ao_wrapper(total_scan_samples)
    do_task=tasksfactory.create_do_wrapper(total_scan_samples)
    ai_task=tasksfactory.create_ai_wrapper(ai_vmin,ai_vmax,samps_per_read_chan)
    ai_task.initialize(scan_layout)
    # auto_start=False by default
    ao_task.write(ao_data)
    do_task.write(do_data)
#    print(ai_task.read(2000,1))
#    print(ai_task.read(2000,1))
 #   show_image(ai_task.frame_image)
#    exit()
    t_ai = threading.Thread(
        target=detector_thread,
        args=(ai_task,scan_layout,stop_event,image_queue, specs_bidirectional),
        daemon=True
    )
    try:
        # arm the tasks:
        ai_task.start()
        do_task.start()
        # This has to be last since it starts the sample clock and hence also the other tasks.
        ao_task.start()
        time.sleep(0.05)
        t_ai.start()
        display.display_loop(stop_event,image_queue)
    finally:
        stop_event.set()
        t_ai.join()
        ai_task.stop()
        do_task.stop()
        ao_task.stop()
        display.close()
        print("All threads stopped.")

def test():
    ao_data, do_data, scan_layout = setup_and_check_pattern(specs_bidirectional)
    print("AO data shape:", ao_data.shape)
    print("DO data shape:", do_data.shape)
    print("Scan layout:", scan_layout["line_samples"], "samples per line")
    print("Scan layout:", scan_layout["max_line_samples"], "max samples per line")
    print("Scan layout:", scan_layout["complete_line_samples"], "complete line samples (including flyback)")
    print(len(scan_layout["complete_line_samples"]))
    print(len(scan_layout["line_samples"]))

if __name__ == "__main__":
    #test()
    run()

