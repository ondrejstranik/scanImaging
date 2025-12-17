# simple_sine_wave_signals.py --- Generate continuous sine wave signals on two analog output channels with configurable parameters.
# This is a simple test for the core functionality of NI DAQ analog output using the nidaqmx package.
# Signals can be tested with an oscilloscope or by connecting to other DAQ input channels.
# Adjust frequency, amplitude, offset, and phase difference as needed.
# Can be connected to galvo scanners, but make sure to set parameters within safe limits for your hardware.

import time
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType

# Device and channel names
device = "Dev1"          # Change to your device name in NI MAX
channel0 = f"{device}/ao0"
channel1 = f"{device}/ao1"

# Waveform parameters (customize)
# Warning: fequency should be set according to the capabilities of your output device
freq = 10.0              # Hz
# Warning: max voltage amplitude + offset must be within the limits of output device
amplitude0 = 5.0         # Volts peak for channel 0
amplitude1 = 5.0         # Volts peak for channel 1
offset0 = 0.0            # DC offset (Volts) for channel 0
offset1 = 0.0            # DC offset (Volts) for channel 1
phase_diff_deg = 90.0    # phase difference between ch1 and ch0 in degrees

rate = 100000            # Samples per second
duration = 1.0           # seconds per buffer (samps_per_chan)
samples = int(rate * duration)

# Build time vector and waveforms
t = np.linspace(0, duration, samples, endpoint=False)
phase_diff_rad = np.deg2rad(phase_diff_deg)

waveform0 = amplitude0 * np.sin(2.0 * np.pi * freq * t) + offset0
waveform1 = amplitude1 * np.sin(2.0 * np.pi * freq * t + phase_diff_rad) + offset1

# For multi-channel writes NI expects an array of shape (samples, channels)
data = np.vstack((waveform0, waveform1)).T  # shape -> (samples, 2)

# Create AO task and write the buffer
with nidaqmx.Task() as task:
    # add both analog output channels
    task.ao_channels.add_ao_voltage_chan(channel0, min_val=-10.0, max_val=10.0)
    task.ao_channels.add_ao_voltage_chan(channel1, min_val=-10.0, max_val=10.0)

    # Configure continuous generation
    task.timing.cfg_samp_clk_timing(rate,
                                    sample_mode=AcquisitionType.CONTINUOUS,
                                    samps_per_chan=samples)

    # Write buffer (do not auto-start so we can verify)
    task.write(data, auto_start=False)

    # Start generation
    task.start()
    print(f"Generating {freq} Hz sine waves on {channel0} and {channel1}...")
    print(f"Channel0: amp={amplitude0} V, offset={offset0} V")
    print(f"Channel1: amp={amplitude1} V, offset={offset1} V, phase diff={phase_diff_deg} deg")

    try:
        # keep running until Ctrl+C; sleep to avoid busy loop
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping generation...")
        # optionally write zeros to outputs to leave them quiet
        zeros = np.zeros((samples, 2))
        try:
            task.stop()
            task.write(zeros, auto_start=False)
        except Exception:
            # if writing fails after stop, ignore and continue cleanup
            pass
        finally:
            # ensure task is stopped (context manager will close it)
            try:
                task.stop()
            except Exception:
                pass

    print("Stopped.")
