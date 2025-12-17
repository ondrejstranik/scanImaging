# Simple script to read voltage from a DET36A/M photodiode using a NI DAQ device.
# Make sure you have the nidaqmx package installed and a compatible NI DAQ device connected.
# Adjust DEVICE and CHANNEL variables as needed for your setup.
# Can be used to monitor any other voltage signal within the DAQ's input range.

import nidaqmx
from nidaqmx.constants import TerminalConfiguration
import time

DEVICE = "Dev1"
CHANNEL = "ai0"    # change if needed (e.g. ai1)
SAMPLE_INTERVAL = 0.02  # seconds between prints (50 Hz)

# Create and configure the AI task
task = nidaqmx.Task()
task.ai_channels.add_ai_voltage_chan(
    f"{DEVICE}/{CHANNEL}",
    terminal_config=TerminalConfiguration.RSE,  # single-ended referenced to DAQ ground
    min_val=0.0,   # DET36A/M outputs 0..10 V
    max_val=10.0
)

print("Reading photodiode voltage (DET36A/M). Press Ctrl+C to stop.")
try:
    while True:
        # read single sample
        value = task.read()   # returns a float
        # print in-place on one terminal line
        print(f"\rVoltage: {value:6.3f} V", end="", flush=True)
        time.sleep(SAMPLE_INTERVAL)
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    task.close()