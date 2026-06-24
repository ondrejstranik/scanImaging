"""Galvo scan tool using the PI Imaging SPAD23G (23-pixel array) as detector.

This is a SPAD-detector sibling of ``galvo_scan_read_test.py``. The key
architectural difference:

* ``galvo_scan_read_test.py`` reads a single analog photodiode through the
  NI-DAQ AI stream and re-bins that stream into pixels in software.
* This tool does NOT use the NI-DAQ AI at all. The NI-DAQ drives the galvos
  (AO) and provides the **dwell / line / frame clock triggers** (DO -> SPAD
  SMA inputs). We pull the per-channel data back from the running pSPAD
  application over its **TCP interface** (text port 9999 / binary port 9997 --
  there is no separate Python SDK; see spad23-integration-notes.md).

The SPAD23 supports two paradigms (see spad23-integration-notes.md §2):
  * scanning mode (``CS``): device bins photons into a (23, ny, nx) image with
    its internal dwell clock (only the frame clock can be external); intensity
    only. This maps directly onto ``get_frame()`` and is the first milestone.
  * stream mode (``SB``): per-photon events; the NI-DAQ dwell/line/frame edges
    come back as marker events and the laser clock gives the fine timestamp
    (microtime). This is the path for the eventual FLIM + ISM workflow;
    reconstruction is done in software from the markers (the exact marker
    numbering differs between the docs and the bundled examples -- see
    SPAD_STREAM_SCHEME).

Staged development (matches the agreed plan):
  (a) verify the trigger signals are correct        ->  --check-triggers
  (b) validate the workflow on synthetic SPAD data  ->  default (simulated)
  (c) carefully test the real detector w/o overexposure -> --real (lab machine)

Dev/lab split: this file runs fully simulated on a dev machine with no
hardware. The real SPAD path (``Spad23FrameSource``) is a TCP client with
``TODO(lab)`` seams to be completed/tested on the lab machine that has the DAQ
card + SPAD23 (both USB) and pSPAD running.

Safety (from development/SPAD23/for-code-dev-spad23-safety-guidelines.md):
VOP enabled in a try / disabled unconditionally in finally, per-frame
count-rate monitoring, pile-up detection, and masking of hot pixel index 7.

Crash recovery: if this program terminates with VOP active (SIGKILL, power
loss), the SPAD stays biased. Power-cycle the 5V adapter before restarting
pSPAD to reset the hardware to a known state. Code cannot handle this.
"""

import argparse
import datetime
import queue
import socket
import threading
import time

import numpy as np

# Reuse the proven pattern + AO/DO plumbing from the photodiode tool.
from galvo_scan_read_test import (
    setup_and_check_pattern,
    NIDaqTaskFactory,
    FakeGalvoAoWrapper,
    FakeDoWrapper,
    specs_bidirectional,
    specs_unidirectional,
)

# ----------------------------------------------------------------------------
# SPAD23G constants (see for-code-dev-spad23-safety-guidelines.md, sections 4-5)
# ----------------------------------------------------------------------------
SPAD23G_N_PIXELS = 23
SPAD23G_DEAD_TIME_NS = 50
SPAD23G_SAFE_COUNT_RATE_CPS = 2_000_000       # 1 / (10 * tau_d)
SPAD23G_SHUTDOWN_THRESHOLD_CPS = 6_700_000    # hardware auto-shutdown
SPAD23G_WARNING_FRACTION = 0.5                # warn at 50 % of safe limit
HOT_PIXELS = [7]                              # 0-based; pSPAD pixel 8
SPAD23G_DARK_CPS_HOT = 9_500                  # hot-pixel dark count rate

# SMA clock input limits (section 7)
SPAD23G_MIN_PULSE_NS = 12.0                   # min detectable pulse width
SPAD23G_MAX_SMA_VOLTAGE_HIGHZ = 3.3           # 5 V only if source is 50 ohm

# pSPAD TCP interface (see spad23-integration-notes.md). No separate Python SDK.
# Validated against piimaging.com/doc-pspad (v2.01) on 2026-06-22.
SPAD_HOST = "127.0.0.1"
SPAD_PORT = 9999            # single configurable port; text & binary share it
SPAD_TIMEOUT_S = 10.0

# Stream event codes per the OFFICIAL docs (piimaging.com/doc-pspad, v2.01),
# RELATIVE to the pixel count P (= 23 for 23G):
#   photons     : tag 0 .. P-1   (tag == channel index)
#   P + 0       : coarse-counter wraparound
#   P + 2..8    : dwell/line/frame markers and their combinations (see _decode_marker)
#   P + 11..13  : USB FIFO empty / almost-full / overflow
#
# !! CONFLICT: the bundled example scripts use a DIFFERENT numbering (photons
# 32..54, dwell=9 line=10 frame=12 overflow=17). The two vendor sources disagree.
# CONFIRM the scheme on the actual device firmware before trusting stream
# reconstruction; flip SPAD_STREAM_SCHEME if the examples turn out to be right.
SPAD_STREAM_SCHEME = "docs"                    # "docs" | "examples"  (TODO(lab): confirm)
SPAD_CODE_FIFO_OVERFLOW_OFFSET = 13            # relative to P (docs scheme)

# Operating excess bias voltage. There is NO explicit VOP enable/disable TCP
# command -- bias is set via `V,<Vex>` and "disabled" by setting it to 0.
# TODO(lab): confirm the safe operating value and that V,0 fully removes bias.
SPAD_VEX_OPERATING = 5.0


def _decode_marker(offset):
    """Map a docs-scheme marker offset (tag - n_pixels) to (frame, line, dwell).

    Per piimaging.com/doc-pspad: +2 dwell, +3 line, +4 line+dwell, +5 frame,
    +6 frame+dwell, +7 frame+line, +8 frame+line+dwell. Combined markers must
    be applied frame -> line -> dwell.
    """
    table = {2: (0, 0, 1), 3: (0, 1, 0), 4: (0, 1, 1), 5: (1, 0, 0),
             6: (1, 0, 1), 7: (1, 1, 0), 8: (1, 1, 1)}
    return table.get(offset, (0, 0, 0))


class CountRateExceeded(RuntimeError):
    """Raised when a SPAD pixel exceeds the safe count rate."""


# ----------------------------------------------------------------------------
# Step (a): per-pixel dwell clock + 4-bit DO packing + trigger verification
# ----------------------------------------------------------------------------
def make_dwell_clock(scan_layout, pixels_x):
    """Build a per-pixel dwell clock: exactly ``pixels_x`` pulses per line.

    ``pixel_gate`` from ``safe_scan_pattern`` is a *line-long gate* (high for
    the whole active segment). The SPAD dwell-clock SMA input instead needs one
    rising edge per dwell to advance its X index. We place a ~50%-duty square
    wave across each line's active region, giving exactly ``pixels_x`` rising
    edges (the sample before each line start is in flyback = 0, so the first
    edge is a clean 0->1).
    """
    n = scan_layout["pattern_size"]
    dwell = np.zeros(n, dtype=np.uint8)
    starts = scan_layout["line_start_indices"]
    lens = scan_layout["line_samples"]
    for s, length in zip(starts, lens):
        edges = np.linspace(0, int(length), pixels_x + 1).astype(int)
        for k in range(pixels_x):
            a = s + edges[k]
            b = s + edges[k + 1]
            if b <= a:
                continue
            high = a + max(1, (b - a) // 2)  # first half high
            dwell[a:high] = 1
    return dwell


def unpack_do_bits(do_data):
    """Recover (pixel_gate, line_trig, frame_trig) from the 3-bit packed DO."""
    do = np.asarray(do_data)
    pixel_gate = (do >> 0) & 1
    line_trig = (do >> 1) & 1
    frame_trig = (do >> 2) & 1
    return pixel_gate.astype(np.uint8), line_trig.astype(np.uint8), frame_trig.astype(np.uint8)


def pack_spad_do(dwell_clock, line_trig, frame_trig, pixel_gate):
    """Pack the four SPAD digital signals into one DO word (4 lines).

    bit0 = dwell clock, bit1 = line clock, bit2 = frame clock,
    bit3 = pixel gate (acquisition enable). Requires DO port ``line0:3``.
    """
    return (
        (dwell_clock.astype(np.uint32) << 0)
        | (line_trig.astype(np.uint32) << 1)
        | (frame_trig.astype(np.uint32) << 2)
        | (pixel_gate.astype(np.uint32) << 3)
    ).astype(np.uint32)


def setup_spad_pattern(pattern_specs=specs_bidirectional):
    """Build AO data, 4-bit SPAD DO data (with dwell clock), and scan layout."""
    ao_data, do3, scan_layout = setup_and_check_pattern(pattern_specs)
    pixel_gate, line_trig, frame_trig = unpack_do_bits(do3)
    pixels_x = pattern_specs["pixels_x"]
    dwell_clock = make_dwell_clock(scan_layout, pixels_x)
    do_data = pack_spad_do(dwell_clock, line_trig, frame_trig, pixel_gate)
    scan_layout["dwell_clock"] = dwell_clock
    scan_layout["line_trig"] = line_trig
    scan_layout["frame_trig"] = frame_trig
    return ao_data, do_data, scan_layout


def _pulse_widths_samples(signal):
    """Return array of high-run lengths (in samples) for a binary signal."""
    sig = np.asarray(signal).astype(int)
    d = np.diff(sig, prepend=0, append=0)
    rises = np.where(d == 1)[0]
    falls = np.where(d == -1)[0]
    return falls - rises


def verify_triggers(pattern_specs=specs_bidirectional, plot=False):
    """Step (a): assert the dwell/line/frame triggers are correct for the SPAD.

    Returns the scan_layout. Raises AssertionError / ValueError on any problem.
    """
    ao_data, do_data, scan_layout = setup_spad_pattern(pattern_specs)
    rate = scan_layout["rate"]
    nx = pattern_specs["pixels_x"]
    ny = pattern_specs["pixels_y"]

    dwell = scan_layout["dwell_clock"]
    line_trig = scan_layout["line_trig"]
    frame_trig = scan_layout["frame_trig"]
    starts = scan_layout["line_start_indices"]
    lens = scan_layout["line_samples"]

    # 1) exactly pixels_x dwell pulses per active line.
    for i, (s, length) in enumerate(zip(starts, lens)):
        seg = dwell[s:s + int(length)]
        rises = np.sum(np.diff(seg.astype(int), prepend=0) == 1)
        if rises != nx:
            raise AssertionError(
                f"Line {i}: {rises} dwell pulses, expected {nx}")

    # 2) line/frame structure.
    n_line_pulses = np.sum(np.diff(line_trig.astype(int), prepend=0) == 1)
    n_frame_pulses = np.sum(np.diff(frame_trig.astype(int), prepend=0) == 1)
    if n_line_pulses != ny:
        raise AssertionError(f"{n_line_pulses} line pulses, expected {ny}")
    if n_frame_pulses != 1:
        raise AssertionError(f"{n_frame_pulses} frame pulses, expected 1")

    # 3) line pulses coincide with line starts (rising edge at start of line).
    line_rise = np.where(np.diff(line_trig.astype(int), prepend=0) == 1)[0]
    if not np.array_equal(line_rise, starts):
        raise AssertionError("Line clock rising edges do not match line starts")

    # 4) SPAD timing/voltage constraints.
    sample_ns = 1e9 / rate
    min_dwell_w = int(_pulse_widths_samples(dwell).min())
    min_pulse_ns = min_dwell_w * sample_ns
    if min_pulse_ns < SPAD23G_MIN_PULSE_NS:
        raise AssertionError(
            f"Min dwell pulse {min_pulse_ns:.1f} ns < SPAD min "
            f"{SPAD23G_MIN_PULSE_NS} ns")

    print("--- Trigger verification (step a) ---")
    print(f"rate            : {rate} S/s ({sample_ns:.2f} ns/sample)")
    print(f"lines (ny)      : {ny}  (line pulses: {n_line_pulses})")
    print(f"dwell/line (nx) : {nx}  (verified on every line)")
    print(f"frame pulses    : {n_frame_pulses}")
    print(f"min dwell pulse : {min_pulse_ns:.1f} ns "
          f"(SPAD min {SPAD23G_MIN_PULSE_NS} ns)  OK")
    print(f"VOLTAGE NOTE    : keep DO -> SMA <= {SPAD23G_MAX_SMA_VOLTAGE_HIGHZ} V "
          f"into High-Z inputs (5 V only from a 50 ohm source). Verify wiring.")
    print("All trigger checks passed.")

    if plot:
        _plot_triggers(scan_layout, pattern_specs)
    return scan_layout


def _plot_triggers(scan_layout, pattern_specs, max_samples=4000):
    import matplotlib.pyplot as plt
    rate = scan_layout["rate"]
    n = min(max_samples, scan_layout["pattern_size"])
    t = np.arange(n) / rate * 1e3  # ms
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.step(t, scan_layout["dwell_clock"][:n] * 0.8 + 0.0, where="post", label="dwell clock")
    ax.step(t, scan_layout["line_trig"][:n] * 0.8 + 1.2, where="post", label="line clock")
    ax.step(t, scan_layout["frame_trig"][:n] * 0.8 + 2.4, where="post", label="frame clock")
    ax.step(t, scan_layout["pixel_gate"][:n] * 0.8 + 3.6, where="post", label="pixel gate")
    ax.set_xlabel("time (ms)")
    ax.set_yticks([])
    ax.legend(loc="upper right")
    ax.set_title("SPAD trigger signals (first %d samples)" % n)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# Hexagonal pixel geometry (for ISM-realistic sub-pixel shifts + montage)
# ----------------------------------------------------------------------------
def hex_pixel_offsets(n_pixels=SPAD23G_N_PIXELS):
    """Return ``n_pixels`` (dy, dx) offsets on a hex grid, sorted by radius.

    Used to give each SPAD channel a slightly shifted view of the scene (the
    physical origin of ISM). Units are hex-cell fractions; scaled to sub-pixel
    shifts by the caller.
    """
    pts = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = q + 0.5 * r
            y = 0.8660254 * r
            pts.append((x, y))
    pts.sort(key=lambda p: p[0] ** 2 + p[1] ** 2)
    pts = pts[:n_pixels]
    # Return (dy, dx) centered on the array centroid.
    arr = np.array([(y, x) for (x, y) in pts], dtype=float)
    arr -= arr.mean(axis=0, keepdims=True)
    return arr


# ----------------------------------------------------------------------------
# SPAD frame-source abstraction
# ----------------------------------------------------------------------------
class SpadFrameSourceBase:
    """Common interface + safety logic shared by real and fake sources.

    Interface:
        initialize(scan_layout, pattern_specs)
        enable_vop() / disable_vop()
        start() / stop()
        get_frame(timeout) -> np.ndarray (23, ny, nx) of counts, or None
    """

    def __init__(self):
        self.dwell_time_s = None
        self.nx = None
        self.ny = None

    def initialize(self, scan_layout, pattern_specs):
        self.nx = pattern_specs["pixels_x"]
        self.ny = pattern_specs["pixels_y"]
        # Time the SPAD integrates per dwell = active samples per pixel / rate.
        spp = scan_layout["max_line_samples"] / self.nx
        self.dwell_time_s = spp / scan_layout["rate"]

    # -- safety helpers (SDK-independent) ------------------------------------
    @staticmethod
    def mask_hot_pixels(cube):
        """Set hot-pixel channels to NaN (applied before any analysis)."""
        masked = cube.astype(float)
        masked[HOT_PIXELS] = np.nan
        return masked

    def counts_to_cps(self, cube):
        return cube / self.dwell_time_s

    def check_count_rates(self, cube):
        """Raise CountRateExceeded if any (non-hot) pixel exceeds the safe rate.

        ``cube`` should already be hot-pixel-masked (NaN). NaNs are ignored.
        """
        cps = self.counts_to_cps(cube)
        peak = np.nanmax(cps) if np.isfinite(np.nanmax(cps)) else 0.0
        if peak >= SPAD23G_WARNING_FRACTION * SPAD23G_SAFE_COUNT_RATE_CPS:
            print(f"[SPAD][warn] peak rate {peak:.2e} cps "
                  f"(>{SPAD23G_WARNING_FRACTION:.0%} of safe limit)")
        if peak > SPAD23G_SAFE_COUNT_RATE_CPS:
            # locate offending channel (hot/masked channels are all-NaN)
            chan_max = np.array([np.nanmax(c) if np.any(np.isfinite(c)) else -np.inf
                                 for c in cps])
            ch = int(np.argmax(chan_max))
            raise CountRateExceeded(
                f"channel {ch} peak {peak:.2e} cps exceeds safe limit "
                f"{SPAD23G_SAFE_COUNT_RATE_CPS:.2e} cps")

    @staticmethod
    def detect_pileup(prev_cube, curr_cube, threshold_fraction=0.2):
        """Return True if counts dropped sharply under constant illumination.

        In pile-up the measured rate *falls* as true flux rises -- a dangerous
        condition that hides over-illumination.
        """
        prev = np.nansum(prev_cube)
        curr = np.nansum(curr_cube)
        if prev <= 0:
            return False
        return bool((prev - curr) / (prev + 1e-6) > threshold_fraction)

    # -- to be provided by subclasses ----------------------------------------
    def enable_vop(self):
        raise NotImplementedError

    def disable_vop(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def get_frame(self, timeout):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class FakeSpad23FrameSource(SpadFrameSourceBase):
    """Simulated SPAD23 for dev: ISM-realistic 23-channel synthetic frames.

    Each channel sees a sub-pixel-shifted view of the scene (hex geometry),
    Poisson-sampled into photon counts. Hot pixel (index 7) is a flat dark
    channel. Optionally injects an over-rate condition to exercise the safety
    path.
    """

    def __init__(self, image_type="resolution_target", shift_px=1.5,
                 peak_fraction=0.2, simulate_overexposure_after=None):
        super().__init__()
        self.image_type = image_type
        self.shift_px = shift_px
        self.peak_fraction = peak_fraction  # peak rate as fraction of safe limit
        self.simulate_overexposure_after = simulate_overexposure_after
        self._channels = None
        self._frame_count = 0
        self._vop = False
        self._rng = np.random.default_rng()

    def initialize(self, scan_layout, pattern_specs):
        super().initialize(scan_layout, pattern_specs)
        base = self._load_base_image(self.ny, self.nx)
        offsets = hex_pixel_offsets(SPAD23G_N_PIXELS) * self.shift_px
        # Mean counts that put the brightest pixel at peak_fraction of safe rate.
        peak_counts = self.peak_fraction * SPAD23G_SAFE_COUNT_RATE_CPS * self.dwell_time_s
        from scipy.ndimage import shift as nd_shift
        channels = np.zeros((SPAD23G_N_PIXELS, self.ny, self.nx), dtype=float)
        for c in range(SPAD23G_N_PIXELS):
            dy, dx = offsets[c]
            shifted = nd_shift(base, (dy, dx), order=1, mode="nearest")
            channels[c] = np.clip(shifted, 0, None) * peak_counts
        self._mean_channels = channels
        self._frame_count = 0

    def _load_base_image(self, ny, nx):
        """Load the resolution target / grid, with a synthetic fallback."""
        try:
            from pathlib import Path
            from PIL import Image
            base_path = Path(__file__).resolve().parent
            p = base_path / "../../scanImaging/instrument/virtual/images/radial-sine-144.png"
            img = Image.open(p).convert("L").resize((nx, ny), resample=Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32)
            if arr.max() != 0:
                arr /= arr.max()
            if self.image_type == "grid":
                raise FileNotFoundError  # fall through to synthetic grid
            return arr
        except Exception:
            return self._synthetic_image(ny, nx)

    def _synthetic_image(self, ny, nx):
        if self.image_type == "grid":
            arr = np.zeros((ny, nx), dtype=np.float32)
            arr[:, ::max(1, nx // 10)] = 1.0
            arr[::max(1, ny // 10), :] = 1.0
            return arr
        # radial sine resolution target
        yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
        cy, cx = ny / 2, nx / 2
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        ang = np.arctan2(yy - cy, xx - cx)
        arr = 0.5 * (1 + np.sin(ang * 12) * np.cos(r / 4))
        return np.clip(arr, 0, 1).astype(np.float32)

    def enable_vop(self):
        self._vop = True
        print("[FakeSPAD] VOP enabled (simulated)")

    def disable_vop(self):
        self._vop = False
        print("[FakeSPAD] VOP disabled (simulated)")

    def start(self):
        print("[FakeSPAD] continuous scanning started (simulated)")

    def get_frame(self, timeout):
        # Pace to roughly one frame per scan period for a live feel.
        time.sleep(min(timeout, 0.05))
        cube = self._rng.poisson(self._mean_channels).astype(float)
        # Hot pixel: flat dark channel (counts from its dark rate).
        for hp in HOT_PIXELS:
            cube[hp] = self._rng.poisson(
                SPAD23G_DARK_CPS_HOT * self.dwell_time_s, size=cube[hp].shape)
        self._frame_count += 1
        if (self.simulate_overexposure_after is not None
                and self._frame_count >= self.simulate_overexposure_after):
            cube[5] *= 50.0  # drive channel 5 over the safe limit
        return cube

    def stop(self):
        print("[FakeSPAD] scanning stopped (simulated)")


class Spad23FrameSource(SpadFrameSourceBase):
    """Real SPAD23 over the pSPAD TCP interface (lab machine).

    Control is a TCP socket to the running pSPAD application on a single
    configurable port (default 9999; text & binary share it) -- there is no
    separate Python SDK (see spad23-integration-notes.md). Two modes:

    * ``mode="scanning"`` (binary ``CS``): the device assembles a (23, ny, nx)
      intensity image per call. Frame-synced to the NI-DAQ frame clock
      (``ext_frame_clk=1``); with ``measurement_time=0`` it also uses the
      EXTERNAL (NI-DAQ) dwell clock. No microtime. Recommended first milestone.
    * ``mode="stream"`` (binary ``SB``): per-photon events; the NI-DAQ
      dwell/line/frame edges arrive as marker events and the laser clock gives
      the fine timestamp (microtime). Frames are reconstructed in software.
      Path for the FLIM + ISM workflow. NOTE: docs vs example marker numbering
      conflict -- see SPAD_STREAM_SCHEME.

    The socket mechanics and response parsing are concrete; the genuine unknowns
    (VOP via ``V`` has no explicit enable/disable; USB-speed check; stream code
    scheme; scanning plane count) are marked ``TODO(lab)``. Safety logic
    (count-rate / pile-up / hot-pixel mask) is inherited. Follow the
    startup/shutdown order in for-code-dev-spad23-safety-guidelines.md §2-3.
    """

    def __init__(self, image_type=None, mode="scanning", measurement_time=None,
                 port=SPAD_PORT):
        super().__init__()
        if mode not in ("scanning", "stream"):
            raise ValueError("mode must be 'scanning' or 'stream'")
        self.mode = mode
        self.port = port
        # scanning: us/pixel (0 => external dwell clock). stream: ms.
        if measurement_time is None:
            measurement_time = 0.0 if mode == "scanning" else 1000.0
        self.measurement_time = measurement_time
        self._sock = None

    # -- TCP plumbing --------------------------------------------------------
    def _connect(self, port):
        self._sock = socket.create_connection(
            (SPAD_HOST, port), timeout=SPAD_TIMEOUT_S)
        banner = self._sock.recv(8192)
        print("[SPAD] connected:", banner.decode("utf8", "replace").strip())

    def _send(self, command):
        self._sock.sendall(bytes(command + "\n", "utf8"))

    def _read_until_done(self):
        """Read a streamed response until it ends with DONE; raise on ERROR."""
        buf = bytearray()
        while True:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("SPAD closed the connection mid-stream")
            buf.extend(chunk)
            if buf[-4:] == b"DONE":
                return bytes(buf[:-4])
            if buf[-5:] == b"ERROR":
                raise RuntimeError(f"SPAD returned ERROR: {bytes(buf[-160:])!r}")

    def initialize(self, scan_layout, pattern_specs):
        super().initialize(scan_layout, pattern_specs)
        self._connect(self.port)
        # TODO(lab): verify USB 3.0 link speed (R command); refuse high-rate
        # stream on USB 2.0.
        if self.mode == "stream":
            # Ensure the TDC is calibrated before timestamped streaming.
            self._send("T,v,1")
            if self._sock.recv(8192).decode("utf8", "replace").startswith(
                    "TDC calibration is invalid"):
                self._send("T,c,1")
                print("[SPAD] TDC calibrate:",
                      self._sock.recv(8192).decode("utf8", "replace").strip())

    # -- VOP: no explicit enable/disable command; set excess bias via `V` -----
    def enable_vop(self):
        # Docs: no explicit VOP-enable command -- set the excess bias voltage.
        # TODO(lab): confirm Vex value (and whether CALIB breakdown is needed).
        self._send(f"V,{SPAD_VEX_OPERATING}")

    def disable_vop(self):
        # "Disable" by setting excess voltage to 0. MUST run on every exit.
        # TODO(lab): confirm V,0 fully removes bias (else use the menu / QUIT).
        if self._sock is not None:
            try:
                self._send("V,0")
            except OSError:
                pass  # best-effort on a broken socket

    def start(self):
        # Acquisition is issued per-frame inside get_frame() (the CS/SB command
        # both starts and runs a measurement), so nothing to arm here.
        pass

    def get_frame(self, timeout):
        if self.mode == "scanning":
            return self._get_frame_scanning()
        return self._get_frame_stream()

    def _get_frame_scanning(self):
        """One CS measurement -> (23, ny, nx) counts (frame-synced to NI-DAQ)."""
        nx, ny = self.nx, self.ny
        # CS,<dwell_us>,<nr_frames>,<pix_x>,<pix_y>,<ext_frame_clk>
        self._send(f"CS,{self.measurement_time},1,{nx},{ny},1")
        data = self._read_until_done()
        plane = nx * ny
        cube = np.zeros((SPAD23G_N_PIXELS, ny, nx), dtype=float)
        # 23 consecutive pixel-planes, one byte per pixel per channel.
        # TODO(lab): confirm plane order / count (vendor example loops 22).
        for c in range(SPAD23G_N_PIXELS):
            seg = data[c * plane:(c + 1) * plane]
            if len(seg) < plane:
                break
            cube[c] = np.frombuffer(seg, dtype=np.uint8).reshape(ny, nx).astype(float)
        return cube

    def _get_frame_stream(self):
        """One SB measurement -> (23, ny, nx) counts reconstructed from markers.

        Photons (docs scheme: tag 0..22 == channel) are binned into (channel,
        y, x) using the dwell/line/frame markers. Combined markers (frame+line+
        dwell etc.) are applied frame -> line -> dwell. The per-photon ``fine``
        24-bit microtime (bytes 3..5) is available and should be retained for
        FLIM in the main workflow.
        TODO(lab): confirm SPAD_STREAM_SCHEME, retain microtime, handle
        bidirectional line reversal, and stream continuously instead of per-call.
        """
        nx, ny = self.nx, self.ny
        self._send(f"SB,{self.measurement_time}")
        data = self._read_until_done()
        cube = np.zeros((SPAD23G_N_PIXELS, ny, nx), dtype=float)
        n = SPAD23G_N_PIXELS
        # Markers emitted at the START of each frame/line/dwell, so first line
        # marker -> y=0 and first dwell marker -> x=0. TODO(lab): confirm.
        x = y = -1
        seen_frame = False
        for i in range(0, len(data) - 6 + 1, 6):
            tag = data[i]
            if tag < n:  # photon on channel == tag
                # fine = int.from_bytes(data[i+3:i+6], 'big')  # microtime (ps) -> FLIM
                if seen_frame and 0 <= y < ny and 0 <= x < nx:
                    cube[tag, y, x] += 1
                continue
            offset = tag - n
            if offset == SPAD_CODE_FIFO_OVERFLOW_OFFSET:
                print("[SPAD][warn] FIFO overflow -- data lost")
                continue
            is_frame, is_line, is_dwell = _decode_marker(offset)
            if is_frame:
                seen_frame = True
                y = x = -1
            if is_line:
                y += 1
                x = -1
            if is_dwell:
                x += 1
        return cube

    def stop(self):
        try:
            self.disable_vop()
        finally:
            if self._sock is not None:
                self._sock.close()
                self._sock = None
            print("[SPAD] socket closed")


# ----------------------------------------------------------------------------
# Acquisition thread
# ----------------------------------------------------------------------------
def spad_detector_thread(source, scan_layout, pattern_specs, stop_event,
                         image_queue, frame_holder=None):
    """Pull assembled SPAD frames, run safety checks, push cubes to the queue."""
    prev_cube = None
    while not stop_event.is_set():
        try:
            cube = source.get_frame(timeout=5.0)
        except Exception as exc:  # noqa: BLE001 - want VOP off on any failure
            print(f"[SPAD] get_frame failed: {exc}")
            stop_event.set()
            break
        if cube is None:
            continue

        masked = SpadFrameSourceBase.mask_hot_pixels(cube)
        try:
            source.check_count_rates(masked)
            if prev_cube is not None and source.detect_pileup(prev_cube, masked):
                raise CountRateExceeded("pile-up suspected (counts dropped sharply)")
        except CountRateExceeded as exc:
            print(f"[SPAD][SAFETY] {exc} -> disabling VOP and stopping")
            try:
                source.disable_vop()
            finally:
                stop_event.set()
            break
        prev_cube = masked

        if frame_holder is not None:
            frame_holder[0] = masked
        try:
            if image_queue.full():
                image_queue.get_nowait()
            image_queue.put_nowait(masked)
        except queue.Full:
            pass
    print("[SPAD] detector thread exiting")


# ----------------------------------------------------------------------------
# Montage display (23 channels + summed intensity)
# ----------------------------------------------------------------------------
class MontageDisplay:
    def __init__(self, n_channels=SPAD23G_N_PIXELS):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.n_channels = n_channels
        self.ncols = 5
        self.nrows = 5  # 25 panels: 23 channels + 1 sum + 1 spare

    def initialize_display(self, pattern_specs):
        ny, nx = pattern_specs["pixels_y"], pattern_specs["pixels_x"]
        self.fig, axes = self.plt.subplots(
            self.nrows, self.ncols, figsize=(11, 11))
        self.axes = axes.ravel()
        self.images = []
        blank = np.zeros((ny, nx))
        for i, ax in enumerate(self.axes):
            im = ax.imshow(blank, cmap="inferno", origin="upper", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if i < self.n_channels:
                ax.set_title(f"ch {i}" + (" (hot)" if i in HOT_PIXELS else ""),
                             fontsize=7)
            elif i == self.n_channels:
                ax.set_title("sum", fontsize=8, color="tab:cyan")
            else:
                ax.axis("off")
            self.images.append(im)
        self.fig.suptitle("SPAD23 live channels")
        self.plt.tight_layout()
        self.plt.ion()
        # Clear matplotlib default key bindings so ours don't conflict.
        for k in list(self.plt.rcParams.keys()):
            if k.startswith("keymap."):
                self.plt.rcParams[k] = []

    def display_loop(self, stop_event, image_queue, frame_holder=None):
        while not stop_event.is_set():
            try:
                cube = image_queue.get(timeout=0.5)
            except queue.Empty:
                self.plt.pause(0.001)
                continue
            if frame_holder is not None:
                frame_holder[0] = cube
            for i in range(self.n_channels):
                ch = cube[i]
                self.images[i].set_data(np.nan_to_num(ch))
                finite = ch[np.isfinite(ch)]
                if finite.size:
                    self.images[i].set_clim(finite.min(), max(finite.max(), 1))
            summed = np.nansum(cube, axis=0)
            self.images[self.n_channels].set_data(summed)
            self.images[self.n_channels].set_clim(summed.min(), max(summed.max(), 1))
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.plt.pause(0.001)

    def setup_key_handler(self, stop_event, frame_holder, scan_layout, pattern_specs):
        def on_key_press(event):
            key = event.key
            if key in ("x", "escape"):
                print("Exiting...")
                stop_event.set()
            elif key == "p":
                print(f"nx={pattern_specs['pixels_x']} ny={pattern_specs['pixels_y']} "
                      f"rate={scan_layout['rate']} channels={SPAD23G_N_PIXELS}")
            elif key == "S":
                cube = frame_holder[0]
                if cube is None:
                    print("No frame yet, cannot save.")
                    return
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                np.savez(
                    f"spad_cube_{ts}.npz",
                    cube=cube,
                    pixels_x=pattern_specs["pixels_x"],
                    pixels_y=pattern_specs["pixels_y"],
                    rate=scan_layout["rate"],
                    hot_pixels=np.array(HOT_PIXELS),
                    bidirectional=pattern_specs.get("bidirectional", True),
                )
                print(f"Saved spad_cube_{ts}.npz")

        self.fig.canvas.mpl_connect("key_press_event", on_key_press)

    def close(self):
        self.plt.ioff()
        self.plt.close(self.fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run(pattern_specs=specs_bidirectional, image_type="resolution_target",
        hardware="fake", simulate_overexposure_after=None, spad_mode="scanning"):
    ao_data, do_data, scan_layout = setup_spad_pattern(pattern_specs)
    rate = int(pattern_specs["rate"])
    total_scan_samples = ao_data.shape[0]
    if total_scan_samples != do_data.shape[0]:
        raise RuntimeError("AO/DO sample count mismatch!")

    stop_event = threading.Event()
    image_queue = queue.Queue(maxsize=2)
    frame_holder = [None]

    # AO + DO only (no AI). DO needs 4 lines for the dwell clock.
    if hardware == "real":
        factory = NIDaqTaskFactory()
        factory.do_port = "port0/line0:3"
        factory.set_rate(rate)
        ao_task = factory.create_ao_wrapper(total_scan_samples)
        do_task = factory.create_do_wrapper(total_scan_samples)
        source = Spad23FrameSource(image_type=image_type, mode=spad_mode)
    else:
        ao_task = FakeGalvoAoWrapper()
        do_task = FakeDoWrapper()
        source = FakeSpad23FrameSource(
            image_type=image_type,
            simulate_overexposure_after=simulate_overexposure_after)

    ao_task.write(ao_data)
    do_task.write(do_data)
    source.initialize(scan_layout, pattern_specs)

    display = MontageDisplay()
    display.initialize_display(pattern_specs)
    display.setup_key_handler(stop_event, frame_holder, scan_layout, pattern_specs)

    t_det = threading.Thread(
        target=spad_detector_thread,
        args=(source, scan_layout, pattern_specs, stop_event, image_queue),
        kwargs={"frame_holder": frame_holder},
        daemon=True,
    )

    print("\n--- SPAD23 live view (focus the image window) ---")
    print("  S: save current 23-channel cube (.npz)   p: print params   x / Esc: exit")
    print(f"  dwell time/pixel ~ {source.dwell_time_s * 1e6:.2f} us")
    print("-------------------------------------------------\n")

    # Safety: VOP enabled in try, disabled unconditionally in finally.
    source.enable_vop()
    try:
        source.start()
        do_task.start()
        ao_task.start()  # AO last: it generates the sample clock for the DO.
        time.sleep(0.05)
        t_det.start()
        display.display_loop(stop_event, image_queue, frame_holder)
    finally:
        stop_event.set()
        t_det.join(timeout=2.0)
        try:
            source.disable_vop()
        finally:
            source.stop()
            do_task.stop()
            ao_task.stop()
            display.close()
        print("All threads stopped; VOP disabled.")


def test():
    """Lightweight self-checks (no hardware, no GUI)."""
    ao_data, do_data, scan_layout = setup_spad_pattern(specs_bidirectional)
    assert ao_data.shape[0] == do_data.shape[0]
    nx = specs_bidirectional["pixels_x"]
    ny = specs_bidirectional["pixels_y"]
    # dwell pulses per line
    dwell = scan_layout["dwell_clock"]
    for s, length in zip(scan_layout["line_start_indices"], scan_layout["line_samples"]):
        seg = dwell[s:s + int(length)]
        assert np.sum(np.diff(seg.astype(int), prepend=0) == 1) == nx
    # frame source shape
    src = FakeSpad23FrameSource()
    src.initialize(scan_layout, specs_bidirectional)
    cube = src.get_frame(timeout=0.0)
    assert cube.shape == (SPAD23G_N_PIXELS, ny, nx), cube.shape
    masked = SpadFrameSourceBase.mask_hot_pixels(cube)
    assert np.all(np.isnan(masked[HOT_PIXELS[0]]))
    print("test(): all assertions passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NIDAQ galvo scan with SPAD23 detector")
    parser.add_argument("--real", action="store_true",
                        help="Use real NIDAQ + SPAD23 over TCP (lab machine)")
    parser.add_argument("--spad-mode", choices=["scanning", "stream"],
                        default="scanning",
                        help="Real SPAD mode: scanning (intensity) or stream "
                             "(markers + microtime, for FLIM/ISM). Default: scanning")
    parser.add_argument("--check-triggers", action="store_true",
                        help="Step (a): verify dwell/line/frame triggers and exit")
    parser.add_argument("--plot-triggers", action="store_true",
                        help="With --check-triggers, also plot the signals")
    parser.add_argument("--image", choices=["resolution_target", "grid"],
                        default="resolution_target",
                        help="Simulated image type (default: resolution_target)")
    parser.add_argument("--unidirectional", action="store_true",
                        help="Use unidirectional scan pattern (default: bidirectional)")
    parser.add_argument("--simulate-overexposure", type=int, default=None,
                        metavar="N",
                        help="(sim) trigger an over-rate condition after N frames")
    parser.add_argument("--self-test", action="store_true",
                        help="Run lightweight self-checks and exit")
    args = parser.parse_args()

    pattern = specs_unidirectional if args.unidirectional else specs_bidirectional

    if args.self_test:
        test()
    elif args.check_triggers:
        verify_triggers(pattern_specs=pattern, plot=args.plot_triggers)
    else:
        run(pattern_specs=pattern,
            image_type=args.image,
            hardware="real" if args.real else "fake",
            simulate_overexposure_after=args.simulate_overexposure,
            spad_mode=args.spad_mode)
