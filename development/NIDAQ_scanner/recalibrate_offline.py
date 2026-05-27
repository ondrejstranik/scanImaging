"""
Offline recalibration tool for saved .npz calibration files.

Usage:
    python recalibrate_offline.py <file.npz>

Keyboard controls (focus the image window):
  q/a       global lag ±1 sample    w/s  global lag ±5 samples
  e/d       even lag ±1 sample
  t/g       odd lag ±1 sample
  A         auto even/odd calibrate (cross-correlation)
  r         toggle ROI rectangle selector (drag to select)
  c         clear ROI
  p         print current params (copy-pasteable)
  S         save PNG
  x / Esc   exit

If raw_lines are present in the .npz, lag changes trigger a full sample-level
reconstruction. Otherwise pixel-level row shifts are applied (less precise).
"""

import sys
import datetime
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # use a backend that supports key events reliably
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from lag_calibration import estimate_even_odd_shift, reconstruct_from_raw, shift_rows


class CalibrationState:
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)

        self.frame0 = data['frame'].astype(np.float32)
        self.ny, self.nx = self.frame0.shape

        raw = data.get('raw_lines', None)
        self.raw_lines = raw if (raw is not None and raw.ndim == 2) else None

        self.bidirectional = bool(data.get('bidirectional', True))
        self.rate = float(data.get('rate', 100000))
        self.samples_per_pixel = float(data.get('samples_per_pixel', 1.0))

        self.lag0 = {
            'global': int(data.get('scanner_lag_samples', 0)),
            'even':   int(data.get('scanner_lag_samples_even', 0)),
            'odd':    int(data.get('scanner_lag_samples_odd', 0)),
            'xy_scale': float(data.get('xy_scale', 1.0)),
        }
        self.lag = dict(self.lag0)
        self.roi = None

        mode = "sample-level (raw_lines present)" if self.raw_lines is not None else "pixel-level fallback"
        print(f"Loaded {npz_path}")
        print(f"  Frame: {self.ny}×{self.nx}   Mode: {mode}")
        print(f"  Saved lag: global={self.lag0['global']}  even={self.lag0['even']}  odd={self.lag0['odd']}")

    def current_frame(self, include_roi=True):
        frame = self._make_frame()
        if include_roi and self.roi is not None:
            x0, y0, x1, y1 = self.roi
            x0 = max(0, min(x0, frame.shape[1]))
            x1 = max(x0 + 1, min(x1, frame.shape[1]))
            y0 = max(0, min(y0, frame.shape[0]))
            y1 = max(y0 + 1, min(y1, frame.shape[0]))
            frame = frame[y0:y1, x0:x1]
        return frame

    def _make_frame(self):
        if self.raw_lines is not None:
            return reconstruct_from_raw(
                self.raw_lines, self.nx,
                bidirectional=self.bidirectional,
                global_lag=self.lag['global'],
                even_lag=self.lag['even'],
                odd_lag=self.lag['odd'],
                xy_scale=self.lag['xy_scale'],
            )
        else:
            spx = max(self.samples_per_pixel, 1.0)
            g_px = (self.lag['global'] - self.lag0['global']) / spx
            e_px = (self.lag['even']   - self.lag0['even'])   / spx
            o_px = (self.lag['odd']    - self.lag0['odd'])    / spx
            return shift_rows(self.frame0, g_px, e_px, o_px)


class OfflineCalibrationUI:
    def __init__(self, state: CalibrationState):
        self.state = state
        self.roi_mode = False

        # Clear default matplotlib key bindings
        for k in list(plt.rcParams.keys()):
            if k.startswith('keymap.'):
                plt.rcParams[k] = []

        self.fig, self.ax = plt.subplots()
        frame = state.current_frame()
        self.im = self.ax.imshow(frame, cmap='gray', origin='upper', aspect='auto')
        plt.colorbar(self.im, ax=self.ax)
        self._update_title()

        self.rs = RectangleSelector(
            self.ax, self._on_roi_select,
            interactive=True, button=[1],
            minspanx=3, minspany=3,
        )
        self.rs.set_active(False)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _update_display(self):
        frame = self.state.current_frame()
        self.im.set_data(frame)
        self.im.set_clim(frame.min(), frame.max())
        self._update_title()
        self.fig.canvas.draw_idle()

    def _update_title(self):
        s = self.state
        lag = s.lag
        title = (f"g={lag['global']}  e={lag['even']}  o={lag['odd']}  "
                 f"scale={lag['xy_scale']:.2f}")
        if s.roi:
            x0, y0, x1, y1 = s.roi
            title += f"  ROI({x0},{y0})-({x1},{y1})"
        if self.roi_mode:
            title += "  [ROI MODE]"
        self.ax.set_title(title)

    def _on_roi_select(self, eclick, erelease):
        if eclick.xdata is None:
            return
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        self.state.roi = (int(min(x0, x1)), int(min(y0, y1)),
                          int(max(x0, x1)), int(max(y0, y1)))
        self._update_display()
        print(f"ROI set: {self.state.roi}")

    def _on_key(self, event):
        key = event.key
        if key is None:
            return
        s = self.state

        if key == 'q':   s.lag['global'] += 1
        elif key == 'a': s.lag['global'] -= 1
        elif key == 'w': s.lag['global'] += 5
        elif key == 's': s.lag['global'] -= 5
        elif key == 'e': s.lag['even'] += 1
        elif key == 'd': s.lag['even'] -= 1
        elif key == 't': s.lag['odd'] += 1
        elif key == 'g': s.lag['odd'] -= 1
        elif key == 'A':
            self._auto_calibrate()
            return
        elif key == 'r':
            self.roi_mode = not self.roi_mode
            self.rs.set_active(self.roi_mode)
            print(f"ROI mode: {'ON — drag to select region' if self.roi_mode else 'OFF'}")
            self._update_title()
            self.fig.canvas.draw_idle()
            return
        elif key == 'c':
            s.roi = None
            self._update_display()
            print("ROI cleared")
            return
        elif key == 'p':
            self._print_params()
            return
        elif key == 'S':
            self._save_png()
            return
        elif key in ('x', 'escape'):
            self._print_params()
            plt.close(self.fig)
            return
        else:
            return

        self._update_display()

    def _auto_calibrate(self):
        s = self.state
        frame = s.current_frame(include_roi=False)
        shift_px = estimate_even_odd_shift(frame)
        spx = s.samples_per_pixel if s.samples_per_pixel > 0 else 1.0
        delta = round(shift_px * spx)
        s.lag['odd'] += delta
        print(f"[Auto] even/odd shift: {shift_px:+.2f} px → odd lag adjusted by {delta:+d} samples")
        self._update_display()

    def _print_params(self):
        s = self.state
        rate = s.rate
        lag = s.lag
        print("\n--- Offline calibration params ---")
        print(f"scanner_lag_samples:      {lag['global']:4d}   ({lag['global']/rate*1e6:.1f} µs)")
        print(f"scanner_lag_samples_even: {lag['even']:4d}   ({lag['even']/rate*1e6:.1f} µs)")
        print(f"scanner_lag_samples_odd:  {lag['odd']:4d}   ({lag['odd']/rate*1e6:.1f} µs)")
        print(f"xy_scale:                 {lag['xy_scale']:.4f}")
        print("\n# Paste into specs_bidirectional:")
        print(f'"scanner_lag_samples": {lag["global"]},')
        print(f'"scanner_lag_samples_even": {lag["even"]},')
        print(f'"scanner_lag_samples_odd": {lag["odd"]},')
        print(f'"xy_scale": {lag["xy_scale"]},')
        if s.roi:
            print(f"\nROI (pixels): {s.roi}")
        print()

    def _save_png(self):
        frame = self.state.current_frame()
        f_min, f_max = frame.min(), frame.max()
        arr_norm = (frame - f_min) / (f_max - f_min) if f_max > f_min else frame
        arr_u8 = (arr_norm * 255).clip(0, 255).astype(np.uint8)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"recalibrated_{timestamp}.png"
        try:
            from PIL import Image
            Image.fromarray(arr_u8).save(path)
        except ImportError:
            plt.imsave(path, arr_norm, cmap='gray')
        print(f"Saved {path}")

    def run(self):
        print("\n--- Offline calibration controls ---")
        print("  q/a: global ±1    w/s: global ±5")
        print("  e/d: even ±1      t/g: odd ±1")
        print("  A: auto even/odd  r: ROI mode  c: clear ROI")
        print("  p: print params   S: save PNG   x/Esc: exit")
        print("------------------------------------\n")
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python recalibrate_offline.py <calibration_*.npz>")
        sys.exit(1)

    npz_path = sys.argv[1]
    state = CalibrationState(npz_path)
    ui = OfflineCalibrationUI(state)
    ui.run()


if __name__ == '__main__':
    main()
