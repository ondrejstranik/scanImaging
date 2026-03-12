#!/usr/bin/env python3
"""
Interactive timing compensation adjustment tool.

This script allows live adjustment of scanner timing compensation parameters
while viewing the resulting images in real-time. Use keyboard controls to
fine-tune the offset values until optimal alignment is achieved.

Usage:
    python interactive_timing_adjustment.py

Keyboard Controls:
    Global lag (scanner_lag_samples):
        q/a: Increase/decrease by 1
        w/s: Increase/decrease by 5

    Even line offset (scanner_lag_samples_even):
        e/d: Increase/decrease by 1
        r/f: Increase/decrease by 5

    Odd line offset (scanner_lag_samples_odd):
        t/g: Increase/decrease by 1
        y/h: Increase/decrease by 5

    Other:
        p: Print current parameters
        c: Clear/reset all offsets to 0
        x: Exit

The display updates automatically when parameters change.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from galvo_scan_read_test import (
    FakeAiWrapper, setup_and_check_pattern, detector_thread,
    specs_bidirectional
)
import threading
import queue
import time
from copy import deepcopy
import sys
import select
import termios
import tty


class InteractiveAdjustment:
    """Interactive timing compensation adjustment with live preview."""

    def __init__(self, simulate_lag=True):
        # FIXED simulation parameters (simulate hardware lag)
        if simulate_lag:
            self.sim_params = {
                "scanner_lag_samples": 10,
                "scanner_lag_samples_even": 8,
                "scanner_lag_samples_odd": -3,
            }
        else:
            self.sim_params = {
                "scanner_lag_samples": 0,
                "scanner_lag_samples_even": 0,
                "scanner_lag_samples_odd": 0,
            }

        # ADJUSTABLE compensation parameters (what you tune)
        self.params = {
            "scanner_lag_samples": 0,
            "scanner_lag_samples_even": 0,
            "scanner_lag_samples_odd": 0,
        }

        # Thread control
        self.stop_event = threading.Event()
        self.params_changed = threading.Event()
        self.image_queue = queue.Queue(maxsize=2)
        self.current_frame = None

        # For keyboard input
        self.old_settings = None

    def setup_terminal(self):
        """Setup terminal for non-blocking keyboard input."""
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def restore_terminal(self):
        """Restore terminal to original settings."""
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def get_key_nonblocking(self):
        """Get keyboard input without blocking."""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def print_params(self):
        """Print current timing parameters."""
        print("\n" + "="*60)
        print("SIMULATED LAG (fixed):")
        print("="*60)
        print(f"  scanner_lag_samples:       {self.sim_params['scanner_lag_samples']:4d}")
        print(f"  scanner_lag_samples_even:  {self.sim_params['scanner_lag_samples_even']:4d}")
        print(f"  scanner_lag_samples_odd:   {self.sim_params['scanner_lag_samples_odd']:4d}")

        sim_even = self.sim_params['scanner_lag_samples'] + self.sim_params['scanner_lag_samples_even']
        sim_odd = self.sim_params['scanner_lag_samples'] + self.sim_params['scanner_lag_samples_odd']
        print(f"  → Even lines: {sim_even:4d} samples, Odd lines: {sim_odd:4d} samples")

        print("\n" + "="*60)
        print("COMPENSATION (adjustable):")
        print("="*60)
        print(f"  scanner_lag_samples:       {self.params['scanner_lag_samples']:4d}")
        print(f"  scanner_lag_samples_even:  {self.params['scanner_lag_samples_even']:4d}")
        print(f"  scanner_lag_samples_odd:   {self.params['scanner_lag_samples_odd']:4d}")

        # Calculate effective compensation offsets
        comp_even = self.params['scanner_lag_samples'] + self.params['scanner_lag_samples_even']
        comp_odd = self.params['scanner_lag_samples'] + self.params['scanner_lag_samples_odd']

        print(f"\n  → Even lines: {comp_even:4d} samples ({comp_even/200000*1e6:.1f} µs @ 200kHz)")
        print(f"  → Odd lines:  {comp_odd:4d} samples ({comp_odd/200000*1e6:.1f} µs @ 200kHz)")

        # Show residual error
        print("\n" + "="*60)
        print("RESIDUAL ERROR:")
        print("="*60)
        error_even = sim_even - comp_even
        error_odd = sim_odd - comp_odd
        print(f"  Even lines: {error_even:+4d} samples ({error_even/200000*1e6:+.1f} µs)")
        print(f"  Odd lines:  {error_odd:+4d} samples ({error_odd/200000*1e6:+.1f} µs)")
        if error_even == 0 and error_odd == 0:
            print("  ✓ Perfect compensation!")
        print("="*60 + "\n")

    def handle_keyboard(self):
        """Handle keyboard input for parameter adjustment."""
        while not self.stop_event.is_set():
            key = self.get_key_nonblocking()

            if key is None:
                time.sleep(0.05)
                continue

            changed = False

            # Global lag adjustments
            if key == 'q':
                self.params['scanner_lag_samples'] += 1
                changed = True
            elif key == 'a':
                self.params['scanner_lag_samples'] -= 1
                changed = True
            elif key == 'w':
                self.params['scanner_lag_samples'] += 5
                changed = True
            elif key == 's':
                self.params['scanner_lag_samples'] -= 5
                changed = True

            # Even line adjustments
            elif key == 'e':
                self.params['scanner_lag_samples_even'] += 1
                changed = True
            elif key == 'd':
                self.params['scanner_lag_samples_even'] -= 1
                changed = True
            elif key == 'r':
                self.params['scanner_lag_samples_even'] += 5
                changed = True
            elif key == 'f':
                self.params['scanner_lag_samples_even'] -= 5
                changed = True

            # Odd line adjustments
            elif key == 't':
                self.params['scanner_lag_samples_odd'] += 1
                changed = True
            elif key == 'g':
                self.params['scanner_lag_samples_odd'] -= 1
                changed = True
            elif key == 'y':
                self.params['scanner_lag_samples_odd'] += 5
                changed = True
            elif key == 'h':
                self.params['scanner_lag_samples_odd'] -= 5
                changed = True

            # Other commands
            elif key == 'p':
                self.print_params()
            elif key == 'c':
                self.params['scanner_lag_samples'] = 0
                self.params['scanner_lag_samples_even'] = 0
                self.params['scanner_lag_samples_odd'] = 0
                print("\n✓ All offsets cleared to 0")
                changed = True
            elif key == 'x':
                print("\n✓ Exiting...")
                self.stop_event.set()
                break

            if changed:
                self.print_params()
                self.params_changed.set()

    def acquisition_thread_func(self):
        """Continuous acquisition with parameter updates."""
        while not self.stop_event.is_set():
            # Create test specs with COMPENSATION parameters
            test_specs = deepcopy(specs_bidirectional)
            test_specs.update(self.params)

            # Setup pattern (returns tuple: ao_data, do_data, scan_layout)
            _, _, scan_layout_orig = setup_and_check_pattern(test_specs)

            # Create scan_layout for FakeAiWrapper with SIMULATION parameters
            scan_layout_sim = deepcopy(scan_layout_orig)
            scan_layout_sim["scanner_lag_samples"] = self.sim_params["scanner_lag_samples"]
            scan_layout_sim["scanner_lag_samples_even"] = self.sim_params["scanner_lag_samples_even"]
            scan_layout_sim["scanner_lag_samples_odd"] = self.sim_params["scanner_lag_samples_odd"]

            # Create fake AI wrapper with SIMULATION parameters
            ai_wrapper = FakeAiWrapper()
            ai_wrapper.initialize(scan_layout_sim)

            # Setup detector thread with COMPENSATION parameters
            local_stop = threading.Event()

            thread = threading.Thread(
                target=detector_thread,
                args=(ai_wrapper, scan_layout_orig, local_stop, self.image_queue, test_specs),
                daemon=True
            )

            # Start acquisition
            ai_wrapper.start()
            thread.start()

            # Acquire frames until parameters change
            while not self.stop_event.is_set() and not self.params_changed.is_set():
                try:
                    frame = self.image_queue.get(timeout=0.5)
                    self.current_frame = frame
                except queue.Empty:
                    continue

            # Stop acquisition
            local_stop.set()
            thread.join(timeout=1.0)
            ai_wrapper.stop()

            # Clear the changed flag and restart with new parameters
            self.params_changed.clear()

            # Small delay before restarting
            time.sleep(0.1)

    def run(self):
        """Run interactive adjustment session."""
        print("\n" + "="*70)
        print(" INTERACTIVE TIMING COMPENSATION ADJUSTMENT")
        print("="*70)
        print("\nThis tool simulates FIXED scanner lag and lets you adjust")
        print("compensation parameters in real-time to correct it.\n")

        print("HOW IT WORKS:")
        print("  1. FakeAiWrapper simulates scanner lag (fixed values)")
        print("  2. You adjust compensation parameters (keyboard controls)")
        print("  3. detector_thread applies your compensation")
        print("  4. Image updates to show result\n")

        print("GOAL: Adjust compensation until image is well-aligned!")
        print("  - Look for smooth vertical profile (no zigzag)")
        print("  - When residual error = 0, compensation is perfect!\n")

        print("Keyboard Controls:")
        print("  Global lag:  q/a (±1), w/s (±5)")
        print("  Even offset: e/d (±1), r/f (±5)")
        print("  Odd offset:  t/g (±1), y/h (±5)")
        print("  Other:       p (print), c (clear), x (exit)")
        print("="*70 + "\n")

        self.print_params()

        # Setup terminal for keyboard input
        self.setup_terminal()

        try:
            # Start keyboard handler
            keyboard_thread = threading.Thread(target=self.handle_keyboard, daemon=True)
            keyboard_thread.start()

            # Start acquisition thread
            acq_thread = threading.Thread(target=self.acquisition_thread_func, daemon=True)
            acq_thread.start()

            # Setup live display
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            def update_display(frame_num):
                """Update display with latest frame."""
                if self.current_frame is None:
                    return

                frame = self.current_frame

                # Clear axes
                for ax in axes:
                    ax.clear()

                # Plot 1: Full image
                axes[0].imshow(frame, cmap='gray', aspect='auto', vmin=0, vmax=1)
                axes[0].set_title("Acquired Image")
                axes[0].set_xlabel("X pixel")
                axes[0].set_ylabel("Y pixel (line)")

                # Overlay even/odd indicators
                for i in range(0, frame.shape[0], 4):
                    axes[0].axhline(i, color='cyan', alpha=0.15, linewidth=0.5)
                for i in range(1, frame.shape[0], 4):
                    axes[0].axhline(i, color='magenta', alpha=0.15, linewidth=0.5)

                # Plot 2: Vertical profile
                center_col = frame.shape[1] // 2
                for i in range(frame.shape[0]):
                    color = 'cyan' if i % 2 == 0 else 'magenta'
                    axes[1].scatter(i, frame[i, center_col], c=color, s=5, alpha=0.6)

                axes[1].set_xlabel("Line index")
                axes[1].set_ylabel("Intensity at center")
                axes[1].set_title("Center Column Profile\n(check for zigzag)")
                axes[1].grid(True, alpha=0.3)

                # Add parameter text
                param_text = (
                    f"Global: {self.params['scanner_lag_samples']}\n"
                    f"Even: {self.params['scanner_lag_samples_even']}\n"
                    f"Odd: {self.params['scanner_lag_samples_odd']}"
                )
                fig.text(0.99, 0.99, param_text, transform=fig.transFigure,
                        ha='right', va='top', fontsize=10, family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                plt.tight_layout()

            # Start animation
            ani = FuncAnimation(fig, update_display, interval=200, cache_frame_data=False)

            plt.show()

        finally:
            # Cleanup
            self.stop_event.set()
            self.restore_terminal()

            print("\n" + "="*70)
            print("Final Parameters:")
            print("="*70)
            self.print_params()
            print("\n✓ Session complete!")


def main():
    """Main entry point."""
    import sys

    # Check for --no-lag argument
    simulate_lag = "--no-lag" not in sys.argv

    if simulate_lag:
        print("\n" + "="*70)
        print("MODE: SIMULATING LAG (default)")
        print("="*70)
        print("FakeAiWrapper will simulate scanner lag.")
        print("Your job: Adjust compensation to remove the zigzag pattern!")
        print("\nUse --no-lag to disable lag simulation for testing.")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("MODE: NO LAG SIMULATION")
        print("="*70)
        print("FakeAiWrapper will NOT simulate lag.")
        print("Image should stay perfect regardless of compensation values.")
        print("="*70)

    try:
        adjuster = InteractiveAdjustment(simulate_lag=simulate_lag)
        adjuster.run()
    except KeyboardInterrupt:
        print("\n\n✓ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
