#!/usr/bin/env python3
"""
Test script for galvo scanner timing compensation.

This script tests the timing compensation implementation using FakeAiWrapper
to simulate scanner lag effects. It runs multiple test scenarios to verify
that the compensation logic correctly handles various lag configurations.

Test Strategy:
--------------
Each test specifies TWO sets of parameters:
1. sim_params: Lag values used by FakeAiWrapper to SIMULATE scanner lag
2. comp_params: Lag values used by detector_thread to COMPENSATE for lag

By varying these independently, we can test:
- Baseline: No lag, no compensation → Should be well aligned
- Compensation works: Same lag and compensation → Should be well aligned
- No compensation: Lag without compensation → Should show misalignment
- Partial compensation: Wrong compensation values → Should show some misalignment

This validates both the simulation and the compensation logic.

Usage:
    python test_timing_compensation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from galvo_scan_read_test import (
    FakeAiWrapper, setup_and_check_pattern, detector_thread,
    specs_bidirectional
)
import threading
import queue
import time
from copy import deepcopy


def check_line_alignment(frame, title="Line Alignment", output_path=None):
    """
    Visualize line alignment for diagnostic purposes.

    Creates two plots:
    1. Full image with even/odd line overlay
    2. Vertical edge profile to detect zigzag patterns
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Full image with even/odd overlay
    ax1 = axes[0]
    ax1.imshow(frame, cmap='gray', aspect='auto', vmin=0, vmax=1)
    for i in range(0, frame.shape[0], 2):
        ax1.axhline(i, color='cyan', alpha=0.2, linewidth=0.5)
    for i in range(1, frame.shape[0], 2):
        ax1.axhline(i, color='magenta', alpha=0.2, linewidth=0.5)
    ax1.set_title(f"{title}\nEven (cyan) vs Odd (magenta)")
    ax1.set_xlabel("X pixel")
    ax1.set_ylabel("Y pixel (line)")

    # Plot 2: Horizontal profile at center
    ax2 = axes[1]
    center_row = frame.shape[0] // 2
    ax2.plot(frame[center_row, :], 'k-', linewidth=1, label='Center line')
    ax2.set_xlabel("X pixel")
    ax2.set_ylabel("Intensity")
    ax2.set_title("Center horizontal profile")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Vertical profile showing even vs odd lines
    ax3 = axes[2]
    center_col = frame.shape[1] // 2
    for i in range(frame.shape[0]):
        color = 'cyan' if i % 2 == 0 else 'magenta'
        ax3.scatter(i, frame[i, center_col], c=color, s=10, alpha=0.6)
    ax3.set_xlabel("Line index")
    ax3.set_ylabel("Intensity at center column")
    ax3.set_title("Vertical profile (check for zigzag)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  → Diagnostic plot saved to {output_path}")

    return fig


def calculate_alignment_metric(frame):
    """
    Calculate alignment quality metric.

    Computes cross-correlation between even and odd lines at center column.
    Higher values indicate better alignment (less zigzag).

    Returns:
        float: Alignment metric (higher is better, 1.0 is perfect)
    """
    center_col = frame.shape[1] // 2

    # Extract even and odd line profiles
    even_lines = frame[::2, center_col]
    odd_lines = frame[1::2, center_col]

    # Make same length
    min_len = min(len(even_lines), len(odd_lines))
    even_lines = even_lines[:min_len]
    odd_lines = odd_lines[:min_len]

    # Compute correlation
    if np.std(even_lines) < 1e-6 or np.std(odd_lines) < 1e-6:
        return 0.0

    correlation = np.corrcoef(even_lines, odd_lines)[0, 1]
    return correlation


def run_single_test(test_name, sim_params, comp_params, expected_alignment="good"):
    """
    Run a single timing compensation test.

    Args:
        test_name: Descriptive name for this test
        sim_params: Dict with simulation lag parameters (used by FakeAiWrapper)
        comp_params: Dict with compensation parameters (used by detector_thread)
        expected_alignment: "good" or "bad" - expected result

    Returns:
        tuple: (success, alignment_metric, frame)
    """
    print(f"\n{'='*70}")
    print(f"Test: {test_name}")
    print(f"{'='*70}")
    print(f"Simulation (FakeAi lag): {sim_params}")
    print(f"Compensation (detector): {comp_params}")

    # Create test specs with compensation parameters
    test_specs = deepcopy(specs_bidirectional)
    test_specs.update(comp_params)

    # Setup pattern (returns tuple: ao_data, do_data, scan_layout)
    _, _, scan_layout_orig = setup_and_check_pattern(test_specs)

    # Override scan_layout with simulation parameters for FakeAiWrapper
    # This allows us to simulate lag independently from compensation
    scan_layout = deepcopy(scan_layout_orig)
    scan_layout["scanner_lag_samples"] = sim_params.get("scanner_lag_samples", 0)
    scan_layout["scanner_lag_samples_even"] = sim_params.get("scanner_lag_samples_even", 0)
    scan_layout["scanner_lag_samples_odd"] = sim_params.get("scanner_lag_samples_odd", 0)

    # Create fake AI wrapper with SIMULATION parameters
    ai_wrapper = FakeAiWrapper()
    ai_wrapper.initialize(scan_layout)  # Uses sim_params for lag simulation

    # Setup detector thread with COMPENSATION parameters
    stop_event = threading.Event()
    image_queue = queue.Queue(maxsize=2)

    thread = threading.Thread(
        target=detector_thread,
        args=(ai_wrapper, scan_layout_orig, stop_event, image_queue, test_specs),  # Uses comp_params
        daemon=True
    )

    # Start acquisition
    ai_wrapper.start()
    thread.start()

    # Wait for first frame
    try:
        frame = image_queue.get(timeout=5.0)
        print(f"  ✓ Frame acquired: {frame.shape}")
    except queue.Empty:
        print(f"  ✗ FAILED: No frame received")
        stop_event.set()
        return False, 0.0, None

    # Stop acquisition
    stop_event.set()
    thread.join(timeout=2.0)
    ai_wrapper.stop()

    # Calculate alignment metric
    alignment_metric = calculate_alignment_metric(frame)
    print(f"  Alignment metric: {alignment_metric:.4f}")

    # Check result
    if expected_alignment == "good":
        success = alignment_metric > 0.95
        status = "✓ PASS" if success else "✗ FAIL"
    else:  # expected_alignment == "bad"
        success = alignment_metric < 0.90
        status = "✓ PASS" if success else "✗ FAIL"

    print(f"  Expected: {expected_alignment} alignment")
    print(f"  Result: {status}")

    return success, alignment_metric, frame


def run_test_suite():
    """Run complete test suite for timing compensation."""

    print("\n" + "="*70)
    print(" GALVO SCANNER TIMING COMPENSATION TEST SUITE")
    print("="*70)

    results = []
    frames = {}

    # Test 1: Baseline (no lag, no compensation)
    test_name = "Test 1: Baseline (no lag, no compensation)"
    sim_params = {"scanner_lag_samples": 0, "scanner_lag_samples_even": 0, "scanner_lag_samples_odd": 0}
    comp_params = {"scanner_lag_samples": 0, "scanner_lag_samples_even": 0, "scanner_lag_samples_odd": 0}
    success, metric, frame = run_single_test(test_name, sim_params, comp_params, expected_alignment="good")
    results.append((test_name, success, metric))
    frames[test_name] = frame

    # Test 2: Global lag without compensation (should show misalignment)
    test_name = "Test 2: Global lag WITH compensation"
    sim_params = {"scanner_lag_samples": 15, "scanner_lag_samples_even": 0, "scanner_lag_samples_odd": 0}
    comp_params = {"scanner_lag_samples": 15, "scanner_lag_samples_even": 0, "scanner_lag_samples_odd": 0}
    success, metric, frame = run_single_test(test_name, sim_params, comp_params, expected_alignment="good")
    results.append((test_name, success, metric))
    frames[test_name] = frame

    # Test 3: Global lag WITHOUT compensation (should show misalignment)
    test_name = "Test 3: Global lag WITHOUT compensation"
    sim_params = {"scanner_lag_samples": 15, "scanner_lag_samples_even": 0, "scanner_lag_samples_odd": 0}
    comp_params = {"scanner_lag_samples": 0, "scanner_lag_samples_even": 0, "scanner_lag_samples_odd": 0}
    success, metric, frame = run_single_test(test_name, sim_params, comp_params, expected_alignment="bad")
    results.append((test_name, success, metric))
    frames[test_name] = frame

    # Test 4: Asymmetric lag WITH compensation (should be good)
    test_name = "Test 4: Asymmetric lag WITH compensation"
    sim_params = {"scanner_lag_samples": 10, "scanner_lag_samples_even": 15, "scanner_lag_samples_odd": -5}
    comp_params = {"scanner_lag_samples": 10, "scanner_lag_samples_even": 15, "scanner_lag_samples_odd": -5}
    success, metric, frame = run_single_test(test_name, sim_params, comp_params, expected_alignment="good")
    results.append((test_name, success, metric))
    frames[test_name] = frame

    # Test 5: Asymmetric lag WITHOUT compensation (should show severe zigzag)
    test_name = "Test 5: Asymmetric lag WITHOUT compensation"
    sim_params = {"scanner_lag_samples": 10, "scanner_lag_samples_even": 15, "scanner_lag_samples_odd": -5}
    comp_params = {"scanner_lag_samples": 0, "scanner_lag_samples_even": 0, "scanner_lag_samples_odd": 0}
    success, metric, frame = run_single_test(test_name, sim_params, comp_params, expected_alignment="bad")
    results.append((test_name, success, metric))
    frames[test_name] = frame

    # Test 6: Partial compensation (wrong values - should still show some misalignment)
    test_name = "Test 6: Partial compensation (incorrect values)"
    sim_params = {"scanner_lag_samples": 15, "scanner_lag_samples_even": 0, "scanner_lag_samples_odd": 0}
    comp_params = {"scanner_lag_samples": 8, "scanner_lag_samples_even": 0, "scanner_lag_samples_odd": 0}  # Wrong value!
    success, metric, frame = run_single_test(test_name, sim_params, comp_params, expected_alignment="bad")
    results.append((test_name, success, metric))
    frames[test_name] = frame

    # Print summary
    print(f"\n{'='*70}")
    print(" TEST SUMMARY")
    print(f"{'='*70}")

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for test_name, success, metric in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} {test_name:50s} (metric: {metric:.4f})")

    print(f"\n{passed}/{total} tests passed")

    # Generate diagnostic plots
    print(f"\nGenerating diagnostic plots...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (test_name, frame) in enumerate(frames.items()):
        if idx >= len(axes):
            break
        if frame is not None:
            axes[idx].imshow(frame, cmap='gray', aspect='auto', vmin=0, vmax=1)
            # Color border based on expected result
            if "WITHOUT compensation" in test_name or "incorrect" in test_name:
                color = 'red'  # Expected bad
            else:
                color = 'green'  # Expected good
            for spine in axes[idx].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
            axes[idx].set_title(test_name.replace("Test ", ""), fontsize=9)
            axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig("test_timing_compensation_summary.png", dpi=150, bbox_inches='tight')
    print(f"  → Summary plot saved to test_timing_compensation_summary.png")

    # Generate detailed alignment plots for key tests
    for test_name, frame in frames.items():
        if frame is not None and ("Asymmetric" in test_name or "WITHOUT" in test_name):
            filename = test_name.replace(" ", "_").replace(":", "").lower() + ".png"
            check_line_alignment(frame, title=test_name, output_path=filename)

    return passed == total


if __name__ == "__main__":
    import sys

    print("\nTiming Compensation Test Suite")
    print("This script tests the scanner lag compensation using synthetic data.\n")

    # Run test suite
    success = run_test_suite()

    print("\n" + "="*70)
    if success:
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nThe timing compensation implementation is working correctly!")
        print("You can now calibrate on real hardware using the values from the plan.")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        print("="*70)
        print("\nPlease review the diagnostic plots and check the implementation.")
        sys.exit(1)
