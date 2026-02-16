#!/usr/bin/env python3
"""
Real scanner entry point - runs the scanning imaging system with real hardware devices.
Requires actual hardware to be connected.
"""

from scanImaging.main import ScanImaging

if __name__ == "__main__":
    ScanImaging.runReal()
