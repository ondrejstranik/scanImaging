#!/usr/bin/env python3
"""
Virtual scanner entry point - runs the scanning imaging system with virtual devices.
No real hardware required.
"""

from scanImaging.main import ScanImaging

if __name__ == "__main__":
    ScanImaging.runVirtual()
