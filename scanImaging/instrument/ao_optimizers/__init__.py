"""
Adaptive Optics Optimizer Module

This package contains different optimization strategies for AO correction.
Each optimizer implements a specific algorithm (sequential hill climbing, SPGD, etc.)
"""

from .base_optimizer import BaseOptimizer
from .sequential_optimizer import SequentialOptimizer

__all__ = ['BaseOptimizer', 'SequentialOptimizer']
