"""
Adaptive Optics Optimizer Module

This package contains different optimization strategies for AO correction.
Each optimizer implements a specific algorithm (sequential hill climbing, SPGD, etc.)
"""

from .base_optimizer import BaseOptimizer
from .sequential_optimizer import SequentialOptimizer
from .spgd_optimizer import SPGDOptimizer
from .random_search_optimizer import RandomSearchOptimizer
from .fitting import compute_optimal_simple_interpolation, compute_optimal_weighted_fit
from .metrics import get_metric_function, METRIC_FUNCTIONS

__all__ = [
    'BaseOptimizer', 'SequentialOptimizer', 'SPGDOptimizer', 'RandomSearchOptimizer',
    'compute_optimal_simple_interpolation', 'compute_optimal_weighted_fit',
    'get_metric_function', 'METRIC_FUNCTIONS',
]
