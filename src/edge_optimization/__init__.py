"""
Edge Optimization Package: Quantization, Pruning, Lightweight Backbones, and Benchmarking.
"""
from .quantization import ModelQuantizer
from .pruning import ModelPruner
from .benchmark import EdgeBenchmarkSuite, BenchmarkResult

__all__ = [
    "ModelQuantizer",
    "ModelPruner",
    "EdgeBenchmarkSuite",
    "BenchmarkResult",
]
