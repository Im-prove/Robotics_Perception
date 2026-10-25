"""
Unit Tests for Edge Optimization (Pruning, Quantization, Benchmarking).
"""
import pytest
import os
import torch

from src.edge_optimization.pruning import ModelPruner
from src.edge_optimization.quantization import ModelQuantizer
from src.edge_optimization.benchmark import EdgeBenchmarkSuite, BenchmarkResult


def test_pruning_execution():
    """Verify L1 unstructured pruning introduces sparsity in Conv layers."""
    weights_path = "yolov8n.pt"
    if not os.path.exists(weights_path):
        pytest.skip("yolov8n.pt not present")

    pruner = ModelPruner(weights_path)
    out_path = "output/test_pruned.pt"
    stats = pruner.prune_l1_unstructured(amount=0.20, output_path=out_path)

    assert stats["global_sparsity_pct"] > 10.0
    assert stats["pruned_layers"] > 0
    assert os.path.exists(out_path)


def test_benchmark_suite_metrics():
    suite = EdgeBenchmarkSuite(iterations=2, warmup=1)
    res = BenchmarkResult(
        model_name="Test Model",
        backend="PyTorch",
        precision="FP32",
        device="CPU",
        mean_latency_ms=25.0,
        median_latency_ms=24.5,
        p95_latency_ms=28.0,
        fps=40.0,
        model_size_mb=6.2
    )
    suite.results.append(res)
    df = suite.to_dataframe()
    assert not df.empty
    assert "mean_latency_ms" in df.columns
    assert "fps" in df.columns
    md = suite.to_markdown()
    assert "Test Model" in md
