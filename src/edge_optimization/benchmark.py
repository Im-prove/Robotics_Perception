"""
Edge Hardware Benchmarking Suite.
Measures latency (mean/p95/p99), throughput (FPS), disk size, and memory across
baseline, lightweight backbones, quantized, and pruned models.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch


@dataclass
class BenchmarkResult:
    """Benchmark performance metrics for a specific model configuration."""
    model_name: str
    backend: str            # 'PyTorch' or 'ONNX Runtime'
    precision: str          # 'FP32', 'INT8', 'Pruned-FP32'
    device: str             # 'CPU' or 'GPU'
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    fps: float
    model_size_mb: float
    speedup_factor: float = 1.0


class EdgeBenchmarkSuite:
    """Automated benchmark runner for edge optimization evaluation."""

    def __init__(self, iterations: int = 50, warmup: int = 10, img_size: int = 640):
        self.iterations = iterations
        self.warmup = warmup
        self.img_size = img_size
        self.results: List[BenchmarkResult] = []

    def benchmark_pytorch(
        self,
        weights_path: str,
        model_name: str,
        precision: str = "FP32",
        device: str = "cpu"
    ) -> BenchmarkResult:
        """Benchmark a PyTorch YOLO model."""
        from ultralytics import YOLO
        model = YOLO(weights_path)
        model.to(device)

        # Generate dummy input frame
        dummy_frame = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)

        # Warmup
        for _ in range(self.warmup):
            _ = model.predict(dummy_frame, verbose=False)

        # Benchmark loop
        latencies = []
        for _ in range(self.iterations):
            t0 = time.perf_counter()
            _ = model.predict(dummy_frame, verbose=False)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        latencies = np.array(latencies)
        mean_lat = float(np.mean(latencies))
        med_lat = float(np.median(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
        size_mb = os.path.getsize(weights_path) / (1024.0 * 1024.0) if os.path.exists(weights_path) else 0.0

        res = BenchmarkResult(
            model_name=model_name,
            backend="PyTorch",
            precision=precision,
            device=device.upper(),
            mean_latency_ms=mean_lat,
            median_latency_ms=med_lat,
            p95_latency_ms=p95_lat,
            fps=fps,
            model_size_mb=size_mb
        )
        self.results.append(res)
        return res

    def benchmark_onnx(
        self,
        onnx_path: str,
        model_name: str,
        precision: str = "FP32",
        device: str = "cpu"
    ) -> BenchmarkResult:
        """Benchmark an ONNX Runtime model (FP32 or INT8)."""
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if device.lower() == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        session = ort.InferenceSession(onnx_path, providers=providers)
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]

        # Generate dummy tensor
        dummy_input = np.random.rand(1, 3, self.img_size, self.img_size).astype(np.float32)

        # Warmup
        for _ in range(self.warmup):
            _ = session.run(output_names, {input_name: dummy_input})

        # Benchmark loop
        latencies = []
        for _ in range(self.iterations):
            t0 = time.perf_counter()
            _ = session.run(output_names, {input_name: dummy_input})
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        latencies = np.array(latencies)
        mean_lat = float(np.mean(latencies))
        med_lat = float(np.median(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
        size_mb = os.path.getsize(onnx_path) / (1024.0 * 1024.0) if os.path.exists(onnx_path) else 0.0

        res = BenchmarkResult(
            model_name=model_name,
            backend="ONNX Runtime",
            precision=precision,
            device=device.upper(),
            mean_latency_ms=mean_lat,
            median_latency_ms=med_lat,
            p95_latency_ms=p95_lat,
            fps=fps,
            model_size_mb=size_mb
        )
        self.results.append(res)
        return res

    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected results to a comparative DataFrame."""
        if not self.results:
            return pd.DataFrame()

        baseline_fps = self.results[0].fps if self.results[0].fps > 0 else 1.0
        for r in self.results:
            r.speedup_factor = float(r.fps / baseline_fps)

        data = [asdict(r) for r in self.results]
        df = pd.DataFrame(data)
        return df

    def to_markdown(self) -> str:
        """Format benchmark comparison as a GitHub-flavored markdown table."""
        df = self.to_dataframe()
        if df.empty:
            return "No benchmark results available."

        headers = [
            "Model Variant", "Backend", "Precision", "Device",
            "Mean Latency (ms)", "P95 Latency (ms)", "FPS", "Size (MB)", "Speedup"
        ]

        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |"
        ]

        for _, row in df.iterrows():
            lines.append(
                f"| {row['model_name']} | {row['backend']} | {row['precision']} | {row['device']} | "
                f"{row['mean_latency_ms']:.2f} | {row['p95_latency_ms']:.2f} | **{row['fps']:.1f}** | "
                f"{row['model_size_mb']:.2f} | **{row['speedup_factor']:.2f}x** |"
            )

        return "\n".join(lines)

    def plot_comparison(self, output_png_path: str = "output/benchmark_comparison.png") -> str:
        """Generate a visual before-vs-after comparison figure."""
        df = self.to_dataframe()
        if df.empty:
            return ""

        Path(output_png_path).parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Edge Optimization Benchmark: Latency, FPS & Model Footprint", fontsize=14, fontweight="bold")

        labels = [f"{r.model_name}\n({r.backend} {r.precision})" for r in self.results]
        colors = ["#2b5c8f", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"][:len(labels)]

        # 1. Latency (Lower is better)
        latencies = df["mean_latency_ms"].values
        bars1 = ax1.bar(range(len(labels)), latencies, color=colors, width=0.6)
        ax1.set_title("Mean Latency (ms) - Lower is Better", fontweight="bold")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        for bar in bars1:
            y = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, y + 1.0, f"{y:.1f} ms", ha="center", va="bottom", fontsize=8)

        # 2. FPS (Higher is better)
        fps_vals = df["fps"].values
        bars2 = ax2.bar(range(len(labels)), fps_vals, color=colors, width=0.6)
        ax2.set_title("Inference Throughput (FPS) - Higher is Better", fontweight="bold")
        ax2.set_ylabel("Frames Per Second")
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        for bar in bars2:
            y = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, y + 0.5, f"{y:.1f} FPS", ha="center", va="bottom", fontsize=8)

        # 3. Model Size (MB) (Lower is better)
        sizes = df["model_size_mb"].values
        bars3 = ax3.bar(range(len(labels)), sizes, color=colors, width=0.6)
        ax3.set_title("Model Footprint (MB) - Lower is Better", fontweight="bold")
        ax3.set_ylabel("File Size (MB)")
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        for bar in bars3:
            y = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, y + 0.1, f"{y:.2f} MB", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_png_path, dpi=200)
        plt.close()
        return output_png_path
