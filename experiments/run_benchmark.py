"""
Edge Optimization Benchmarking Suite Runner.
Evaluates and compares:
1. Baseline YOLOv8n (PyTorch FP32)
2. Lightweight Backbone YOLO11n (PyTorch FP32)
3. Pruned YOLOv8n (PyTorch 30% Sparsity)
4. ONNX Runtime Exported Model (FP32)
5. Edge Quantized Model (ONNX Dynamic INT8)

Outputs latency (ms), FPS, model footprint (MB), and speedup factors.
"""
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.edge_optimization.quantization import ModelQuantizer
from src.edge_optimization.pruning import ModelPruner
from src.edge_optimization.benchmark import EdgeBenchmarkSuite


def main():
    print("=================================================================")
    print("         ROBOTICS EDGE OPTIMIZATION BENCHMARK SUITE             ")
    print("=================================================================\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Evaluation Device: {device.upper()}")
    if device == "cuda":
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    benchmark_suite = EdgeBenchmarkSuite(iterations=40, warmup=10, img_size=640)
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Baseline Model: YOLOv8n (PyTorch FP32)
    print("\n[Step 1/5] Benchmarking Baseline YOLOv8n (PyTorch FP32)...")
    benchmark_suite.benchmark_pytorch(
        weights_path="yolov8n.pt",
        model_name="YOLOv8n Baseline",
        precision="FP32",
        device=device
    )

    # 2. Lightweight Backbone: YOLO11n (PyTorch FP32)
    print("\n[Step 2/5] Benchmarking Lightweight Backbone YOLO11n (PyTorch FP32)...")
    if os.path.exists("yolo11n.pt"):
        benchmark_suite.benchmark_pytorch(
            weights_path="yolo11n.pt",
            model_name="YOLO11n Lightweight Backbone",
            precision="FP32",
            device=device
        )
    else:
        print("  [!] yolo11n.pt not found, skipping backbone comparison.")

    # 3. Pruned Model: YOLOv8n with 30% L1 Sparsity
    print("\n[Step 3/5] Applying and Benchmarking L1 Pruning (30% Sparsity)...")
    pruner = ModelPruner("yolov8n.pt")
    pruned_weights_path = "output/yolov8n_pruned_30pct.pt"
    prune_stats = pruner.prune_l1_unstructured(amount=0.30, output_path=pruned_weights_path)

    benchmark_suite.benchmark_pytorch(
        weights_path=pruned_weights_path,
        model_name="YOLOv8n Pruned (30%)",
        precision=f"Sparsity {prune_stats['global_sparsity_pct']:.1f}%",
        device=device
    )

    # 4. Standard ONNX Model (FP32)
    print("\n[Step 4/5] Exporting and Benchmarking Standard ONNX Model...")
    quantizer = ModelQuantizer("yolov8n.pt")
    onnx_fp32_path = "output/yolov8n_fp32.onnx"
    quantizer.export_to_onnx(output_path=onnx_fp32_path, imgsz=640)

    benchmark_suite.benchmark_onnx(
        onnx_path=onnx_fp32_path,
        model_name="YOLOv8n ONNX Runtime",
        precision="FP32",
        device=device
    )

    # 5. Edge Quantized Model: Dynamic INT8 ONNX
    print("\n[Step 5/5] Generating and Benchmarking Dynamic INT8 Quantized Model...")
    onnx_int8_path = "output/yolov8n_dynamic_int8.onnx"
    quantizer.quantize_onnx_int8(onnx_input_path=onnx_fp32_path, int8_output_path=onnx_int8_path)

    benchmark_suite.benchmark_onnx(
        onnx_path=onnx_int8_path,
        model_name="YOLOv8n Edge Quantized",
        precision="INT8 Dynamic",
        device=device
    )

    # Compile and display results
    print("\n=================================================================")
    print("                    BENCHMARK RESULTS TABLE                     ")
    print("=================================================================\n")
    md_table = benchmark_suite.to_markdown()
    print(md_table)

    # Save Markdown report
    report_path = output_dir / "benchmark_results.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Edge Optimization Benchmark Results\n\n")
        f.write(f"Hardware Device: **{device.upper()}**\n\n")
        f.write(md_table + "\n\n")
        f.write("![Benchmark Comparison](benchmark_comparison.png)\n")
    print(f"\n[OK] Saved markdown report to: {report_path}")

    # Generate comparison plot
    plot_path = benchmark_suite.plot_comparison(str(output_dir / "benchmark_comparison.png"))
    print(f"[OK] Generated visualization chart at: {plot_path}")


if __name__ == "__main__":
    main()
