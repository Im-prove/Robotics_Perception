"""
Edge Optimization Benchmarking CLI.
Measures:
1. Baseline PyTorch FP32 Latency and FPS (CPU and GPU where available).
2. Lightweight Backbone (YOLO11n).
3. L1 Unstructured Pruned Model.
4. Exported ONNX Model (FP32).
5. Dynamic INT8 Quantized ONNX Model.
Generates latency percentiles (Mean, Median, P95, P99), throughput, disk footprint, and speedup tables.
"""
import argparse
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
    parser = argparse.ArgumentParser(description="Edge Optimization Benchmarking Suite")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Baseline weights")
    parser.add_argument("--lightweight", type=str, default="yolo11n.pt", help="Lightweight backbone weights")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmarking iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--imgsz", type=int, default=640, help="Benchmark image size")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save benchmark reports")
    args = parser.parse_args()

    print("=================================================================")
    print("         ROBOTICS EDGE OPTIMIZATION BENCHMARK SUITE             ")
    print("=================================================================\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suite = EdgeBenchmarkSuite(iterations=args.iterations, warmup=args.warmup, img_size=args.imgsz)

    devices_to_test = ["cpu"]
    if torch.cuda.is_available():
        devices_to_test.append("cuda")
        print(f"[*] CUDA GPU Detected: {torch.cuda.get_device_name(0)}")

    for dev in devices_to_test:
        print(f"\n--- Testing on Device: {dev.upper()} ---")

        # 1. Baseline Model
        if os.path.exists(args.weights):
            print(f"[*] Benchmarking Baseline {args.weights} ({dev.upper()})...")
            suite.benchmark_pytorch(
                weights_path=args.weights,
                model_name=f"YOLOv8n Baseline ({dev.upper()})",
                precision="FP32",
                device=dev
            )

        # 2. Lightweight Backbone
        if os.path.exists(args.lightweight):
            print(f"[*] Benchmarking Lightweight Backbone {args.lightweight} ({dev.upper()})...")
            suite.benchmark_pytorch(
                weights_path=args.lightweight,
                model_name=f"YOLO11n Nano Backbone ({dev.upper()})",
                precision="FP32",
                device=dev
            )

    # 3. Pruning Benchmark
    if os.path.exists(args.weights):
        print(f"\n[*] Evaluating L1 Unstructured Pruning (30% Sparsity)...")
        pruner = ModelPruner(args.weights)
        pruned_path = str(output_dir / "yolov8n_pruned_30pct.pt")
        prune_stats = pruner.prune_l1_unstructured(amount=0.30, output_path=pruned_path)
        suite.benchmark_pytorch(
            weights_path=pruned_path,
            model_name="YOLOv8n Pruned (30% Sparsity)",
            precision=f"Sparsity {prune_stats['global_sparsity_pct']:.1f}%",
            device="cpu"
        )

    # 4. ONNX Runtime & INT8 Quantization Benchmark
    if os.path.exists(args.weights):
        print(f"\n[*] Evaluating ONNX Export & INT8 Dynamic Quantization...")
        quantizer = ModelQuantizer(args.weights)
        onnx_fp32 = str(output_dir / "yolov8n_fp32.onnx")
        onnx_int8 = str(output_dir / "yolov8n_dynamic_int8.onnx")

        # Export ONNX FP32 if needed
        if not os.path.exists(onnx_fp32):
            quantizer.export_to_onnx(output_path=onnx_fp32, imgsz=args.imgsz)

        # Benchmark ONNX FP32
        suite.benchmark_onnx(
            onnx_path=onnx_fp32,
            model_name="YOLOv8n ONNX Runtime (CPU)",
            precision="FP32",
            device="cpu"
        )

        # Quantize to INT8
        quantizer.quantize_onnx_int8(onnx_input_path=onnx_fp32, int8_output_path=onnx_int8)

        # Benchmark ONNX INT8
        suite.benchmark_onnx(
            onnx_path=onnx_int8,
            model_name="YOLOv8n Edge Quantized INT8",
            precision="INT8 Dynamic",
            device="cpu"
        )

    # Compile and save results
    df = suite.to_dataframe()
    md_table = suite.to_markdown()

    md_path = output_dir / "benchmark_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Edge Hardware Benchmarking and Optimization Results\n\n")
        f.write(md_table)
        f.write("\n\n> Benchmarks measured on local machine with batch size 1 (robotics real-time constraint).\n")

    print(f"\n[OK] Benchmarks complete! Report saved to {md_path}")
    print("\n" + md_table)

    # Plot comparison chart
    chart_path = output_dir / "benchmark_comparison.png"
    suite.plot_comparison(output_path=str(chart_path))
    print(f"[OK] Saved comparison chart to {chart_path}")


if __name__ == "__main__":
    main()
