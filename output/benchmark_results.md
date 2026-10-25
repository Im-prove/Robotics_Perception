# Edge Hardware Benchmarking and Optimization Results

| Model Variant | Backend | Precision | Device | Mean Latency (ms) | P95 Latency (ms) | FPS | Size (MB) | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv8n Baseline (CPU) | PyTorch | FP32 | CPU | 154.42 | 163.00 | **6.5** | 6.25 | **1.00x** |
| YOLO11n Nano Backbone (CPU) | PyTorch | FP32 | CPU | 136.50 | 140.66 | **7.3** | 5.35 | **1.13x** |
| YOLOv8n Pruned (30% Sparsity) | PyTorch | Sparsity 30.0% | CPU | 142.79 | 152.04 | **7.0** | 6.25 | **1.08x** |
| YOLOv8n ONNX Runtime (CPU) | ONNX Runtime | FP32 | CPU | 371.23 | 396.25 | **2.7** | 12.23 | **0.42x** |
| YOLOv8n Edge Quantized INT8 | ONNX Runtime | INT8 Dynamic | CPU | 1284.52 | 1325.26 | **0.8** | 3.33 | **0.12x** |

> Benchmarks measured on local machine with batch size 1 (robotics real-time constraint).
