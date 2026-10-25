# Edge Optimization Benchmark Results

Hardware Device: **CPU**

| Model Variant | Backend | Precision | Device | Mean Latency (ms) | P95 Latency (ms) | FPS | Size (MB) | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLOv8n Baseline | PyTorch | FP32 | CPU | 41.19 | 49.75 | **24.3** | 6.25 | **1.00x** |
| YOLO11n Lightweight Backbone | PyTorch | FP32 | CPU | 41.95 | 50.87 | **23.8** | 5.35 | **0.98x** |
| YOLOv8n Pruned (30%) | PyTorch | Sparsity 30.0% | CPU | 124.17 | 197.24 | **8.1** | 6.25 | **0.33x** |
| YOLOv8n ONNX Runtime | ONNX Runtime | FP32 | CPU | 69.18 | 93.21 | **14.5** | 12.23 | **0.60x** |
| YOLOv8n Edge Quantized | ONNX Runtime | INT8 Dynamic | CPU | 591.81 | 1113.84 | **1.7** | 3.33 | **0.07x** |

![Benchmark Comparison](benchmark_comparison.png)
