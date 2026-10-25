"""
Model Quantization for Edge Devices (INT8 and FP16).
Provides ONNX dynamic INT8 quantization, FP16 export, and PyTorch dynamic quantization.
"""
from pathlib import Path
from typing import Optional, Tuple
import os
import torch
from ultralytics import YOLO


class ModelQuantizer:
    """Manages model quantization for resource-constrained robotics hardware."""

    def __init__(self, base_weights_path: str = "yolov8n.pt"):
        self.base_weights_path = base_weights_path
        self.model = YOLO(base_weights_path)

    def export_to_onnx(self, output_path: Optional[str] = None, imgsz: int = 640) -> str:
        """Export PyTorch YOLO model to standard ONNX format."""
        print(f"[*] Exporting {self.base_weights_path} to ONNX (imgsz={imgsz})...")
        exported = self.model.export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False)
        onnx_file = str(exported)

        if output_path and output_path != onnx_file:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.move(onnx_file, output_path)
            return output_path
        return onnx_file

    def quantize_onnx_int8(self, onnx_input_path: str, int8_output_path: str) -> str:
        """
        Quantize float32 ONNX model to INT8 dynamic quantization using ONNX Runtime.
        Produces ~4x model size reduction and high inference speedup on edge CPUs (ARM/x86).
        """
        import onnxruntime
        from onnxruntime.quantization import quantize_dynamic, QuantType

        Path(int8_output_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"[*] Applying dynamic INT8 quantization: {onnx_input_path} -> {int8_output_path}...")

        quantize_dynamic(
            model_input=onnx_input_path,
            model_output=int8_output_path,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False
        )

        orig_size = os.path.getsize(onnx_input_path) / (1024 * 1024)
        quant_size = os.path.getsize(int8_output_path) / (1024 * 1024)
        print(f"[OK] Quantization Complete: {orig_size:.2f} MB -> {quant_size:.2f} MB (Compression: {(1 - quant_size/orig_size)*100:.1f}%)")
        return int8_output_path

    def quantize_pytorch_dynamic(self, output_path: str) -> str:
        """
        Apply PyTorch native dynamic quantization (weights to qint8) to linear layers.
        """
        print(f"[*] Applying PyTorch native dynamic quantization...")
        raw_torch_model = self.model.model
        quantized_torch = torch.ao.quantization.quantize_dynamic(
            raw_torch_model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(quantized_torch.state_dict(), output_path)
        return output_path
