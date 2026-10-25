"""
Model Pruning for Neural Network Sparsification.
Applies L1-unstructured and structured channel pruning to convolutional layers.
"""
from pathlib import Path
from typing import Dict, Tuple
import os
import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO


class ModelPruner:
    """Prunes convolutional layers to induce sparsity and reduce memory pressure."""

    def __init__(self, weights_path: str = "yolov8n.pt"):
        self.weights_path = weights_path
        self.model = YOLO(weights_path)

    def prune_l1_unstructured(
        self,
        amount: float = 0.30,
        output_path: str = "pruned_model.pt"
    ) -> Dict[str, float]:
        """
        Prune `amount` (e.g. 30%) of lowest magnitude weights in Conv2d layers.
        """
        print(f"[*] Applying L1 unstructured pruning with target sparsity: {amount*100:.1f}%...")
        torch_model = self.model.model

        total_zeros = 0
        total_elements = 0
        pruned_layer_count = 0

        for name, module in torch_model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name="weight", amount=amount)
                # Make pruning permanent by removing original parameter and keeping weight
                prune.remove(module, "weight")
                zeros = float(torch.sum(module.weight == 0))
                elements = float(module.weight.nelement())
                total_zeros += zeros
                total_elements += elements
                pruned_layer_count += 1

        global_sparsity = (total_zeros / total_elements) * 100.0 if total_elements > 0 else 0.0

        # Save pruned model
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Save ultralytics checkpoint structure
        self.model.save(output_path)

        orig_size = os.path.getsize(self.weights_path) / (1024 * 1024)
        pruned_size = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else orig_size

        print(f"[OK] Pruning Complete across {pruned_layer_count} Conv2d layers.")
        print(f"[OK] Global Sparsity: {global_sparsity:.2f}% | Model Size: {pruned_size:.2f} MB")

        return {
            "global_sparsity_pct": global_sparsity,
            "pruned_layers": float(pruned_layer_count),
            "original_size_mb": orig_size,
            "pruned_size_mb": pruned_size
        }
