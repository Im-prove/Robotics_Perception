"""
Transfer Learning and Fine-Tuning Pipeline for YOLO Object Detector.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import torch
from ultralytics import YOLO


class TransferLearningTrainer:
    """Handles transfer learning and fine-tuning on navigation datasets."""

    def __init__(
        self,
        base_model_path: str = "yolov8n.pt",
        data_yaml_path: str = "data/processed_yolo/data.yaml",
        device: str = "cpu"
    ):
        self.base_model_path = base_model_path
        self.data_yaml_path = data_yaml_path
        self.device = device
        self.model = YOLO(base_model_path)

    def fine_tune(
        self,
        epochs: int = 15,
        imgsz: int = 640,
        batch: int = 16,
        lr0: float = 0.01,
        freeze_backbone_layers: int = 10,
        project_dir: str = "runs/train",
        experiment_name: str = "fine_tune_robotics"
    ) -> Dict[str, Any]:
        """
        Fine-tune pretrained weights on target classes.
        Freezes the feature extraction backbone layers to preserve general visual representations
        while training the detection heads for Cones, Barriers, and Stop Signs.
        """
        print(f"[*] Starting transfer learning on {self.base_model_path} for {epochs} epochs...")
        print(f"[*] Freezing first {freeze_backbone_layers} backbone layers for feature preservation.")

        results = self.model.train(
            data=self.data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            freeze=freeze_backbone_layers,
            project=project_dir,
            name=experiment_name,
            device=self.device,
            verbose=True,
            val=True
        )

        best_weights = Path(project_dir) / experiment_name / "weights" / "best.pt"
        return {
            "best_weights": str(best_weights),
            "results": results
        }

    def validate(self, weights_path: Optional[str] = None) -> Dict[str, float]:
        """Validate model and compute mAP metrics."""
        eval_model = YOLO(weights_path) if weights_path else self.model
        metrics = eval_model.val(data=self.data_yaml_path, device=self.device)

        return {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr)
        }
