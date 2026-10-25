"""
Detection module shim for Problem Statement layout compatibility.
Exports BaseDetector, YOLODetector, Detection.
"""
from ..detector.base_detector import BaseDetector, Detection
from ..detector.yolo_detector import YOLODetector
from ..detector.dataset_adapter import BDD100KAdapter
from ..detector.train_transfer import TransferLearningTrainer

__all__ = [
    "BaseDetector",
    "Detection",
    "YOLODetector",
    "BDD100KAdapter",
    "TransferLearningTrainer",
]
