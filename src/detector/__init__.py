"""
Detector Package for Robotic Object Detection.
"""
from .base_detector import BaseDetector, Detection
from .yolo_detector import YOLODetector
from .dataset_adapter import BDD100KAdapter
from .train_transfer import TransferLearningTrainer

__all__ = [
    "BaseDetector",
    "Detection",
    "YOLODetector",
    "BDD100KAdapter",
    "TransferLearningTrainer",
]
