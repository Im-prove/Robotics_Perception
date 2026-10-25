"""
Abstract Base Object Detector Interface for Robotics Perception.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
import numpy as np


@dataclass
class Detection:
    """Represents a single detected obstacle."""
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    distance_m: Optional[float] = None            # Estimated Euclidean distance
    distance_z_m: Optional[float] = None          # Estimated longitudinal depth
    offset_x_m: Optional[float] = None            # Lateral offset
    track_id: Optional[int] = None

    @property
    def label_text(self) -> str:
        """Format as required by the assignment: e.g., 'Cone, 1.5m' or 'Stop Sign, 3.2m'."""
        if self.distance_m is not None:
            return f"{self.class_name}, {self.distance_m:.1f}m"
        return f"{self.class_name} {self.confidence:.2f}"

    @property
    def width(self) -> float:
        return float(self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height(self) -> float:
        return float(self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def center(self) -> Tuple[float, float]:
        return (
            float((self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2.0),
            float((self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2.0)
        )


class BaseDetector(ABC):
    """Abstract base class for object detectors."""

    @abstractmethod
    def predict(self, image: np.ndarray, conf_thresh: float = 0.35) -> List[Detection]:
        """Run object detection on an RGB or BGR image."""
        pass

    @abstractmethod
    def get_model_size_mb(self) -> float:
        """Return model disk/memory footprint in Megabytes."""
        pass
