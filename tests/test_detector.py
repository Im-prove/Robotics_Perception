"""
Unit Tests for Detector Modules and Label Formatting.
"""
import pytest
import numpy as np

from src.detector.base_detector import Detection
from src.utils.visualization import Visualizer


def test_detection_label_format():
    """Verify annotation text matches exact prompt requirements: 'Cone, 1.5m' and 'Stop Sign, 3.2m'."""
    det_cone = Detection(
        class_id=0,
        class_name="Cone",
        confidence=0.92,
        bbox_xyxy=(100, 150, 150, 250),
        distance_m=1.5
    )
    assert det_cone.label_text == "Cone, 1.5m"

    det_stop = Detection(
        class_id=1,
        class_name="Stop Sign",
        confidence=0.88,
        bbox_xyxy=(400, 200, 500, 300),
        distance_m=3.2
    )
    assert det_stop.label_text == "Stop Sign, 3.2m"

    det_barrier = Detection(
        class_id=2,
        class_name="Barrier",
        confidence=0.85,
        bbox_xyxy=(200, 200, 450, 320),
        distance_m=2.8
    )
    assert det_barrier.label_text == "Barrier, 2.8m"


def test_visualizer_drawing():
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = [
        Detection(
            class_id=0,
            class_name="Cone",
            confidence=0.95,
            bbox_xyxy=(500, 300, 600, 500),
            distance_m=1.5
        )
    ]
    annotated = Visualizer.draw_detections(canvas, detections)
    assert annotated.shape == (720, 1280, 3)
    # Check that pixels were drawn (not pure black)
    assert np.any(annotated > 0)
