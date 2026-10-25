"""
Unit and Integration Tests for Perception Pipeline and Problem Statement Shims.
"""
import pytest
import numpy as np

from src.inference.pipeline import RoboticsPerceptionPipeline
from src.detection.detector import YOLODetector, Detection
from src.geometry.distance import (
    estimate_distance_from_size,
    estimate_ground_distance,
    KalmanDistanceFusion
)
from src.geometry.camera import CameraIntrinsics, CameraExtrinsics
from src.visualization.annotate import Visualizer


def test_problem_statement_standalone_formulas():
    """Verify Section 4 standalone geometric equations."""
    # Method A: Z = fy * H / h
    # fy = 800px, H = 0.70m, h = 280px => Z = 800 * 0.70 / 280 = 2.0m
    z = estimate_distance_from_size(focal_y_px=800.0, object_height_m=0.70, bbox_height_px=280.0)
    assert z is not None
    assert abs(z - 2.0) < 1e-4

    # Method B: D = H_c / tan(theta + alpha)
    # H_c = 0.65m, pitch = 2.5 deg, alpha = atan((v - cy) / fy)
    d = estimate_ground_distance(
        bottom_y_px=550.0,
        camera_height_m=0.65,
        focal_y_px=800.0,
        principal_y_px=360.0,
        pitch_down_deg=2.5
    )
    assert d is not None
    assert 1.0 < d < 10.0


def test_pipeline_synthetic_frame():
    """Verify end-to-end perception pipeline on an image frame."""
    pipeline = RoboticsPerceptionPipeline(
        weights_path="yolov8n.pt",
        camera_config_path="configs/camera_config.yaml",
        device="cpu",
        conf_thresh=0.25,
        enable_bev=True,
        enable_temporal_tracking=True
    )

    # Synthetic frame (720, 1280, 3)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    telemetry = pipeline.process_frame(frame)

    assert telemetry.frame_index == 1
    assert telemetry.total_pipeline_latency_ms > 0
    assert telemetry.fps > 0
    assert telemetry.annotated_frame is not None
    assert telemetry.annotated_frame.shape == (720 + 36, 1280, 3) # Includes HUD bar
