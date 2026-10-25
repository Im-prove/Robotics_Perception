"""
Unit Tests for Extra Credit: Stereo Epipolar Depth, BEV Homography, and Optical Flow.
"""
import pytest
import numpy as np

from src.distance_estimation.geometry import CameraIntrinsics, CameraExtrinsics
from src.detector.base_detector import Detection
from src.extra_credit.epipolar_geometry import StereoDepthEstimator, StereoConfig
from src.extra_credit.homography_bev import BirdEyeViewMapper
from src.extra_credit.optical_flow_tracker import OpticalFlowTracker


def test_stereo_depth_derivation():
    """Verify disparity-to-depth formula Z = f * B / d."""
    cfg = StereoConfig(baseline_m=0.12, focal_length_px=800.0)
    stereo_est = StereoDepthEstimator(cfg)

    # For disparity d = 32 pixels:
    # Z = 800 * 0.12 / 32 = 96 / 32 = 3.0 meters
    disp = np.full((100, 100), 32.0, dtype=np.float32)
    depth = stereo_est.disparity_to_depth(disp)

    assert abs(depth[50, 50] - 3.0) < 1e-4

    # For disparity d = 64 pixels:
    # Z = 800 * 0.12 / 64 = 1.5 meters
    disp2 = np.full((100, 100), 64.0, dtype=np.float32)
    depth2 = stereo_est.disparity_to_depth(disp2)
    assert abs(depth2[50, 50] - 1.5) < 1e-4


def test_bev_homography_mapping():
    intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0)
    extrinsics = CameraExtrinsics(height_m=0.65, pitch_deg=2.5)
    bev_mapper = BirdEyeViewMapper(intrinsics, extrinsics)

    # Verify homography matrix exists and is non-singular
    assert bev_mapper.homography_matrix.shape == (3, 3)
    det = np.linalg.det(bev_mapper.homography_matrix)
    assert abs(det) > 1e-6

    # Test coordinate conversion
    u, v = bev_mapper.metric_to_bev_pixel(x_m=0.0, z_m=5.0)
    assert 0 <= u < 600
    assert 0 <= v < 600


def test_optical_flow_tracker():
    tracker = OpticalFlowTracker()
    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    det = Detection(
        class_id=0,
        class_name="Cone",
        confidence=0.9,
        bbox_xyxy=(200, 200, 260, 300),
        distance_m=4.0
    )

    # Frame 1 update
    tracks1 = tracker.update(frame1, [det], timestamp=0.0)
    assert len(tracks1) == 1
    assert tracks1[0].current_distance_m == 4.0

    # Frame 2 update: cone moved closer to 3.5m (approaching)
    det.distance_m = 3.5
    det.bbox_xyxy = (205, 205, 275, 315)
    tracks2 = tracker.update(frame2, [det], timestamp=0.033)
    assert len(tracks2) == 1
    # Approach velocity should be positive (closing in)
    assert tracks2[0].velocity_mps > 0
