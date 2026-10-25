"""
Unit Tests for Camera Geometry, Kalman Distance Fusion, and Physical Uncertainty.
"""
import pytest
import numpy as np

from src.distance_estimation.geometry import CameraIntrinsics, CameraExtrinsics
from src.distance_estimation.object_priors import ObjectPriors, ObjectDimension
from src.distance_estimation.pinhole_estimator import PinholeDistanceEstimator
from src.distance_estimation.ground_plane_estimator import GroundPlaneDistanceEstimator
from src.distance_estimation.fusion_estimator import KalmanDistanceFusion, FusedDistanceEstimator
from src.distance_estimation.kalman_tracker import ObstacleDistanceKalmanFilter, MultiObjectKalmanTracker


@pytest.fixture
def camera_setup():
    intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0, image_width=1280, image_height=720)
    extrinsics = CameraExtrinsics(height_m=0.65, pitch_deg=2.5, x_offset_m=0.15, y_offset_m=0.0)
    return intrinsics, extrinsics


def test_pinhole_projection_and_ray(camera_setup):
    intrinsics, _ = camera_setup
    u, v = intrinsics.project_3d_to_pixel(x=0.0, y=0.0, z=2.0)
    assert abs(u - 640.0) < 1e-3
    assert abs(v - 360.0) < 1e-3

    # Point at X=0.5m, Y=0.25m, Z=2.0m
    u, v = intrinsics.project_3d_to_pixel(x=0.5, y=0.25, z=2.0)
    assert abs(u - (800.0 * 0.5 / 2.0 + 640.0)) < 1e-3
    assert abs(v - (800.0 * 0.25 / 2.0 + 360.0)) < 1e-3


def test_pinhole_cone_distance_estimation(camera_setup):
    intrinsics, extrinsics = camera_setup
    estimator = PinholeDistanceEstimator(intrinsics, extrinsics)

    # Simulated traffic cone (real height = 0.70m) at 2.0m distance
    # Projected height = fy * H / Z = 800 * 0.70 / 2.0 = 280 pixels
    true_dist = 2.0
    h_px = (intrinsics.fy * 0.70) / true_dist
    w_px = (intrinsics.fx * 0.38) / true_dist

    bbox = (640.0 - w_px / 2, 360.0 - h_px / 2, 640.0 + w_px / 2, 360.0 + h_px / 2)
    res = estimator.estimate_from_bbox("Cone", bbox)

    assert res is not None
    assert abs(res.distance_z_m - true_dist) < 0.05
    assert abs(res.distance_m - true_dist) < 0.05
    assert res.confidence > 0.5
    assert res.variance > 0.0
    assert res.std_dev_m > 0.0


def test_pinhole_stopsign_distance_estimation(camera_setup):
    intrinsics, extrinsics = camera_setup
    estimator = PinholeDistanceEstimator(intrinsics, extrinsics)

    # Simulated stop sign (real height = 0.75m) at 3.2m distance (from assignment prompt)
    true_dist = 3.2
    h_px = (intrinsics.fy * 0.75) / true_dist
    w_px = (intrinsics.fx * 0.75) / true_dist

    bbox = (700.0, 200.0, 700.0 + w_px, 200.0 + h_px)
    res = estimator.estimate_from_bbox("Stop Sign", bbox)

    assert res is not None
    assert abs(res.distance_z_m - true_dist) < 0.05
    assert res.robot_x_m is not None


def test_ground_plane_estimation(camera_setup):
    intrinsics, extrinsics = camera_setup
    ground_est = GroundPlaneDistanceEstimator(intrinsics, extrinsics)

    # An obstacle on the ground at distance Z = 3.0m
    # Z = h_cam / tan(pitch + alpha)
    z_true = 3.0
    total_angle = np.arctan(extrinsics.height_m / z_true)
    alpha = total_angle - extrinsics.pitch_rad
    v_bottom = intrinsics.cy + intrinsics.fy * np.tan(alpha)

    bbox = (600.0, v_bottom - 100, 680.0, v_bottom)
    res = ground_est.estimate_from_bottom_contact(bbox)

    assert res is not None
    assert abs(res.distance_z_m - z_true) < 0.15
    assert res.variance > 0.0
    assert res.method == "ground_plane"


def test_fused_distance_estimator(camera_setup):
    intrinsics, extrinsics = camera_setup
    fused_est = KalmanDistanceFusion(intrinsics, extrinsics)

    # Cone at 1.5m
    true_z = 1.5
    h_px = (intrinsics.fy * 0.70) / true_z
    w_px = (intrinsics.fx * 0.38) / true_z
    bbox = (500.0, 300.0, 500.0 + w_px, 300.0 + h_px)

    res = fused_est.estimate("Cone", bbox)
    assert res is not None
    assert abs(res.distance_m - true_z) < 0.25
    assert res.method in ["kalman_fused", "fused", "pinhole_height"]


def test_kalman_measurement_fusion_reduces_variance(camera_setup):
    intrinsics, extrinsics = camera_setup
    fused_est = KalmanDistanceFusion(intrinsics, extrinsics)

    # Cone with both valid pinhole and ground measurements
    true_z = 2.5
    total_angle = np.arctan(extrinsics.height_m / true_z)
    alpha = total_angle - extrinsics.pitch_rad
    v_bottom = intrinsics.cy + intrinsics.fy * np.tan(alpha)
    h_px = (intrinsics.fy * 0.70) / true_z
    w_px = (intrinsics.fx * 0.38) / true_z
    bbox = (640.0 - w_px/2, v_bottom - h_px, 640.0 + w_px/2, v_bottom)

    pin_res = fused_est.pinhole_est.estimate_from_bbox("Cone", bbox)
    gnd_res = fused_est.ground_est.estimate_from_bottom_contact(bbox)
    fused_res = fused_est.fuse_measurements(pin_res, gnd_res, "Cone")

    assert pin_res is not None
    assert gnd_res is not None
    assert fused_res is not None

    # Information form of Kalman filter guarantees fused variance <= min(individual variances)
    assert fused_res.variance <= min(pin_res.variance, gnd_res.variance) + 1e-4
    assert fused_res.method == "kalman_fused"


def test_border_clipping_penalization(camera_setup):
    intrinsics, extrinsics = camera_setup
    estimator = PinholeDistanceEstimator(intrinsics, extrinsics)

    # Box inside frame
    bbox_inside = (400.0, 200.0, 480.0, 450.0)
    res_inside = estimator.estimate_from_bbox("Cone", bbox_inside)

    # Box clipped at image border (y1 <= 2)
    bbox_clipped = (400.0, 1.0, 480.0, 251.0)
    res_clipped = estimator.estimate_from_bbox("Cone", bbox_clipped)

    assert res_inside is not None
    assert res_clipped is not None
    assert res_clipped.is_border_clipped is True
    # Clipped box must have higher uncertainty / variance
    assert res_clipped.variance > res_inside.variance


def test_camera_to_robot_frame_transform(camera_setup):
    _, extrinsics = camera_setup
    # Object at X_c=0, Y_c=0, Z_c=3.0m in camera frame
    rx, ry, rz = extrinsics.transform_camera_to_robot(x_cam=0.0, y_cam=0.0, z_cam=3.0)

    # Expected forward distance: 3.0 * cos(pitch) + x_offset
    expected_x = 3.0 * np.cos(extrinsics.pitch_rad) + extrinsics.x_offset_m
    assert abs(rx - expected_x) < 1e-3
    assert abs(ry - 0.0) < 1e-3

    # Invert back to camera frame
    cx, cy, cz = extrinsics.transform_robot_to_camera(rx, ry, rz)
    assert abs(cx - 0.0) < 1e-3
    assert abs(cy - 0.0) < 1e-3
    assert abs(cz - 3.0) < 1e-3


def test_temporal_kalman_tracker_and_ttc():
    tracker = MultiObjectKalmanTracker(process_noise_accel=1.0)

    # Target starts at 5.0m at t=0.0s
    s1 = tracker.update_track(track_id=1, class_name="Cone", measured_distance=5.0, measurement_variance=0.05, timestamp=0.0)
    assert s1.distance_m == 5.0

    # Target moves to 4.5m at t=0.5s (approach speed ~ 1.0 m/s closing in)
    s2 = tracker.update_track(track_id=1, class_name="Cone", measured_distance=4.5, measurement_variance=0.05, timestamp=0.5)
    assert s2.velocity_mps < 0.0 # Velocity is negative (closing in)
    assert s2.ttc_seconds is not None # Time-to-collision is calculated
    assert 3.0 <= s2.ttc_seconds <= 6.5
