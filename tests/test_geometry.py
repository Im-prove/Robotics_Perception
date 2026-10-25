"""
Unit Tests for Camera Geometry and Distance Estimation.
"""
import pytest
import numpy as np

from src.distance_estimation.geometry import CameraIntrinsics, CameraExtrinsics
from src.distance_estimation.object_priors import ObjectPriors, ObjectDimension
from src.distance_estimation.pinhole_estimator import PinholeDistanceEstimator
from src.distance_estimation.ground_plane_estimator import GroundPlaneDistanceEstimator
from src.distance_estimation.fusion_estimator import FusedDistanceEstimator


@pytest.fixture
def camera_setup():
    intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0)
    extrinsics = CameraExtrinsics(height_m=0.65, pitch_deg=2.5)
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
    intrinsics, _ = camera_setup
    estimator = PinholeDistanceEstimator(intrinsics)

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


def test_pinhole_stopsign_distance_estimation(camera_setup):
    intrinsics, _ = camera_setup
    estimator = PinholeDistanceEstimator(intrinsics)

    # Simulated stop sign (real height = 0.75m) at 3.2m distance (from assignment prompt)
    true_dist = 3.2
    h_px = (intrinsics.fy * 0.75) / true_dist
    w_px = (intrinsics.fx * 0.75) / true_dist

    bbox = (700.0, 200.0, 700.0 + w_px, 200.0 + h_px)
    res = estimator.estimate_from_bbox("Stop Sign", bbox)

    assert res is not None
    assert abs(res.distance_z_m - true_dist) < 0.05


def test_ground_plane_estimation(camera_setup):
    intrinsics, extrinsics = camera_setup
    ground_est = GroundPlaneDistanceEstimator(intrinsics, extrinsics)

    # An obstacle on the ground at distance Z = 3.0m
    # Z = h_cam / tan(pitch + alpha)
    # tan(total_angle) = h_cam / Z
    z_true = 3.0
    total_angle = np.arctan(extrinsics.height_m / z_true)
    alpha = total_angle - extrinsics.pitch_rad
    v_bottom = intrinsics.cy + intrinsics.fy * np.tan(alpha)

    bbox = (600.0, v_bottom - 100, 680.0, v_bottom)
    res = ground_est.estimate_from_bottom_contact(bbox)

    assert res is not None
    assert abs(res.distance_z_m - z_true) < 0.15


def test_fused_distance_estimator(camera_setup):
    intrinsics, extrinsics = camera_setup
    fused_est = FusedDistanceEstimator(intrinsics, extrinsics)

    # Cone at 1.5m
    true_z = 1.5
    h_px = (intrinsics.fy * 0.70) / true_z
    w_px = (intrinsics.fx * 0.38) / true_z
    bbox = (500.0, 300.0, 500.0 + w_px, 300.0 + h_px)

    res = fused_est.estimate("Cone", bbox)
    assert res is not None
    assert abs(res.distance_m - true_z) < 0.25
    assert res.method in ["fused", "pinhole_height"]
