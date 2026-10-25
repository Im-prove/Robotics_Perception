"""
Distance estimation module shim for Problem Statement layout compatibility.
Exports PinholeDistanceEstimator, GroundPlaneDistanceEstimator, KalmanDistanceFusion, FusedDistanceEstimator, DistanceResult.
"""
from ..distance_estimation.pinhole_estimator import PinholeDistanceEstimator, DistanceResult
from ..distance_estimation.ground_plane_estimator import GroundPlaneDistanceEstimator
from ..distance_estimation.fusion_estimator import KalmanDistanceFusion, FusedDistanceEstimator
from ..distance_estimation.object_priors import ObjectPriors, ObjectDimension

# Standalone helper functions matching Problem Statement Section 4 equations directly:
import math
from typing import Optional


def estimate_distance_from_size(
    focal_y_px: float,
    object_height_m: float,
    bbox_height_px: float
) -> Optional[float]:
    """
    Pinhole camera relationship (Problem Statement Section 4.A):
    Z = (f_y * H) / h
    """
    if focal_y_px <= 0 or object_height_m <= 0 or bbox_height_px <= 1.0:
        return None
    return float((focal_y_px * object_height_m) / bbox_height_px)


def estimate_ground_distance(
    bottom_y_px: float,
    camera_height_m: float,
    focal_y_px: float,
    principal_y_px: float,
    pitch_down_deg: float
) -> Optional[float]:
    """
    Ground plane intersection relationship (Problem Statement Section 4.B):
    alpha = arctan((v - c_y) / f_y)
    D = H_c / tan(theta + alpha)
    """
    if camera_height_m <= 0 or focal_y_px <= 0:
        return None
    ray_angle = math.atan((bottom_y_px - principal_y_px) / focal_y_px)
    depression = math.radians(pitch_down_deg) + ray_angle
    if depression <= math.radians(0.75):
        return None
    tan_dep = math.tan(depression)
    if tan_dep < 1e-4:
        return None
    return float(camera_height_m / tan_dep)


__all__ = [
    "PinholeDistanceEstimator",
    "GroundPlaneDistanceEstimator",
    "KalmanDistanceFusion",
    "FusedDistanceEstimator",
    "DistanceResult",
    "ObjectPriors",
    "ObjectDimension",
    "estimate_distance_from_size",
    "estimate_ground_distance",
]
