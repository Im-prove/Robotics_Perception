"""
Distance Estimation Package for Robotic Navigation.
Provides geometric pinhole modeling, ground-plane intersection,
and optimal Kalman Measurement and Temporal Filter Fusion.
"""
from .geometry import CameraIntrinsics, CameraExtrinsics, load_camera_from_yaml
from .object_priors import ObjectPriors, ObjectDimension
from .pinhole_estimator import PinholeDistanceEstimator, DistanceResult
from .ground_plane_estimator import GroundPlaneDistanceEstimator
from .fusion_estimator import KalmanDistanceFusion, FusedDistanceEstimator
from .kalman_tracker import (
    KalmanTrackState,
    ObstacleDistanceKalmanFilter,
    MultiObjectKalmanTracker
)

__all__ = [
    "CameraIntrinsics",
    "CameraExtrinsics",
    "load_camera_from_yaml",
    "ObjectPriors",
    "ObjectDimension",
    "DistanceResult",
    "PinholeDistanceEstimator",
    "GroundPlaneDistanceEstimator",
    "KalmanDistanceFusion",
    "FusedDistanceEstimator",
    "KalmanTrackState",
    "ObstacleDistanceKalmanFilter",
    "MultiObjectKalmanTracker",
]
