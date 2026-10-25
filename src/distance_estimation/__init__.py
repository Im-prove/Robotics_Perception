"""
Distance Estimation Package for Robotic Navigation
Provides geometric pinhole modeling, ground-plane intersection, and multi-hypothesis sensor fusion.
"""
from .geometry import CameraIntrinsics, CameraExtrinsics
from .object_priors import ObjectPriors, ObjectDimension
from .pinhole_estimator import PinholeDistanceEstimator
from .ground_plane_estimator import GroundPlaneDistanceEstimator
from .fusion_estimator import FusedDistanceEstimator

__all__ = [
    "CameraIntrinsics",
    "CameraExtrinsics",
    "ObjectPriors",
    "ObjectDimension",
    "PinholeDistanceEstimator",
    "GroundPlaneDistanceEstimator",
    "FusedDistanceEstimator",
]
