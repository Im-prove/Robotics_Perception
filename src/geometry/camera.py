"""
Camera geometry module shim for Problem Statement layout compatibility.
Exports CameraIntrinsics, CameraExtrinsics, load_camera_from_yaml.
"""
from ..distance_estimation.geometry import (
    CameraIntrinsics,
    CameraExtrinsics,
    load_camera_from_yaml
)

__all__ = [
    "CameraIntrinsics",
    "CameraExtrinsics",
    "load_camera_from_yaml",
]
