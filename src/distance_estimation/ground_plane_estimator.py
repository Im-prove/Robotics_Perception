"""
Ground Plane Intersection Distance Estimator.
Estimates distance to ground-standing obstacles (cones, barriers)
using camera height and pitch angle without needing object height.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from .geometry import CameraIntrinsics, CameraExtrinsics
from .pinhole_estimator import DistanceResult


class GroundPlaneDistanceEstimator:
    """Estimates distance using ground-plane contact ray intersection."""

    def __init__(self, intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def estimate_from_bottom_contact(
        self,
        bbox_xyxy: Tuple[float, float, float, float]
    ) -> Optional[DistanceResult]:
        """
        Derivation:
            Camera is mounted at height h_cam above the flat ground plane, pitched down by angle theta.
            For a ground obstacle, the bottom center of the bounding box (u_c, y2)
            represents the ground contact point.

            Angle alpha of the ray relative to camera optical axis:
            alpha = arctan((y2 - cy) / fy)

            Total angle relative to horizontal:
            psi = theta + alpha

            If psi > 0 (ray points down towards ground):
            Z = h_cam / tan(psi)
            X = (u_c - cx) * Z / fx
            Y = h_cam
            D = sqrt(X^2 + Y^2 + Z^2)
        """
        x1, y1, x2, y2 = bbox_xyxy
        u_center = float((x1 + x2) / 2.0)
        v_bottom = float(y2)

        # Angular elevation of bottom contact ray
        alpha = np.arctan2(v_bottom - self.intrinsics.cy, self.intrinsics.fy)
        total_pitch = self.extrinsics.pitch_rad + alpha

        # Ensure ray intersects ground in front of the camera
        if total_pitch <= np.radians(0.5):
            return None # Ray is parallel or pointing up towards the horizon

        z_depth = self.extrinsics.height_m / np.tan(total_pitch)
        if z_depth <= 0.1 or z_depth > 50.0:
            return None

        x_m = ((u_center - self.intrinsics.cx) * z_depth) / self.intrinsics.fx
        y_m = self.extrinsics.height_m
        euclidean_dist = float(np.sqrt(x_m**2 + y_m**2 + z_depth**2))

        # Confidence: higher when contact point is cleanly in lower half of image
        confidence = float(np.clip((v_bottom - self.intrinsics.cy) / (self.intrinsics.image_height / 2.0), 0.2, 0.95))

        return DistanceResult(
            distance_m=euclidean_dist,
            distance_z_m=float(z_depth),
            offset_x_m=float(x_m),
            offset_y_m=float(y_m),
            confidence=confidence,
            method="ground_plane"
        )
