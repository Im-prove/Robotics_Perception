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

        # Angular elevation of bottom contact ray relative to optical axis
        alpha = np.arctan2(v_bottom - self.intrinsics.cy, self.intrinsics.fy)
        total_pitch = self.extrinsics.pitch_rad + alpha

        # Ensure ray intersects ground in front of the camera (avoid horizon singularity)
        min_depression_rad = np.radians(0.75) # 0.75 degrees minimum depression
        if total_pitch <= min_depression_rad:
            return None # Ray is parallel or pointing up towards the horizon

        tan_pitch = float(np.tan(total_pitch))
        if tan_pitch < 1e-4:
            return None

        h_cam = self.extrinsics.height_m
        z_depth = h_cam / tan_pitch
        if z_depth <= 0.1 or z_depth > 50.0:
            return None

        x_m = ((u_center - self.intrinsics.cx) * z_depth) / self.intrinsics.fx
        # Y in camera frame for ground contact point
        y_m = ((v_bottom - self.intrinsics.cy) * z_depth) / self.intrinsics.fy
        euclidean_dist = float(np.sqrt(x_m**2 + y_m**2 + z_depth**2))

        # Analytical variance derivation via error propagation:
        # Z = h_cam / tan(theta + alpha), alpha = atan((v - cy) / fy)
        # dZ/dpsi = - h_cam * sec^2(psi) / tan^2(psi)
        # dalpha/dv = cos^2(alpha) / fy
        # sigma_v = 1.5 px (bottom boundary detection uncertainty)
        # sigma_theta = 0.0087 rad (~0.5 deg vehicle pitch vibration)
        # sigma_h = 0.02 m (2 cm suspension height variation)
        sigma_v = 1.5
        sigma_theta = np.radians(0.5)
        sigma_h = 0.02

        cos_alpha = np.cos(alpha)
        sec_pitch = 1.0 / np.cos(total_pitch)
        dZ_dpsi = - h_cam * (sec_pitch**2) / (tan_pitch**2 + 1e-6)
        dalpha_dv = (cos_alpha**2) / self.intrinsics.fy
        dZ_dv = dZ_dpsi * dalpha_dv
        dZ_dh = 1.0 / (tan_pitch + 1e-6)

        var_ground = (dZ_dv * sigma_v)**2 + (dZ_dpsi * sigma_theta)**2 + (dZ_dh * sigma_h)**2
        var_ground = max(0.01, float(var_ground))

        # Confidence: higher when contact point is cleanly in lower half of image and not near horizon
        confidence = float(np.clip((v_bottom - self.intrinsics.cy) / (self.intrinsics.image_height / 2.0), 0.2, 0.95))

        # Transform to robot base frame
        r_x, r_y, r_z = self.extrinsics.transform_camera_to_robot(x_m, y_m, z_depth)

        return DistanceResult(
            distance_m=euclidean_dist,
            distance_z_m=float(z_depth),
            offset_x_m=float(x_m),
            offset_y_m=float(y_m),
            confidence=confidence,
            method="ground_plane",
            variance=var_ground,
            std_dev_m=float(np.sqrt(var_ground)),
            robot_x_m=r_x,
            robot_y_m=r_y,
            robot_z_m=r_z,
            is_border_clipped=(v_bottom >= self.intrinsics.image_height - 2)
        )
