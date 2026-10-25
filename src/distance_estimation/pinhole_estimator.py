"""
Pinhole Camera Model Distance Estimator.
Derives 3D distance from detected bounding box geometry and known physical dimensions.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from .geometry import CameraIntrinsics, CameraExtrinsics
from .object_priors import ObjectPriors, ObjectDimension


@dataclass
class DistanceResult:
    """Estimated 3D position and distance from camera perspective."""
    distance_m: float       # Euclidean distance from camera center (sqrt(X^2 + Y^2 + Z^2))
    distance_z_m: float     # Longitudinal depth along optical axis Z
    offset_x_m: float       # Lateral offset (+ right, - left)
    offset_y_m: float       # Vertical offset (+ down, - up)
    confidence: float       # Distance estimation confidence score [0, 1]
    method: str             # Estimation method used ("pinhole_height", "pinhole_width", "ground_plane", "kalman_fused")
    variance: float = 0.0   # Measurement uncertainty variance (sigma^2 in m^2)
    std_dev_m: float = 0.0  # Standard deviation (sigma in m)
    robot_x_m: Optional[float] = None # Forward distance in robot base_link frame (+X forward)
    robot_y_m: Optional[float] = None # Lateral offset in robot base_link frame (+Y left)
    robot_z_m: Optional[float] = None # Height in robot base_link frame (+Z up)
    is_border_clipped: bool = False   # True if bounding box intersects frame boundaries

    def __post_init__(self):
        if self.variance > 0 and self.std_dev_m == 0.0:
            self.std_dev_m = float(np.sqrt(self.variance))


class PinholeDistanceEstimator:
    """Estimates object 3D distance using pinhole geometry, real-world priors, and analytical uncertainty."""

    def __init__(self, intrinsics: CameraIntrinsics, extrinsics: Optional[CameraExtrinsics] = None):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    def estimate_from_bbox(
        self,
        class_name: str,
        bbox_xyxy: Tuple[float, float, float, float],
        prefer_dimension: str = "height"
    ) -> Optional[DistanceResult]:
        """
        Estimate distance for a bounding box [x1, y1, x2, y2].

        Derivation:
            From similar triangles:
            h_pixels / f_y = H_real / Z
            => Z = (f_y * H_real) / h_pixels
            w_pixels / f_x = W_real / Z
            => Z = (f_x * W_real) / w_pixels

            X = (u_center - cx) * Z / f_x
            Y = (v_center - cy) * Z / f_y
            D = sqrt(X^2 + Y^2 + Z^2)

        Uncertainty Derivation (Error Propagation):
            dZ/dh = - (f_y * H) / h^2 = - Z^2 / (f_y * H)
            sigma_Z^2 = (Z^2 / (f_y * H))^2 * sigma_h^2 + (Z / H)^2 * sigma_H^2
        """
        x1, y1, x2, y2 = bbox_xyxy
        w_pixels = max(1.0, float(x2 - x1))
        h_pixels = max(1.0, float(y2 - y1))
        u_center = float((x1 + x2) / 2.0)
        v_center = float((y1 + y2) / 2.0)

        # Border clipping detection (object partially outside camera FOV)
        border_margin = 3.0
        is_clipped = (
            x1 <= border_margin
            or y1 <= border_margin
            or x2 >= (self.intrinsics.image_width - border_margin)
            or y2 >= (self.intrinsics.image_height - border_margin)
        )

        priors = ObjectPriors.get(class_name)
        if priors is None:
            return None

        # Determine Z depth from height or width
        if prefer_dimension == "width" or (prefer_dimension == "auto" and w_pixels > 2.0 * h_pixels):
            dim_real = priors.width_m
            dim_px = w_pixels
            focal = self.intrinsics.fx
            method = "pinhole_width"
        else:
            dim_real = priors.height_m
            dim_px = h_pixels
            focal = self.intrinsics.fy
            method = "pinhole_height"

        z_depth = (focal * dim_real) / dim_px

        # Clip to realistic valid limits
        z_depth = float(np.clip(z_depth, priors.min_dist_m, priors.max_dist_m))

        # 3D coordinates in camera coordinate frame
        x_m = ((u_center - self.intrinsics.cx) * z_depth) / self.intrinsics.fx
        y_m = ((v_center - self.intrinsics.cy) * z_depth) / self.intrinsics.fy
        euclidean_dist = float(np.sqrt(x_m**2 + y_m**2 + z_depth**2))

        # Analytical measurement variance calculation
        # sigma_px = 1.5 pixels (standard edge detector localization uncertainty)
        # sigma_H = 0.05 * dim_real (5% physical manufacturing variation)
        sigma_px = 1.5
        sigma_prior = 0.05 * dim_real

        dZ_dpx = - (z_depth**2) / (focal * dim_real + 1e-6)
        dZ_dH = z_depth / (dim_real + 1e-6)
        var_z = (dZ_dpx * sigma_px)**2 + (dZ_dH * sigma_prior)**2

        # Floor variance to sensible minimum
        var_z = max(0.01, float(var_z))

        # Border clipping penalty: clipped bounding boxes underestimate pixel size and overestimate distance
        if is_clipped:
            var_z *= 5.0

        # Confidence heuristic: aspect ratio consistency and sufficient pixel resolution
        observed_aspect = h_pixels / w_pixels
        expected_aspect = priors.aspect_ratio
        aspect_error = abs(observed_aspect - expected_aspect) / (expected_aspect + 1e-5)
        aspect_score = max(0.2, 1.0 - min(aspect_error, 0.8))
        size_score = min(1.0, h_pixels / 50.0) # Higher confidence when at least 50px tall
        base_conf = float(0.5 * aspect_score + 0.5 * size_score)
        if is_clipped:
            base_conf *= 0.65

        # Transform to robot base frame if extrinsics configured
        r_x, r_y, r_z = None, None, None
        if self.extrinsics is not None:
            r_x, r_y, r_z = self.extrinsics.transform_camera_to_robot(x_m, y_m, z_depth)

        return DistanceResult(
            distance_m=euclidean_dist,
            distance_z_m=z_depth,
            offset_x_m=x_m,
            offset_y_m=y_m,
            confidence=base_conf,
            method=method,
            variance=var_z,
            std_dev_m=float(np.sqrt(var_z)),
            robot_x_m=r_x,
            robot_y_m=r_y,
            robot_z_m=r_z,
            is_border_clipped=is_clipped
        )
