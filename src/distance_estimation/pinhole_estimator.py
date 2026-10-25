"""
Pinhole Camera Model Distance Estimator.
Derives 3D distance from detected bounding box geometry and known physical dimensions.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from .geometry import CameraIntrinsics
from .object_priors import ObjectPriors, ObjectDimension


@dataclass
class DistanceResult:
    """Estimated 3D position and distance from camera perspective."""
    distance_m: float       # Euclidean distance from camera center (sqrt(X^2 + Y^2 + Z^2))
    distance_z_m: float     # Longitudinal depth along optical axis Z
    offset_x_m: float       # Lateral offset (+ right, - left)
    offset_y_m: float       # Vertical offset (+ down, - up)
    confidence: float       # Distance estimation confidence score [0, 1]
    method: str             # Estimation method used ("pinhole_height", "pinhole_width", "ground_plane", "fused")


class PinholeDistanceEstimator:
    """Estimates object 3D distance using pinhole geometry and real-world priors."""

    def __init__(self, intrinsics: CameraIntrinsics):
        self.intrinsics = intrinsics

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
        """
        x1, y1, x2, y2 = bbox_xyxy
        w_pixels = max(1.0, float(x2 - x1))
        h_pixels = max(1.0, float(y2 - y1))
        u_center = float((x1 + x2) / 2.0)
        v_center = float((y1 + y2) / 2.0)

        priors = ObjectPriors.get(class_name)
        if priors is None:
            return None

        # Determine Z depth from height or width
        if prefer_dimension == "width" or (prefer_dimension == "auto" and w_pixels > 2.0 * h_pixels):
            z_depth = (self.intrinsics.fx * priors.width_m) / w_pixels
            method = "pinhole_width"
        else:
            z_depth = (self.intrinsics.fy * priors.height_m) / h_pixels
            method = "pinhole_height"

        # Clip to realistic valid limits
        z_depth = float(np.clip(z_depth, priors.min_dist_m, priors.max_dist_m))

        # 3D coordinates in camera coordinate frame
        x_m = ((u_center - self.intrinsics.cx) * z_depth) / self.intrinsics.fx
        y_m = ((v_center - self.intrinsics.cy) * z_depth) / self.intrinsics.fy
        euclidean_dist = float(np.sqrt(x_m**2 + y_m**2 + z_depth**2))

        # Confidence heuristic: larger bounding boxes and aspect ratio matching priors give higher confidence
        observed_aspect = h_pixels / w_pixels
        expected_aspect = priors.aspect_ratio
        aspect_error = abs(observed_aspect - expected_aspect) / (expected_aspect + 1e-5)
        aspect_score = max(0.2, 1.0 - min(aspect_error, 0.8))
        size_score = min(1.0, h_pixels / 50.0) # Higher confidence when at least 50px tall
        confidence = float(0.5 * aspect_score + 0.5 * size_score)

        return DistanceResult(
            distance_m=euclidean_dist,
            distance_z_m=z_depth,
            offset_x_m=x_m,
            offset_y_m=y_m,
            confidence=confidence,
            method=method
        )
