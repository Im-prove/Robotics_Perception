"""
Multi-Hypothesis Kalman Sensor Fusion Distance Estimator.
Combines geometric pinhole apparent-size estimation and ground-plane contact ray intersection
using optimal Kalman Measurement Fusion (Covariance Intersection) and Discrete Temporal Kalman Filtering.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import logging
import numpy as np

from .geometry import CameraIntrinsics, CameraExtrinsics
from .pinhole_estimator import PinholeDistanceEstimator, DistanceResult
from .ground_plane_estimator import GroundPlaneDistanceEstimator
from .kalman_tracker import MultiObjectKalmanTracker, KalmanTrackState

logger = logging.getLogger("DistanceFusion")


class KalmanDistanceFusion:
    """
    Industrial-grade multi-method distance estimator using 1D Kalman state fusion
    and discrete temporal tracking.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        outlier_threshold_sigma: float = 3.0,
        enable_temporal_tracking: bool = True
    ):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.pinhole_est = PinholeDistanceEstimator(intrinsics, extrinsics)
        self.ground_est = GroundPlaneDistanceEstimator(intrinsics, extrinsics)
        self.outlier_threshold_sigma = outlier_threshold_sigma
        self.enable_temporal_tracking = enable_temporal_tracking
        self.temporal_tracker = MultiObjectKalmanTracker(max_missed_frames=15, process_noise_accel=1.5)

    def fuse_measurements(
        self,
        pinhole_res: Optional[DistanceResult],
        ground_res: Optional[DistanceResult],
        class_name: str
    ) -> Optional[DistanceResult]:
        """
        Optimal 1D Kalman Measurement Fusion (Weighted Least Squares / Covariance Intersection):

        State: True distance z
        Hypothesis 1: z_size ~ N(z, sigma_size^2)
        Hypothesis 2: z_ground ~ N(z, sigma_ground^2)

        Optimal Fused Distance:
            z_fused = (sigma_ground^2 * z_size + sigma_size^2 * z_ground) / (sigma_size^2 + sigma_ground^2)

        Optimal Fused Variance:
            sigma_fused^2 = (sigma_size^2 * sigma_ground^2) / (sigma_size^2 + sigma_ground^2)
            1 / sigma_fused^2 = 1 / sigma_size^2 + 1 / sigma_ground^2
        """
        if pinhole_res is None and ground_res is None:
            return None
        if pinhole_res is None:
            return ground_res
        if ground_res is None:
            return pinhole_res

        var_size = max(1e-4, pinhole_res.variance)
        var_ground = max(1e-4, ground_res.variance)

        # Innovation and Normalized Innovation Squared (NIS)
        diff_z = float(pinhole_res.distance_z_m - ground_res.distance_z_m)
        nis = (diff_z * diff_z) / (var_size + var_ground)

        # Mahalanobis outlier gating: if hypotheses disagree significantly (> 3-sigma)
        threshold_nis = self.outlier_threshold_sigma ** 2
        if nis > threshold_nis:
            logger.debug(
                f"Discrepancy detected between size ({pinhole_res.distance_z_m:.2f}m) "
                f"and ground ({ground_res.distance_z_m:.2f}m) for {class_name} (NIS={nis:.1f})."
            )
            # Select the hypothesis with lower uncertainty / higher confidence
            if var_ground < var_size:
                return ground_res
            else:
                return pinhole_res

        # Kalman state fusion
        total_inv_var = (1.0 / var_size) + (1.0 / var_ground)
        fused_var = 1.0 / total_inv_var

        w_size = fused_var / var_size
        w_ground = fused_var / var_ground

        fused_z = float(w_size * pinhole_res.distance_z_m + w_ground * ground_res.distance_z_m)
        fused_dist = float(w_size * pinhole_res.distance_m + w_ground * ground_res.distance_m)
        fused_x = float(w_size * pinhole_res.offset_x_m + w_ground * ground_res.offset_x_m)
        fused_y = float(w_size * pinhole_res.offset_y_m + w_ground * ground_res.offset_y_m)

        # Combined confidence
        fused_conf = float(min(1.0, pinhole_res.confidence + ground_res.confidence * 0.35))

        # Transform to robot base frame
        r_x, r_y, r_z = self.extrinsics.transform_camera_to_robot(fused_x, fused_y, fused_z)

        return DistanceResult(
            distance_m=round(fused_dist, 2),
            distance_z_m=round(fused_z, 2),
            offset_x_m=round(fused_x, 2),
            offset_y_m=round(fused_y, 2),
            confidence=round(fused_conf, 2),
            method="kalman_fused",
            variance=round(fused_var, 4),
            std_dev_m=round(float(np.sqrt(fused_var)), 2),
            robot_x_m=round(r_x, 2) if r_x is not None else None,
            robot_y_m=round(r_y, 2) if r_y is not None else None,
            robot_z_m=round(r_z, 2) if r_z is not None else None,
            is_border_clipped=(pinhole_res.is_border_clipped or ground_res.is_border_clipped)
        )

    def estimate(
        self,
        class_name: str,
        bbox_xyxy: Tuple[float, float, float, float],
        track_id: Optional[int] = None,
        timestamp: float = 0.0,
        ema_alpha: float = 0.35  # Preserved for backward compatibility
    ) -> Optional[DistanceResult]:
        """
        Estimate robust distance by fusing multiple geometric hypotheses.

        - Stop signs are elevated signs; ground contact does not apply -> use pinhole.
        - Cones and barriers touch the ground -> fuse pinhole and ground contact with Kalman filter.
        - Multi-frame temporal Kalman tracking filters frame-to-frame noise and computes velocity.
        """
        norm_class = class_name.strip().lower()

        pinhole_res = self.pinhole_est.estimate_from_bbox(norm_class, bbox_xyxy)

        # Class-specific strategy:
        # If elevated sign (like stop sign), ground plane is not applicable
        if "stop" in norm_class or "sign" in norm_class:
            if pinhole_res is None:
                return None
            res = pinhole_res
        else:
            # Ground-standing object (cone, barrier)
            ground_res = self.ground_est.estimate_from_bottom_contact(bbox_xyxy)
            res = self.fuse_measurements(pinhole_res, ground_res, class_name)

        if res is None:
            return None

        # Multi-frame temporal Kalman Filter tracking across video frames
        if track_id is not None and self.enable_temporal_tracking:
            kalman_state = self.temporal_tracker.update_track(
                track_id=track_id,
                class_name=class_name,
                measured_distance=res.distance_m,
                measurement_variance=res.variance if res.variance > 0 else 0.10,
                timestamp=timestamp
            )
            res.distance_m = kalman_state.distance_m
            res.variance = kalman_state.variance_dist
            res.std_dev_m = float(np.sqrt(kalman_state.variance_dist))

        return res


# Backward-compatible alias
FusedDistanceEstimator = KalmanDistanceFusion
