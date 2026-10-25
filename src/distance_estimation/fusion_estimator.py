"""
Multi-Hypothesis Sensor Fusion Distance Estimator.
Combines geometric pinhole height/width priors and ground-plane contact ray intersection.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
from .geometry import CameraIntrinsics, CameraExtrinsics
from .pinhole_estimator import PinholeDistanceEstimator, DistanceResult
from .ground_plane_estimator import GroundPlaneDistanceEstimator


class FusedDistanceEstimator:
    """Combines pinhole metric projection and ground contact plane estimation."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        pinhole_weight: float = 0.60,
        ground_weight: float = 0.40
    ):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.pinhole_est = PinholeDistanceEstimator(intrinsics)
        self.ground_est = GroundPlaneDistanceEstimator(intrinsics, extrinsics)
        self.pinhole_weight = pinhole_weight
        self.ground_weight = ground_weight
        self._track_history: Dict[int, float] = {}

    def estimate(
        self,
        class_name: str,
        bbox_xyxy: Tuple[float, float, float, float],
        track_id: Optional[int] = None,
        ema_alpha: float = 0.35
    ) -> Optional[DistanceResult]:
        """
        Estimate robust distance by fusing multiple geometric hypotheses.

        - Stop signs are elevated signs; ground contact does not apply -> use pinhole.
        - Cones and barriers touch the ground -> fuse pinhole and ground contact.
        - Optional EMA smoothing across video frames for stable navigation.
        """
        norm_class = class_name.strip().lower()

        pinhole_res = self.pinhole_est.estimate_from_bbox(norm_class, bbox_xyxy)

        # If elevated sign (like stop sign), ground plane is not applicable
        if "stop" in norm_class or "sign" in norm_class:
            if pinhole_res is None:
                return None
            res = pinhole_res
        else:
            # Ground-standing object (cone, barrier)
            ground_res = self.ground_est.estimate_from_bottom_contact(bbox_xyxy)

            if pinhole_res is not None and ground_res is not None:
                # Weighted fusion based on confidence
                w_p = self.pinhole_weight * pinhole_res.confidence
                w_g = self.ground_weight * ground_res.confidence
                total_w = w_p + w_g

                if total_w > 0:
                    fused_dist = (w_p * pinhole_res.distance_m + w_g * ground_res.distance_m) / total_w
                    fused_z = (w_p * pinhole_res.distance_z_m + w_g * ground_res.distance_z_m) / total_w
                    fused_x = (w_p * pinhole_res.offset_x_m + w_g * ground_res.offset_x_m) / total_w
                    fused_y = (w_p * pinhole_res.offset_y_m + w_g * ground_res.offset_y_m) / total_w
                    res = DistanceResult(
                        distance_m=float(fused_dist),
                        distance_z_m=float(fused_z),
                        offset_x_m=float(fused_x),
                        offset_y_m=float(fused_y),
                        confidence=float(min(1.0, pinhole_res.confidence + ground_res.confidence * 0.3)),
                        method="fused"
                    )
                else:
                    res = pinhole_res
            elif pinhole_res is not None:
                res = pinhole_res
            elif ground_res is not None:
                res = ground_res
            else:
                return None

        # Temporal EMA smoothing across video frames if track_id provided
        if track_id is not None and res is not None:
            if track_id in self._track_history:
                prev_dist = self._track_history[track_id]
                smoothed_dist = ema_alpha * res.distance_m + (1.0 - ema_alpha) * prev_dist
                self._track_history[track_id] = smoothed_dist
                res.distance_m = smoothed_dist
            else:
                self._track_history[track_id] = res.distance_m

        return res
