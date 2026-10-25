"""
Unified Robotics Navigation Perception Pipeline.
Orchestrates Object Detection, Multi-Hypothesis Kalman Distance Fusion,
Temporal Kalman Tracking, 3D Coordinate Mapping, and Visual HUD Annotation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import time
import logging
import cv2
import numpy as np

from ..detector.base_detector import BaseDetector, Detection
from ..detector.yolo_detector import YOLODetector
from ..distance_estimation.geometry import CameraIntrinsics, CameraExtrinsics, load_camera_from_yaml
from ..distance_estimation.fusion_estimator import KalmanDistanceFusion
from ..distance_estimation.kalman_tracker import MultiObjectKalmanTracker, KalmanTrackState
from ..extra_credit.homography_bev import BirdEyeViewMapper, BEVConfig
from ..utils.visualization import Visualizer

logger = logging.getLogger("RoboticsPerceptionPipeline")


@dataclass
class FramePerceptionTelemetry:
    """Comprehensive telemetry output for a processed perception frame."""
    frame_index: int
    timestamp: float
    inference_latency_ms: float
    total_pipeline_latency_ms: float
    fps: float
    detections: List[Detection] = field(default_factory=list)
    tracks: Dict[int, KalmanTrackState] = field(default_factory=dict)
    annotated_frame: Optional[np.ndarray] = None
    bev_grid_map: Optional[np.ndarray] = None


class RoboticsPerceptionPipeline:
    """
    Industrial-grade perception stack for Autonomous Mobile Robots (AMR).
    Conforms to real-time robotics constraints and production deployment standards.
    """

    def __init__(
        self,
        weights_path: str = "weights/robotics_best.pt",
        camera_config_path: str = "configs/camera_config.yaml",
        device: str = "cpu",
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        enable_bev: bool = True,
        enable_temporal_tracking: bool = True
    ):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.enable_bev = enable_bev
        self.enable_temporal_tracking = enable_temporal_tracking
        self.frame_count = 0

        # Load camera calibration geometry
        if Path(camera_config_path).exists():
            self.intrinsics, self.extrinsics = load_camera_from_yaml(camera_config_path)
            logger.info(f"Loaded camera config from {camera_config_path}")
        else:
            logger.warning(f"Camera config {camera_config_path} not found. Using industrial default parameters.")
            self.intrinsics = CameraIntrinsics(fx=800.0, fy=800.0, cx=640.0, cy=360.0, image_width=1280, image_height=720)
            self.extrinsics = CameraExtrinsics(height_m=0.65, pitch_deg=2.5)

        # Initialize multi-hypothesis Kalman distance fusion engine
        self.distance_engine = KalmanDistanceFusion(
            intrinsics=self.intrinsics,
            extrinsics=self.extrinsics,
            enable_temporal_tracking=False # We manage temporal tracking centrally in pipeline
        )

        # Initialize detector
        if not Path(weights_path).exists():
            # Fall back to base YOLO if custom weights not yet built
            if Path("yolov8n.pt").exists():
                weights_path = "yolov8n.pt"
            elif Path("yolo11n.pt").exists():
                weights_path = "yolo11n.pt"

        self.detector = YOLODetector(
            weights_path=weights_path,
            device=device,
            distance_estimator=self.distance_engine
        )

        # Initialize Bird's Eye View mapper
        if self.enable_bev:
            self.bev_mapper = BirdEyeViewMapper(
                intrinsics=self.intrinsics,
                extrinsics=self.extrinsics,
                bev_config=BEVConfig(x_range_m=(-3.0, 3.0), z_range_m=(0.5, 12.0), output_size=(600, 600))
            )
        else:
            self.bev_mapper = None

        # Temporal Multi-Object Kalman Filter
        self.kalman_tracker = MultiObjectKalmanTracker(max_missed_frames=10, process_noise_accel=1.5)
        self._next_track_id = 1
        self._active_track_boxes: Dict[int, Tuple[float, float, float, float]] = {}

    def _associate_detections_to_tracks(
        self,
        detections: List[Detection]
    ) -> List[Tuple[Detection, int]]:
        """
        Associates detected 2D bounding boxes to existing tracked objects using IoU and centroid proximity.
        """
        associated = []
        unmatched_dets = []

        # Greedy association based on bbox overlap
        assigned_track_ids = set()
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            best_tid = None
            best_iou = 0.25 # minimum IoU threshold for matching

            for tid, prev_box in self._active_track_boxes.items():
                if tid in assigned_track_ids:
                    continue
                # Calculate IoU
                px1, py1, px2, py2 = prev_box
                ix1 = max(x1, px1)
                iy1 = max(y1, py1)
                ix2 = min(x2, px2)
                iy2 = min(y2, py2)
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                intersection = iw * ih

                area1 = (x2 - x1) * (y2 - y1)
                area2 = (px2 - px1) * (py2 - py1)
                union = area1 + area2 - intersection
                iou = intersection / (union + 1e-6)

                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                assigned_track_ids.add(best_tid)
                self._active_track_boxes[best_tid] = det.bbox_xyxy
                det.track_id = best_tid
                associated.append((det, best_tid))
            else:
                unmatched_dets.append(det)

        # Assign new IDs for unmatched detections
        for det in unmatched_dets:
            new_id = self._next_track_id
            self._next_track_id += 1
            det.track_id = new_id
            self._active_track_boxes[new_id] = det.bbox_xyxy
            associated.append((det, new_id))

        # Coast tracks that had no detections
        current_active = [tid for _, tid in associated]
        for tid in list(self._active_track_boxes.keys()):
            if tid not in current_active:
                del self._active_track_boxes[tid]

        return associated

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> FramePerceptionTelemetry:
        """
        Process a single image frame through the entire robotics navigation pipeline.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty or None.")

        t_start = time.perf_counter()
        curr_time = timestamp if timestamp is not None else t_start
        self.frame_count += 1

        # 1. Detection Inference
        t_infer_start = time.perf_counter()
        detections = self.detector.predict(
            image=frame,
            conf_thresh=self.conf_thresh,
            filter_target_only=False
        )
        t_infer_end = time.perf_counter()
        infer_latency_ms = (t_infer_end - t_infer_start) * 1000.0

        # 2. Tracking Association & Temporal Kalman Filtering
        tracked_states: Dict[int, KalmanTrackState] = {}
        if self.enable_temporal_tracking and detections:
            associated = self._associate_detections_to_tracks(detections)
            active_tids = []

            for det, tid in associated:
                active_tids.append(tid)
                if det.distance_m is not None:
                    meas_var = det.variance if (det.variance and det.variance > 0) else 0.15
                    k_state = self.kalman_tracker.update_track(
                        track_id=tid,
                        class_name=det.class_name,
                        measured_distance=det.distance_m,
                        measurement_variance=meas_var,
                        timestamp=curr_time
                    )
                    tracked_states[tid] = k_state
                    # Update detection distance with temporal Kalman filtered state
                    det.distance_m = k_state.distance_m
                    det.variance = k_state.variance_dist
                    det.std_dev_m = float(np.sqrt(k_state.variance_dist))

            self.kalman_tracker.predict_coasting(active_tids, curr_time)

        # 3. Bird's Eye View Grid Mapping
        bev_grid = None
        if self.enable_bev and self.bev_mapper is not None:
            bev_bg = self.bev_mapper.warp_image_to_bev(frame)
            bev_grid = self.bev_mapper.render_bev_grid_map(detections, background_bev=bev_bg)

        # 4. Total Pipeline Latency & FPS
        t_end = time.perf_counter()
        total_latency_ms = (t_end - t_start) * 1000.0
        fps = 1000.0 / max(0.1, total_latency_ms)

        # 5. Visual Dashboard Rendering
        dashboard = Visualizer.render_dashboard(
            main_frame=frame,
            detections=detections,
            bev_map=bev_grid,
            fps=fps,
            model_name=Path(self.detector.weights_path).name
        )

        return FramePerceptionTelemetry(
            frame_index=self.frame_count,
            timestamp=curr_time,
            inference_latency_ms=round(infer_latency_ms, 2),
            total_pipeline_latency_ms=round(total_latency_ms, 2),
            fps=round(fps, 1),
            detections=detections,
            tracks=tracked_states,
            annotated_frame=dashboard,
            bev_grid_map=bev_grid
        )
