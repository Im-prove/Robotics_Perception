"""
Optical Flow Obstacle Tracker and Time-to-Collision (TTC) Estimator.
Tracks dynamic obstacles (e.g., rolling cones, moving obstacles) across consecutive video frames
using Lucas-Kanade feature flow and computes approach velocity and Time-To-Collision.
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
from ..detector.base_detector import Detection


@dataclass
class TrackedObstacle:
    """Obstacle state tracked across consecutive video frames."""
    track_id: int
    class_name: str
    current_bbox: Tuple[float, float, float, float]
    current_distance_m: float
    velocity_mps: float       # Approach velocity (positive = closing in, negative = moving away)
    ttc_seconds: Optional[float] # Time to collision
    trajectory: List[Tuple[int, int]] # Historical (x, y) centroids
    features: np.ndarray      # Tracked corner points (N, 1, 2)


class OpticalFlowTracker:
    """Tracks obstacles across video frames using Lucas-Kanade optical flow."""

    def __init__(self, max_corners: int = 15, max_history: int = 20):
        self.max_corners = max_corners
        self.max_history = max_history
        self.tracks: Dict[int, TrackedObstacle] = {}
        self.next_track_id = 1
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_time: Optional[float] = None

        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

    def update(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        timestamp: float
    ) -> List[TrackedObstacle]:
        """
        Update obstacle tracks with new frame and detections.
        Computes optical flow vectors, distance changes, and Time-To-Collision (TTC).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        dt = (timestamp - self.prev_time) if self.prev_time is not None else 0.033
        dt = max(0.001, dt)

        active_tracks: List[TrackedObstacle] = []

        # If we have previous frame and tracks, propagate tracked features using LK flow
        if self.prev_gray is not None and self.tracks:
            for track_id, track in list(self.tracks.items()):
                if len(track.features) > 0:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, track.features, None, **self.lk_params
                    )
                    good_new = p1[st == 1] if p1 is not None else []
                    if len(good_new) >= 3:
                        track.features = good_new.reshape(-1, 1, 2)
                    else:
                        track.features = np.empty((0, 1, 2), dtype=np.float32)

        # Match new detections to existing tracks using IoU / spatial proximity
        unmatched_detections = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            dist = det.distance_m if det.distance_m is not None else 5.0

            best_id = None
            best_dist = float("inf")

            for t_id, track in self.tracks.items():
                if track.trajectory:
                    last_cx, last_cy = track.trajectory[-1]
                    pixel_dist = np.hypot(cx - last_cx, cy - last_cy)
                    if pixel_dist < 80.0 and pixel_dist < best_dist:
                        best_dist = pixel_dist
                        best_id = t_id

            if best_id is not None:
                # Update matched track
                t = self.tracks[best_id]
                prev_dist = t.current_distance_m
                # Approach velocity: rate at which distance decreases (closing in is positive)
                delta_d = prev_dist - dist
                instant_vel = delta_d / dt
                t.velocity_mps = 0.7 * t.velocity_mps + 0.3 * instant_vel if t.velocity_mps != 0.0 else instant_vel
                t.current_distance_m = dist
                t.current_bbox = det.bbox_xyxy
                t.trajectory.append((cx, cy))
                if len(t.trajectory) > self.max_history:
                    t.trajectory.pop(0)

                # Compute Time-To-Collision (TTC) in seconds
                if t.velocity_mps > 0.1:  # Closing in
                    t.ttc_seconds = max(0.1, t.current_distance_m / t.velocity_mps)
                else:
                    t.ttc_seconds = None

                # Refresh feature points inside bbox
                bbox_mask = np.zeros_like(gray)
                bx1, by1, bx2, by2 = [int(v) for v in det.bbox_xyxy]
                bbox_mask[max(0, by1):min(gray.shape[0], by2), max(0, bx1):min(gray.shape[1], bx2)] = 255
                pts = cv2.goodFeaturesToTrack(gray, mask=bbox_mask, **self.feature_params)
                if pts is not None:
                    t.features = pts

                det.track_id = best_id
                active_tracks.append(t)
            else:
                unmatched_detections.append(det)

        # Initialize new tracks for unmatched detections
        for det in unmatched_detections:
            t_id = self.next_track_id
            self.next_track_id += 1
            x1, y1, x2, y2 = det.bbox_xyxy
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            dist = det.distance_m if det.distance_m is not None else 5.0

            bbox_mask = np.zeros_like(gray)
            bx1, by1, bx2, by2 = [int(v) for v in det.bbox_xyxy]
            bbox_mask[max(0, by1):min(gray.shape[0], by2), max(0, bx1):min(gray.shape[1], bx2)] = 255
            pts = cv2.goodFeaturesToTrack(gray, mask=bbox_mask, **self.feature_params)
            features = pts if pts is not None else np.empty((0, 1, 2), dtype=np.float32)

            new_track = TrackedObstacle(
                track_id=t_id,
                class_name=det.class_name,
                current_bbox=det.bbox_xyxy,
                current_distance_m=dist,
                velocity_mps=0.0,
                ttc_seconds=None,
                trajectory=[(cx, cy)],
                features=features
            )
            self.tracks[t_id] = new_track
            det.track_id = t_id
            active_tracks.append(new_track)

        self.prev_gray = gray.copy()
        self.prev_time = timestamp
        return active_tracks

    def render_tracking_overlay(self, image: np.ndarray, tracks: List[TrackedObstacle]) -> np.ndarray:
        """Draw motion trajectories, optical flow vectors, and collision alert badges."""
        canvas = image.copy()

        for t in tracks:
            # Draw trajectory path
            pts = t.trajectory
            for i in range(1, len(pts)):
                thickness = int(np.sqrt(16.0 / (len(pts) - i + 1))) + 1
                cv2.line(canvas, pts[i - 1], pts[i], (0, 255, 255), thickness, cv2.LINE_AA)

            # Draw optical flow feature points
            if len(t.features) > 0:
                for pt in t.features:
                    fx, fy = int(pt[0][0]), int(pt[0][1])
                    cv2.circle(canvas, (fx, fy), 2, (0, 255, 0), -1)

            # Collision hazard warning badge
            x1, y1, x2, y2 = [int(v) for v in t.current_bbox]
            if t.ttc_seconds is not None:
                if t.ttc_seconds < 2.0:
                    badge_color = (0, 0, 255) # Red CRITICAL
                    status_text = f"CRITICAL TTC: {t.ttc_seconds:.1f}s"
                elif t.ttc_seconds < 4.0:
                    badge_color = (0, 165, 255) # Orange CAUTION
                    status_text = f"CAUTION TTC: {t.ttc_seconds:.1f}s"
                else:
                    badge_color = (0, 255, 0)
                    status_text = f"TTC: {t.ttc_seconds:.1f}s"

                # Draw badge above bbox
                cv2.rectangle(canvas, (x1, max(0, y1 - 42)), (x1 + 175, max(18, y1 - 22)), badge_color, -1)
                cv2.putText(
                    canvas,
                    status_text,
                    (x1 + 4, max(14, y1 - 27)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        return canvas
