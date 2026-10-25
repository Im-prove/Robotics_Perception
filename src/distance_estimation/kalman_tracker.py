"""
Discrete Kalman Filter Tracker for Distance and Approach Velocity (Time-to-Collision).
Provides state estimation, outlier rejection, track life cycle, and smoothing across video frames.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np


@dataclass
class KalmanTrackState:
    """State of an obstacle tracked across video frames."""
    track_id: int
    class_name: str
    distance_m: float           # Filtered distance state (meters)
    velocity_mps: float         # Estimated range rate (m/s, negative = closing in, positive = receding)
    variance_dist: float        # Distance estimation error covariance (m^2)
    variance_vel: float         # Velocity estimation error covariance ((m/s)^2)
    ttc_seconds: Optional[float]# Time-to-collision in seconds (if closing in)
    missed_frames: int = 0      # Count of consecutive missed frames
    total_updates: int = 1      # Total measurement updates received
    last_timestamp: float = 0.0 # Timestamp of last update in seconds


class ObstacleDistanceKalmanFilter:
    """
    1D Constant Velocity Kalman Filter for tracking obstacle distance and range rate.

    State:
        x = [d, v]^T
        where d is distance (m) and v is relative range rate (m/s).

    System Dynamics:
        d_{k} = d_{k-1} + v_{k-1} * dt + 0.5 * dt^2 * w
        v_{k} = v_{k-1} + dt * w
        where w ~ N(0, q) is acceleration process noise.

    State Transition Matrix F:
        F = [[1, dt],
             [0,  1]]

    Process Noise Matrix Q (Continuous White Noise Acceleration):
        Q = q * [[dt^3 / 3, dt^2 / 2],
                 [dt^2 / 2, dt      ]]

    Measurement Matrix H:
        H = [[1, 0]]  (only distance is measured from vision geometry)
        R = [sigma_meas^2]
    """

    def __init__(
        self,
        track_id: int,
        class_name: str,
        initial_distance: float,
        initial_variance: float,
        timestamp: float = 0.0,
        process_noise_accel: float = 1.2
    ):
        self.track_id = track_id
        self.class_name = class_name
        self.q = process_noise_accel # process noise spectral density (m^2/s^3)

        # State vector: [distance, velocity]
        self.x = np.array([[float(initial_distance)], [0.0]], dtype=np.float64)

        # State error covariance P
        # Initial velocity uncertainty is higher (e.g. 4.0 (m/s)^2)
        self.P = np.array([
            [max(0.01, float(initial_variance)), 0.0],
            [0.0, 4.0]
        ], dtype=np.float64)

        self.last_timestamp = float(timestamp)
        self.missed_frames = 0
        self.total_updates = 1

    def predict(self, current_timestamp: float) -> Tuple[float, float]:
        """
        Kalman Filter Prediction Step.
        Returns predicted (distance, velocity).
        """
        dt = current_timestamp - self.last_timestamp
        if dt <= 0.0:
            dt = 0.033 # Default to ~30 FPS if timestamp delta is zero or negative
        dt = float(min(1.0, max(0.001, dt)))

        # State transition
        F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ], dtype=np.float64)

        # Process noise covariance Q
        dt2 = dt * dt
        dt3 = dt2 * dt
        Q = self.q * np.array([
            [dt3 / 3.0, dt2 / 2.0],
            [dt2 / 2.0, dt]
        ], dtype=np.float64)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.last_timestamp = current_timestamp

        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, measured_distance: float, measurement_variance: float) -> Tuple[float, float]:
        """
        Kalman Filter Measurement Update Step.
        Returns updated (distance, velocity).
        """
        z = float(measured_distance)
        R = float(max(1e-4, measurement_variance))

        # Measurement model: H = [1, 0]
        H = np.array([[1.0, 0.0]], dtype=np.float64)

        # Innovation (measurement residual)
        y = z - float(self.x[0, 0])

        # Innovation covariance
        S = float(self.P[0, 0] + R)

        # Mahalanobis gating: reject extreme outliers (> 4-sigma)
        nis = (y * y) / S
        if nis > 16.0:
            # Outlier rejected - update missed frames count and keep predicted state
            self.missed_frames += 1
            return float(self.x[0, 0]), float(self.x[1, 0])

        # Optimal Kalman Gain: K = P * H^T * S^-1
        K = self.P @ H.T / S

        # State update: x = x + K * y
        self.x = self.x + K * y

        # Covariance update (Joseph form for numerical symmetry and positive definiteness):
        # P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        I = np.eye(2, dtype=np.float64)
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K * R @ K.T

        # Clamp distance to physically valid range
        self.x[0, 0] = max(0.1, float(self.x[0, 0]))

        self.missed_frames = 0
        self.total_updates += 1

        return float(self.x[0, 0]), float(self.x[1, 0])

    def get_state(self) -> KalmanTrackState:
        """Return current tracked state object."""
        dist = float(self.x[0, 0])
        vel = float(self.x[1, 0])

        # Time to collision: if closing in (negative range rate)
        # Note: In standard robotics coordinate frame, closing in means range is decreasing (v < 0).
        ttc = None
        if vel < -0.10: # closing speed > 0.1 m/s
            ttc = round(float(-dist / vel), 2)

        return KalmanTrackState(
            track_id=self.track_id,
            class_name=self.class_name,
            distance_m=round(dist, 2),
            velocity_mps=round(vel, 2),
            variance_dist=float(self.P[0, 0]),
            variance_vel=float(self.P[1, 1]),
            ttc_seconds=ttc,
            missed_frames=self.missed_frames,
            total_updates=self.total_updates,
            last_timestamp=self.last_timestamp
        )


class MultiObjectKalmanTracker:
    """Manages active Kalman filters across tracks, handling life cycle and missed frames."""

    def __init__(self, max_missed_frames: int = 15, process_noise_accel: float = 1.2):
        self.max_missed_frames = max_missed_frames
        self.process_noise = process_noise_accel
        self.trackers: Dict[int, ObstacleDistanceKalmanFilter] = {}

    def update_track(
        self,
        track_id: int,
        class_name: str,
        measured_distance: float,
        measurement_variance: float,
        timestamp: float
    ) -> KalmanTrackState:
        """Update or initialize a tracked obstacle with a new distance measurement."""
        if track_id not in self.trackers:
            self.trackers[track_id] = ObstacleDistanceKalmanFilter(
                track_id=track_id,
                class_name=class_name,
                initial_distance=measured_distance,
                initial_variance=measurement_variance,
                timestamp=timestamp,
                process_noise_accel=self.process_noise
            )
            return self.trackers[track_id].get_state()

        kf = self.trackers[track_id]
        kf.predict(timestamp)
        kf.update(measured_distance, measurement_variance)
        return kf.get_state()

    def predict_coasting(self, active_track_ids: List[int], timestamp: float) -> None:
        """Coast tracks that were not detected in current frame."""
        for t_id, kf in list(self.trackers.items()):
            if t_id not in active_track_ids:
                kf.predict(timestamp)
                kf.missed_frames += 1
                if kf.missed_frames > self.max_missed_frames:
                    del self.trackers[t_id]

    def get_track_state(self, track_id: int) -> Optional[KalmanTrackState]:
        """Fetch state for an active track."""
        if track_id in self.trackers:
            return self.trackers[track_id].get_state()
        return None

    def reset(self) -> None:
        """Clear all active tracks."""
        self.trackers.clear()
