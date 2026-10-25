"""
Camera Intrinsics and Extrinsics Geometric Representations.
"""
from dataclasses import dataclass, field
import numpy as np
import yaml
from typing import Tuple, Optional


@dataclass
class CameraIntrinsics:
    """Represents a pinhole camera intrinsic matrix and sensor specifications."""
    fx: float = 800.0
    fy: float = 800.0
    cx: float = 640.0
    cy: float = 360.0
    image_width: int = 1280
    image_height: int = 720
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    @property
    def matrix(self) -> np.ndarray:
        """Return 3x3 intrinsic matrix K."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @property
    def hfov_deg(self) -> float:
        """Horizontal Field of View in degrees."""
        return float(2 * np.degrees(np.arctan(self.image_width / (2.0 * self.fx))))

    @property
    def vfov_deg(self) -> float:
        """Vertical Field of View in degrees."""
        return float(2 * np.degrees(np.arctan(self.image_height / (2.0 * self.fy))))

    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """Convert 2D pixel coordinate (u, v) into a normalized 3D unit ray in camera frame."""
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        z = 1.0
        ray = np.array([x, y, z], dtype=np.float64)
        return ray / np.linalg.norm(ray)

    def project_3d_to_pixel(self, x: float, y: float, z: float) -> Tuple[float, float]:
        """Project 3D point (X, Y, Z) in camera frame to 2D pixel (u, v)."""
        if z <= 1e-4:
            raise ValueError(f"Point is behind or on the camera optical center (Z={z} <= 0)")
        u = (self.fx * x / z) + self.cx
        v = (self.fy * y / z) + self.cy
        return float(u), float(v)


@dataclass
class CameraExtrinsics:
    """Represents camera mounting position and orientation relative to robot base_link."""
    height_m: float = 0.65       # Camera height above ground
    pitch_deg: float = 2.5       # Pitch angle (+ downwards toward ground)
    roll_deg: float = 0.0        # Roll angle
    yaw_deg: float = 0.0         # Yaw angle relative to robot forward heading
    x_offset_m: float = 0.15     # Forward offset of camera from robot base center (meters)
    y_offset_m: float = 0.0      # Lateral offset of camera from robot base center (meters)

    @property
    def pitch_rad(self) -> float:
        return float(np.radians(self.pitch_deg))

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Rotation matrix from camera frame to robot base frame."""
        pitch = self.pitch_rad
        roll = np.radians(self.roll_deg)
        yaw = np.radians(self.yaw_deg)

        # Pitch rotation around X axis (camera frame: X right, Y down, Z forward)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx

    def transform_camera_to_robot(self, x_cam: float, y_cam: float, z_cam: float) -> Tuple[float, float, float]:
        """
        Transform 3D point from Camera Optical Frame to Robot Base Frame (ROS REP-103 convention):
        - Camera Optical Frame: +X right, +Y down, +Z forward (along optical axis).
        - Robot Base Frame (base_link): +X forward, +Y left, +Z up (from ground footprint).
        """
        # Account for downward camera pitch:
        theta = self.pitch_rad
        # Forward in world = Z_cam * cos(theta) - Y_cam * sin(theta) + x_offset
        x_robot = z_cam * np.cos(theta) - y_cam * np.sin(theta) + self.x_offset_m
        # Lateral left in robot = -X_cam + y_offset
        y_robot = -x_cam + self.y_offset_m
        # Height above ground in robot = height_m - (Y_cam * cos(theta) + Z_cam * sin(theta))
        z_robot = self.height_m - (y_cam * np.cos(theta) + z_cam * np.sin(theta))

        return float(x_robot), float(y_robot), float(z_robot)

    def transform_robot_to_camera(self, x_robot: float, y_robot: float, z_robot: float) -> Tuple[float, float, float]:
        """Transform 3D point from Robot Base Frame to Camera Optical Frame."""
        theta = self.pitch_rad
        dx = x_robot - self.x_offset_m
        dy = y_robot - self.y_offset_m
        dz = z_robot - self.height_m

        x_cam = -dy
        y_cam = -dz * np.cos(theta) - dx * np.sin(theta)
        z_cam = dx * np.cos(theta) - dz * np.sin(theta)

        return float(x_cam), float(y_cam), float(z_cam)


def load_camera_from_yaml(config_path: str) -> Tuple[CameraIntrinsics, CameraExtrinsics]:
    """Load camera intrinsics and extrinsics from a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)["camera"]

    intr = data["intrinsics"]
    extr = data["extrinsics"]

    intrinsics = CameraIntrinsics(
        fx=float(intr.get("fx", 800.0)),
        fy=float(intr.get("fy", 800.0)),
        cx=float(intr.get("cx", data.get("image_width", 1280) / 2.0)),
        cy=float(intr.get("cy", data.get("image_height", 720) / 2.0)),
        image_width=int(data.get("image_width", 1280)),
        image_height=int(data.get("image_height", 720)),
        k1=float(intr.get("k1", 0.0)),
        k2=float(intr.get("k2", 0.0)),
        p1=float(intr.get("p1", 0.0)),
        p2=float(intr.get("p2", 0.0)),
    )

    extrinsics = CameraExtrinsics(
        height_m=float(extr.get("height_m", 0.65)),
        pitch_deg=float(extr.get("pitch_deg", 2.5)),
        roll_deg=float(extr.get("roll_deg", 0.0)),
        yaw_deg=float(extr.get("yaw_deg", 0.0)),
        x_offset_m=float(extr.get("x_offset_m", 0.15)),
        y_offset_m=float(extr.get("y_offset_m", 0.0)),
    )

    return intrinsics, extrinsics
