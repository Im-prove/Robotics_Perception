"""
Homography and Bird's-Eye View (BEV) Perspective Transform.
Warps perspective camera views to an orthographic top-down ground plane
and maps detected obstacles to a 2D local navigation grid.
"""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2
from ..distance_estimation.geometry import CameraIntrinsics, CameraExtrinsics
from ..detector.base_detector import Detection


@dataclass
class BEVConfig:
    """Bird's Eye View transformation configuration."""
    x_range_m: Tuple[float, float] = (-3.0, 3.0)  # Lateral range: 3m left to 3m right
    z_range_m: Tuple[float, float] = (0.5, 12.0)   # Longitudinal range: 0.5m to 12m forward
    output_size: Tuple[int, int] = (600, 600)      # (height, width) of output BEV image


class BirdEyeViewMapper:
    """Computes homography perspective warp and creates top-down local obstacle maps."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        bev_config: Optional[BEVConfig] = None
    ):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.config = bev_config or BEVConfig()
        self.homography_matrix = self._compute_homography_matrix()
        self.inv_homography = np.linalg.inv(self.homography_matrix)

    def _compute_homography_matrix(self) -> np.ndarray:
        """
        Derive the 3x3 Homography Matrix mapping camera image pixels (u, v)
        to BEV raster pixels (u_bev, v_bev).
        """
        x_min, x_max = self.config.x_range_m
        z_min, z_max = self.config.z_range_m
        h_bev, w_bev = self.config.output_size

        # Define 4 metric points on the flat ground plane (in meters): [X, Y=h_cam, Z]
        src_metric_ground = np.array([
            [x_min * 0.5, z_min],  # Near left
            [x_max * 0.5, z_min],  # Near right
            [x_max, z_max],        # Far right
            [x_min, z_max],        # Far left
        ], dtype=np.float32)

        # Project 3D ground points to 2D camera image pixels (u, v)
        cam_pixels = []
        for X, Z in src_metric_ground:
            # Account for camera pitch
            theta = self.extrinsics.pitch_rad
            h = self.extrinsics.height_m

            # Rotate around X-axis by pitch angle
            Y_cam = h * np.cos(theta) - Z * np.sin(theta)
            Z_cam = h * np.sin(theta) + Z * np.cos(theta)

            u = (self.intrinsics.fx * X / Z_cam) + self.intrinsics.cx
            v = (self.intrinsics.fy * Y_cam / Z_cam) + self.intrinsics.cy
            cam_pixels.append([u, v])

        cam_pixels = np.array(cam_pixels, dtype=np.float32)

        # Destination pixels in the BEV image
        bev_pixels = np.array([
            [self.metric_to_bev_pixel(x_min * 0.5, z_min)],
            [self.metric_to_bev_pixel(x_max * 0.5, z_min)],
            [self.metric_to_bev_pixel(x_max, z_max)],
            [self.metric_to_bev_pixel(x_min, z_max)],
        ], dtype=np.float32).reshape(-1, 2)

        H, _ = cv2.findHomography(cam_pixels, bev_pixels)
        return H

    def metric_to_bev_pixel(self, x_m: float, z_m: float) -> Tuple[int, int]:
        """Convert metric ground coordinates (X meters right, Z meters forward) to BEV pixel (u, v)."""
        x_min, x_max = self.config.x_range_m
        z_min, z_max = self.config.z_range_m
        h_bev, w_bev = self.config.output_size

        u = int((x_m - x_min) / (x_max - x_min) * (w_bev - 1))
        # Z increases away from robot -> displayed towards top of BEV image (smaller v)
        v = int((1.0 - (z_m - z_min) / (z_max - z_min)) * (h_bev - 1))
        return u, v

    def warp_image_to_bev(self, image: np.ndarray) -> np.ndarray:
        """Warp entire perspective camera frame to Bird's Eye View."""
        h_bev, w_bev = self.config.output_size
        bev_img = cv2.warpPerspective(
            image,
            self.homography_matrix,
            (w_bev, h_bev),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(20, 20, 20)
        )
        return bev_img

    def render_bev_grid_map(
        self,
        detections: List[Detection],
        background_bev: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Render a top-down metric navigation grid with obstacle markers and distance rings.
        """
        h_bev, w_bev = self.config.output_size

        if background_bev is not None:
            canvas = background_bev.copy()
            # Overlay semi-transparent dark tint for grid clarity
            overlay = np.full_like(canvas, (25, 25, 25))
            canvas = cv2.addWeighted(canvas, 0.45, overlay, 0.55, 0)
        else:
            canvas = np.full((h_bev, w_bev, 3), (25, 25, 25), dtype=np.uint8)

        # Draw distance range rings (1m, 2m, 3m, 5m, 8m, 10m)
        center_u, center_v = self.metric_to_bev_pixel(0.0, 0.0)
        for r_m in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
            _, ring_v = self.metric_to_bev_pixel(0.0, r_m)
            radius_px = abs(center_v - ring_v)
            cv2.circle(canvas, (center_u, center_v), radius_px, (60, 60, 60), 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"{r_m:.0f}m",
                (center_u + 6, ring_v + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (120, 120, 120),
                1,
                cv2.LINE_AA
            )

        # Draw longitudinal center line (robot heading)
        cv2.line(canvas, (center_u, 0), (center_u, h_bev), (40, 80, 40), 1, cv2.LINE_AA)

        # Draw robot footprint icon at (0, 0)
        bot_u, bot_v = self.metric_to_bev_pixel(0.0, 0.2)
        cv2.rectangle(canvas, (bot_u - 15, bot_v - 20), (bot_u + 15, bot_v + 10), (0, 200, 100), -1)
        cv2.arrowedLine(canvas, (bot_u, bot_v + 5), (bot_u, bot_v - 25), (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)
        cv2.putText(canvas, "ROBOT", (bot_u - 18, bot_v + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 120), 1)

        # Color mapping for obstacles
        color_map = {
            "cone": (0, 165, 255),       # Orange
            "barrier": (0, 0, 255),       # Red
            "stop sign": (30, 30, 220),   # Crimson
        }

        # Project detected obstacles onto BEV
        for det in detections:
            if det.distance_m is None:
                continue

            # If 3D coordinates available, use (offset_x, distance_z)
            x_m = det.offset_x_m if det.offset_x_m is not None else 0.0
            z_m = det.distance_z_m if det.distance_z_m is not None else det.distance_m

            u_p, v_p = self.metric_to_bev_pixel(x_m, z_m)

            # Check bounds
            if 0 <= u_p < w_bev and 0 <= v_p < h_bev:
                cls_norm = det.class_name.lower()
                c = color_map.get(cls_norm, (200, 200, 0))

                # Draw obstacle marker
                cv2.circle(canvas, (u_p, v_p), 9, c, -1, cv2.LINE_AA)
                cv2.circle(canvas, (u_p, v_p), 11, (255, 255, 255), 1, cv2.LINE_AA)

                # Distance text label
                label = f"{det.class_name}: {det.distance_m:.1f}m"
                cv2.putText(canvas, label, (u_p + 12, v_p + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        return canvas
