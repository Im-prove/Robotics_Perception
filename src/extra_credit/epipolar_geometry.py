"""
Epipolar Geometry and Stereo Disparity-Depth Derivation.

Mathematical Derivation:
-----------------------
Consider a rectified standard stereo camera pair with parallel optical axes separated
by baseline distance B (meters) along the horizontal X-axis, with identical focal length f (pixels).

Let P = (X, Y, Z)^T be a 3D point in the left camera coordinate frame.
1. Projection onto Left Camera Image Plane:
   x_L = f * (X / Z) + c_x
   y_L = f * (Y / Z) + c_y

2. Projection onto Right Camera Image Plane:
   In the right camera coordinate frame, P_R = (X - B, Y, Z)^T.
   x_R = f * ((X - B) / Z) + c_x = (f * X / Z) - (f * B / Z) + c_x
   y_R = f * (Y / Z) + c_y = y_L  (Epipolar lines are collinear horizontal scanlines)

3. Disparity Definition:
   Disparity d is defined as the horizontal pixel difference between corresponding points:
   d = x_L - x_R = (f * X / Z + c_x) - ((f * X / Z) - (f * B / Z) + c_x)
   d = (f * B) / Z

4. Depth Recovery:
   Rearranging for depth Z:
   Z = (f * B) / d

   Lateral and vertical coordinates can then be recovered as:
   X = (x_L - c_x) * Z / f
   Y = (y_L - c_y) * Z / f
"""
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2
from ..distance_estimation.geometry import CameraIntrinsics


@dataclass
class StereoConfig:
    """Stereo camera pair calibration parameters."""
    baseline_m: float = 0.12        # 120 mm baseline typical for compact robotics rigs
    focal_length_px: float = 800.0  # Shared focal length
    cx: float = 640.0
    cy: float = 360.0
    min_disparity: int = 0
    num_disparities: int = 64       # Must be divisible by 16
    block_size: int = 7


class StereoDepthEstimator:
    """Computes dense disparity maps and 3D depth from rectified stereo image pairs."""

    def __init__(self, config: Optional[StereoConfig] = None):
        self.config = config or StereoConfig()
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=self.config.min_disparity,
            numDisparities=self.config.num_disparities,
            blockSize=self.config.block_size,
            P1=8 * 3 * self.config.block_size**2,
            P2=32 * 3 * self.config.block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def compute_disparity(self, img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
        """
        Compute pixel disparity map from left and right rectified images.
        Returns float disparity map in pixels.
        """
        gray_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY) if len(img_left.shape) == 3 else img_left
        gray_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY) if len(img_right.shape) == 3 else img_right

        # StereoSGBM outputs disparity scaled by 16
        disp_raw = self.matcher.compute(gray_l, gray_r).astype(np.float32) / 16.0
        return disp_raw

    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity map to metric depth map (meters).
        Formula: Z = (f * B) / d
        """
        depth = np.zeros_like(disparity, dtype=np.float32)
        valid_mask = disparity > 0.5

        f = self.config.focal_length_px
        b = self.config.baseline_m
        depth[valid_mask] = (f * b) / disparity[valid_mask]
        depth[~valid_mask] = np.nan
        return depth

    def estimate_bbox_depth(
        self,
        disparity: np.ndarray,
        bbox_xyxy: Tuple[float, float, float, float]
    ) -> Optional[float]:
        """
        Estimate robust median depth of an object bounded by [x1, y1, x2, y2].
        """
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        h, w = disparity.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Sample central 50% ROI of bounding box to minimize background boundary contamination
        rw = x2 - x1
        rh = y2 - y1
        roi_x1 = int(x1 + 0.25 * rw)
        roi_x2 = int(x2 - 0.25 * rw)
        roi_y1 = int(y1 + 0.25 * rh)
        roi_y2 = int(y2 - 0.25 * rh)

        roi_disp = disparity[roi_y1:roi_y2, roi_x1:roi_x2]
        valid_disp = roi_disp[roi_disp > 1.0]

        if len(valid_disp) < 10:
            return None

        median_disp = float(np.median(valid_disp))
        depth = (self.config.focal_length_px * self.config.baseline_m) / median_disp
        return float(depth)

    def colorize_disparity(self, disparity: np.ndarray) -> np.ndarray:
        """Colorize disparity map for visualization."""
        valid_mask = disparity > 0.5
        norm_disp = np.zeros_like(disparity, dtype=np.uint8)
        if np.any(valid_mask):
            d_min = np.percentile(disparity[valid_mask], 5)
            d_max = np.percentile(disparity[valid_mask], 95)
            clipped = np.clip(disparity, d_min, d_max)
            norm_disp = ((clipped - d_min) / (d_max - d_min + 1e-6) * 255.0).astype(np.uint8)

        color_disp = cv2.applyColorMap(norm_disp, cv2.COLORMAP_TURBO)
        color_disp[~valid_mask] = 0
        return color_disp

    def generate_synthetic_stereo_pair(
        self,
        image_left: np.ndarray,
        synth_depth_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a synthetic right camera image from left image and depth map.
        Useful for testing disparity algorithms when only monocular data is available.
        """
        h, w = image_left.shape[:2]
        f = self.config.focal_length_px
        b = self.config.baseline_m

        # Compute theoretical disparity: d = f * B / Z
        disparity = np.zeros_like(synth_depth_map)
        valid = synth_depth_map > 0.1
        disparity[valid] = (f * b) / synth_depth_map[valid]

        # Shift pixels horizontally: x_R = x_L - d
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + disparity).astype(np.float32)
        map_y = grid_y.astype(np.float32)

        image_right = cv2.remap(image_left, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return image_left, image_right
