"""
Extra Credit Demonstration Suite:
1. Epipolar Geometry: Stereo disparity derivation and metric depth map generation.
2. Homography & Bird's-Eye View: Perspective warp and 2D local navigation grid.
3. Optical Flow: Multi-frame tracking, approach velocity, and Time-to-Collision (TTC) alerts.
"""
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2

from src.distance_estimation.geometry import load_camera_from_yaml
from src.distance_estimation.fusion_estimator import FusedDistanceEstimator
from src.detector.yolo_detector import YOLODetector
from src.extra_credit.epipolar_geometry import StereoDepthEstimator, StereoConfig
from src.extra_credit.homography_bev import BirdEyeViewMapper
from src.extra_credit.optical_flow_tracker import OpticalFlowTracker
from src.utils.visualization import Visualizer


def run_stereo_epipolar_demo(intrinsics, output_dir: Path):
    """Demonstrate Epipolar Geometry and Disparity-to-Depth conversion."""
    print("\n--- [Demo 1/3] Epipolar Geometry & Stereo Depth Estimation ---")
    img_left = cv2.imread("data/bdd100k_sample/images/scene_01_close_cone_stopsign.jpg")
    if img_left is None:
        print("  [!] Error: Sample image not found.")
        return

    stereo_cfg = StereoConfig(
        baseline_m=0.12,
        focal_length_px=intrinsics.fx,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
        num_disparities=64,
        block_size=7
    )
    stereo_est = StereoDepthEstimator(stereo_cfg)

    # Create synthetic depth ground truth (cone at 1.5m, stop sign at 3.2m, background at 15m)
    h, w = img_left.shape[:2]
    synth_depth = np.full((h, w), 15.0, dtype=np.float32)
    # Cone region approx [y: 350-550, x: 500-650]
    synth_depth[350:550, 500:650] = 1.5
    # Stop sign region approx [y: 180-380, x: 750-950]
    synth_depth[180:380, 750:950] = 3.2

    # Synthesize stereo right image from left + depth
    _, img_right = stereo_est.generate_synthetic_stereo_pair(img_left, synth_depth)

    # Compute disparity via StereoSGBM
    t0 = time.perf_counter()
    disparity = stereo_est.compute_disparity(img_left, img_right)
    depth_map = stereo_est.disparity_to_depth(disparity)
    t1 = time.perf_counter()

    color_disp = stereo_est.colorize_disparity(disparity)

    # Estimate depth for cone bounding box
    cone_bbox = (500, 350, 650, 550)
    cone_depth = stereo_est.estimate_bbox_depth(disparity, cone_bbox)
    print(f"  Stereo Disparity Computation Time: {(t1-t0)*1000:.1f}ms")
    print(f"  Formula: Z = (f * B) / d = ({stereo_cfg.focal_length_px} * {stereo_cfg.baseline_m}) / d")
    if cone_depth:
        print(f"  Estimated Stereo Depth for Cone: {cone_depth:.2f}m (True: 1.50m | Error: {abs(cone_depth - 1.5):.2f}m)")

    # Save visual comparison
    stereo_panel = np.hstack([img_left, color_disp])
    cv2.putText(stereo_panel, "Left Rectified Camera Frame", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(stereo_panel, "Stereo Disparity Map (Z = f*B/d)", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    out_p = output_dir / "demo_stereo_epipolar_depth.jpg"
    cv2.imwrite(str(out_p), stereo_panel)
    print(f"  [OK] Saved stereo disparity demo to: {out_p}")


def run_bev_homography_demo(intrinsics, extrinsics, output_dir: Path):
    """Demonstrate Bird's-Eye View Homography Warp and Top-Down Obstacle Map."""
    print("\n--- [Demo 2/3] Homography & Bird's-Eye View (BEV) Transform ---")
    img = cv2.imread("data/bdd100k_sample/images/scene_03_intersection_multi_object.jpg")
    if img is None:
        print("  [!] Error: Sample image not found.")
        return

    bev_mapper = BirdEyeViewMapper(intrinsics, extrinsics)
    dist_est = FusedDistanceEstimator(intrinsics, extrinsics)
    detector = YOLODetector("yolov8n.pt", distance_estimator=dist_est)

    detections = detector.predict(img, conf_thresh=0.25, filter_target_only=False)

    # Warp perspective image to BEV
    t0 = time.perf_counter()
    bev_img = bev_mapper.warp_image_to_bev(img)
    bev_grid = bev_mapper.render_bev_grid_map(detections, background_bev=bev_img)
    t1 = time.perf_counter()

    print(f"  BEV Homography Warp Time: {(t1-t0)*1000:.1f}ms")
    print(f"  Projected {len(detections)} obstacles to 2D top-down ground coordinates.")

    dashboard = Visualizer.render_dashboard(img, detections, bev_map=bev_grid, model_name="BEV Homography")
    out_p = output_dir / "demo_bev_homography_dashboard.jpg"
    cv2.imwrite(str(out_p), dashboard)
    print(f"  [OK] Saved BEV navigation dashboard to: {out_p}")


def run_optical_flow_tracking_demo(output_dir: Path):
    """Demonstrate Lucas-Kanade Optical Flow obstacle tracking and Time-to-Collision (TTC)."""
    print("\n--- [Demo 3/3] Optical Flow Tracking & Time-to-Collision (TTC) ---")
    video_path = "data/sample_video/robot_navigation_moving_cone.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [!] Error: Sample video not found at {video_path}")
        return

    intrinsics, extrinsics = load_camera_from_yaml("configs/camera_config.yaml")
    dist_est = FusedDistanceEstimator(intrinsics, extrinsics)
    detector = YOLODetector("yolov8n.pt", distance_estimator=dist_est)
    tracker = OpticalFlowTracker()

    out_video_path = output_dir / "demo_optical_flow_ttc.mp4"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    frame_idx = 0
    sample_snapshot = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx * (1.0 / fps)
        detections = detector.predict(frame, conf_thresh=0.25, filter_target_only=False)

        # Update optical flow tracker
        active_tracks = tracker.update(frame, detections, timestamp=timestamp)

        # Render visualizations
        annotated = Visualizer.draw_detections(frame, detections)
        overlay = tracker.render_tracking_overlay(annotated, active_tracks)

        writer.write(overlay)

        if frame_idx == 45: # Capture mid-trajectory snapshot
            sample_snapshot = overlay.copy()

        frame_idx += 1

    cap.release()
    writer.release()

    if sample_snapshot is not None:
        snap_path = output_dir / "demo_optical_flow_snapshot.jpg"
        cv2.imwrite(str(snap_path), sample_snapshot)
        print(f"  [OK] Saved optical flow mid-trajectory snapshot to: {snap_path}")

    print(f"  [OK] Processed {frame_idx} frames. Saved optical flow tracking video to: {out_video_path}")


def main():
    output_dir = Path("output/extra_credit")
    output_dir.mkdir(parents=True, exist_ok=True)

    intrinsics, extrinsics = load_camera_from_yaml("configs/camera_config.yaml")

    run_stereo_epipolar_demo(intrinsics, output_dir)
    run_bev_homography_demo(intrinsics, extrinsics, output_dir)
    run_optical_flow_tracking_demo(output_dir)

    print("\n[OK] All Extra Credit Demonstrations Completed Successfully!")


if __name__ == "__main__":
    main()
