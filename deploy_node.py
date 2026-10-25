#!/usr/bin/env python3
"""
Production Deployment Node for Autonomous Robotics Navigation.
Integrates Object Detection with Multi-Hypothesis Kalman Distance Fusion,
Temporal Tracking across Video Frames, and Bird's Eye View Local Costmap Generation.
"""
import argparse
from pathlib import Path
import sys
import time
import json
import logging
import cv2
import numpy as np

# Ensure root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.pipeline import RoboticsPerceptionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RoboticsDeploymentNode")


def process_image_file(pipeline: RoboticsPerceptionPipeline, img_path: Path, output_dir: Path):
    """Processes a single image file and saves annotated dashboard and telemetry."""
    img = cv2.imread(str(img_path))
    if img is None:
        logger.error(f"Failed to read image: {img_path}")
        return

    telemetry = pipeline.process_frame(img)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save visual dashboard
    out_img_path = output_dir / f"annotated_{img_path.name}"
    cv2.imwrite(str(out_img_path), telemetry.annotated_frame)

    # Save standalone BEV map if available
    if telemetry.bev_grid_map is not None:
        out_bev_path = output_dir / f"bev_{img_path.name}"
        cv2.imwrite(str(out_bev_path), telemetry.bev_grid_map)

    logger.info(
        f"Processed {img_path.name} | Latency: {telemetry.total_pipeline_latency_ms:.1f}ms "
        f"({telemetry.fps:.1f} FPS) | Detections: {len(telemetry.detections)}"
    )
    for det in telemetry.detections:
        dist_str = f"{det.distance_m:.2f}m" if det.distance_m is not None else "N/A"
        var_str = f"+/-{det.std_dev_m:.2f}m" if det.std_dev_m is not None else ""
        rob_str = f"Robot[X={det.robot_x_m:+.2f}m, Y={det.robot_y_m:+.2f}m]" if det.robot_x_m is not None else ""
        logger.info(f"  -> [{det.class_name}] Conf: {det.confidence:.2f} | Dist: {dist_str} {var_str} | {rob_str} | Label: '{det.label_text}'")


def process_video_file(pipeline: RoboticsPerceptionPipeline, video_path: Path, output_dir: Path):
    """Processes a video stream with real-time temporal Kalman tracking."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    output_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = output_dir / f"annotated_{video_path.name}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Dashboard adds 36px top HUD header
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height + 36))

    frame_idx = 0
    t0_stream = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            curr_ts = time.perf_counter() - t0_stream

            telemetry = pipeline.process_frame(frame, timestamp=curr_ts)
            writer.write(telemetry.annotated_frame)

            if frame_idx % 30 == 0:
                logger.info(f"Stream frame {frame_idx} | FPS: {telemetry.fps:.1f} | Active Tracks: {len(telemetry.tracks)}")
    finally:
        cap.release()
        writer.release()
        logger.info(f"Video processing finished. Saved to {out_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Autonomous Robotics Perception Deployment Node")
    parser.add_argument("--source", type=str, default="data/bdd100k_sample/images", help="Path to input image, video, or directory")
    parser.add_argument("--weights", type=str, default="weights/robotics_best.pt", help="Path to detector weights")
    parser.add_argument("--config", type=str, default="configs/camera_config.yaml", help="Camera calibration YAML")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu or cuda)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="output/deployment_run", help="Output directory")
    parser.add_argument("--no-bev", action="store_true", help="Disable Bird's Eye View map")
    args = parser.parse_args()

    logger.info("Initializing Autonomous Robotics Perception Pipeline...")
    pipeline = RoboticsPerceptionPipeline(
        weights_path=args.weights,
        camera_config_path=args.config,
        device=args.device,
        conf_thresh=args.conf,
        enable_bev=not args.no_bev,
        enable_temporal_tracking=True
    )

    src_path = Path(args.source)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if src_path.is_file():
        ext = src_path.suffix.lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            process_image_file(pipeline, src_path, out_dir)
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            process_video_file(pipeline, src_path, out_dir)
        else:
            logger.error(f"Unsupported file format: {ext}")
    elif src_path.is_dir():
        image_files = sorted([p for p in src_path.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        logger.info(f"Found {len(image_files)} images in directory {src_path}")
        for p in image_files:
            process_image_file(pipeline, p, out_dir)
    else:
        logger.error(f"Source not found: {src_path}")


if __name__ == "__main__":
    main()
