"""
Robotic Navigation Object Detection and Distance Estimation CLI.
Annotates objects strictly as requested: 'Cone, 1.5m', 'Stop Sign, 3.2m'.
"""
import argparse
from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.distance_estimation.geometry import load_camera_from_yaml
from src.distance_estimation.fusion_estimator import FusedDistanceEstimator
from src.detector.yolo_detector import YOLODetector
from src.extra_credit.homography_bev import BirdEyeViewMapper
from src.utils.visualization import Visualizer


def run_inference_on_image(
    image_path: str,
    detector: YOLODetector,
    bev_mapper: BirdEyeViewMapper,
    output_dir: Path,
    conf_thresh: float = 0.30
):
    """Process a single image, estimate distances, and create visualization."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Error: Could not load image {image_path}")
        return

    t0 = time.perf_counter()
    # Detect navigation objects and compute distances
    detections = detector.predict(img, conf_thresh=conf_thresh, filter_target_only=False)
    t1 = time.perf_counter()
    fps = 1.0 / max(0.001, (t1 - t0))

    print(f"\n--- Results for {Path(image_path).name} (Latency: {(t1-t0)*1000:.1f}ms | {fps:.1f} FPS) ---")
    if not detections:
        print("  No obstacles detected above threshold.")
    for d in detections:
        dist_str = f"{d.distance_m:.2f}m" if d.distance_m is not None else "N/A"
        offset_x_str = f"{d.offset_x_m:+.2f}m" if d.offset_x_m is not None else "N/A"
        print(f"  -> [{d.class_name}] Conf: {d.confidence:.2f} | Bbox: {[int(v) for v in d.bbox_xyxy]} | "
              f"Distance: {dist_str} (Lateral offset X: {offset_x_str}) | Label: '{d.label_text}'")

    # Generate Bird's Eye View ground map
    bev_img = bev_mapper.warp_image_to_bev(img)
    bev_grid = bev_mapper.render_bev_grid_map(detections, background_bev=bev_img)

    # Render dashboard
    dashboard = Visualizer.render_dashboard(
        main_frame=img,
        detections=detections,
        bev_map=bev_grid,
        fps=fps,
        model_name=Path(detector.weights_path).name
    )

    # Save output image
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"annotated_{Path(image_path).name}"
    cv2.imwrite(str(out_path), dashboard)
    print(f"  [OK] Saved annotated dashboard to: {out_path}")

    # Also save standalone BEV map
    bev_out_path = output_dir / f"bev_{Path(image_path).name}"
    cv2.imwrite(str(bev_out_path), bev_grid)


def run_inference_on_video(
    video_path: str,
    detector: YOLODetector,
    bev_mapper: BirdEyeViewMapper,
    output_dir: Path,
    conf_thresh: float = 0.30
):
    """Process a video file with frame-by-frame distance estimation."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video {video_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = output_dir / f"annotated_{Path(video_path).name}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Dashboard adds 36px top bar
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height + 36))

    frame_idx = 0
    print(f"[*] Processing video: {video_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.predict(frame, conf_thresh=conf_thresh, filter_target_only=False)
        bev_img = bev_mapper.warp_image_to_bev(frame)
        bev_grid = bev_mapper.render_bev_grid_map(detections, background_bev=bev_img)

        dashboard = Visualizer.render_dashboard(
            main_frame=frame,
            detections=detections,
            bev_map=bev_grid,
            fps=fps,
            model_name="YOLO Edge"
        )
        writer.write(dashboard)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[OK] Processed {frame_idx} frames. Saved output video to: {out_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Robotics Object Detection and Distance Estimation")
    parser.add_argument("--input", type=str, default="data/bdd100k_sample/images",
                        help="Path to an input image, folder of images, or video")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Model weights (.pt or .onnx)")
    parser.add_argument("--camera_config", type=str, default="configs/camera_config.yaml", help="Camera calibration YAML")
    parser.add_argument("--output_dir", type=str, default="output/inference", help="Directory to save outputs")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    # Load camera calibration
    intrinsics, extrinsics = load_camera_from_yaml(args.camera_config)
    dist_estimator = FusedDistanceEstimator(intrinsics, extrinsics)
    bev_mapper = BirdEyeViewMapper(intrinsics, extrinsics)

    # Initialize detector
    detector = YOLODetector(
        weights_path=args.weights,
        device=args.device,
        distance_estimator=dist_estimator
    )

    out_dir = Path(args.output_dir)
    inp = Path(args.input)

    if inp.is_file() and inp.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
        run_inference_on_video(str(inp), detector, bev_mapper, out_dir, conf_thresh=args.conf)
    elif inp.is_file():
        run_inference_on_image(str(inp), detector, bev_mapper, out_dir, conf_thresh=args.conf)
    elif inp.is_dir():
        image_files = sorted(list(inp.glob("*.jpg")) + list(inp.glob("*.png")) + list(inp.glob("*.jpeg")))
        print(f"[*] Found {len(image_files)} images in {inp}")
        for img_p in image_files:
            run_inference_on_image(str(img_p), detector, bev_mapper, out_dir, conf_thresh=args.conf)
    else:
        print(f"[!] Invalid input path: {args.input}")


if __name__ == "__main__":
    main()
