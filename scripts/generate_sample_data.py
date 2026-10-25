"""
Generates synthetic robotic navigation scenes with cones, barriers, and stop signs
at mathematically exact distances for testing, distance evaluation, and video tracking.
"""
from pathlib import Path
import json
import math
import numpy as np
import cv2
import yaml


def create_realistic_road_scene(
    width: int = 1280,
    height: int = 720,
    pitch_deg: float = 2.5,
    camera_height: float = 0.65
) -> np.ndarray:
    """Render a clean road scene with asphalt ground, horizon, sky, and lane markings."""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Horizon line calculation based on camera pitch
    fy = 800.0
    cy = 360.0
    pitch_rad = math.radians(pitch_deg)
    horizon_v = int(cy - fy * math.sin(pitch_rad))
    horizon_v = max(50, min(height - 50, horizon_v))

    # Sky gradient
    for y in range(horizon_v):
        ratio = y / horizon_v
        b = int(220 - 40 * ratio)
        g = int(180 - 30 * ratio)
        r = int(140 - 20 * ratio)
        img[y, :] = (b, g, r)

    # Asphalt road
    for y in range(horizon_v, height):
        ratio = (y - horizon_v) / (height - horizon_v)
        b = int(45 + 25 * ratio)
        g = int(45 + 25 * ratio)
        r = int(45 + 25 * ratio)
        img[y, :] = (b, g, r)

    # Perspective lane lines
    pts_left = np.array([[width * 0.45, horizon_v], [width * 0.47, horizon_v], [width * 0.05, height], [width * 0.02, height]], np.int32)
    pts_right = np.array([[width * 0.55, horizon_v], [width * 0.53, horizon_v], [width * 0.95, height], [width * 0.98, height]], np.int32)
    cv2.fillPoly(img, [pts_left], (200, 200, 200))
    cv2.fillPoly(img, [pts_right], (200, 200, 200))

    # Dashed center line
    for y in range(horizon_v + 20, height, 40):
        y2 = min(height, y + 22)
        cx1 = int(width * 0.50 + (y - horizon_v) * 0.0)
        cx2 = int(width * 0.50 + (y2 - horizon_v) * 0.0)
        w1 = max(2, int((y - horizon_v) * 0.015))
        w2 = max(2, int((y2 - horizon_v) * 0.015))
        pts_dash = np.array([[cx1 - w1, y], [cx1 + w1, y], [cx2 + w2, y2], [cx2 - w2, y2]], np.int32)
        cv2.fillPoly(img, [pts_dash], (30, 215, 255))

    return img


def draw_metric_cone(
    img: np.ndarray,
    distance_z: float,
    offset_x: float,
    intrinsics: dict,
    camera_height: float = 0.65,
    pitch_deg: float = 2.5
) -> dict:
    """
    Draw a mathematically sized traffic cone at metric coordinates (X, Z).
    Real cone dimensions: H = 0.70m, W = 0.38m base.
    """
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    real_h = 0.70
    real_w = 0.38

    # Projected pixel dimensions
    h_px = (fy * real_h) / distance_z
    w_px = (fx * real_w) / distance_z

    # Ground contact point
    theta = math.radians(pitch_deg)
    # Contact Y in camera frame = camera_height
    # Accounting for pitch
    y_cam = camera_height * math.cos(theta) - distance_z * math.sin(theta)
    z_cam = camera_height * math.sin(theta) + distance_z * math.cos(theta)

    u_center = int((fx * offset_x / z_cam) + cx)
    v_bottom = int((fy * y_cam / z_cam) + cy)
    v_top = int(v_bottom - h_px)

    x1 = int(u_center - w_px / 2.0)
    x2 = int(u_center + w_px / 2.0)
    y1 = v_top
    y2 = v_bottom

    # Draw orange cone triangle
    pts = np.array([
        [u_center, y1],
        [x1, y2],
        [x2, y2]
    ], np.int32)
    cv2.fillPoly(img, [pts], (0, 140, 255))

    # Base rim
    cv2.ellipse(img, (u_center, y2), (int(w_px * 0.6), int(h_px * 0.1)), 0, 0, 360, (0, 110, 230), -1)

    # Reflective white stripes
    s1_y = int(y1 + 0.35 * h_px)
    s2_y = int(y1 + 0.55 * h_px)
    sw1 = int(w_px * 0.35)
    sw2 = int(w_px * 0.55)
    pts_stripe = np.array([
        [u_center - sw1 // 2, s1_y],
        [u_center + sw1 // 2, s1_y],
        [u_center + sw2 // 2, s2_y],
        [u_center - sw2 // 2, s2_y]
    ], np.int32)
    cv2.fillPoly(img, [pts_stripe], (240, 240, 240))

    return {
        "category": "traffic cone",
        "class_name": "Cone",
        "distance_m": float(distance_z),
        "offset_x_m": float(offset_x),
        "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    }


def draw_metric_barrier(
    img: np.ndarray,
    distance_z: float,
    offset_x: float,
    intrinsics: dict,
    camera_height: float = 0.65,
    pitch_deg: float = 2.5
) -> dict:
    """
    Draw a metric construction barrier at (X, Z).
    Real dimensions: H = 0.90m, W = 1.80m.
    """
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    real_h = 0.90
    real_w = 1.80

    h_px = (fy * real_h) / distance_z
    w_px = (fx * real_w) / distance_z

    theta = math.radians(pitch_deg)
    y_cam = camera_height * math.cos(theta) - distance_z * math.sin(theta)
    z_cam = camera_height * math.sin(theta) + distance_z * math.cos(theta)

    u_center = int((fx * offset_x / z_cam) + cx)
    v_bottom = int((fy * y_cam / z_cam) + cy)
    v_top = int(v_bottom - h_px)

    x1 = int(u_center - w_px / 2.0)
    x2 = int(u_center + w_px / 2.0)
    y1 = v_top
    y2 = v_bottom

    # Main barrier body (red/orange barricade)
    cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 180), -1)

    # Diagonal warning stripes (white & orange)
    stripe_w = max(4, int(w_px * 0.08))
    for sx in range(x1 - int(h_px), x2, stripe_w * 2):
        pts_s = np.array([
            [sx, y2],
            [sx + stripe_w, y2],
            [sx + stripe_w + int(h_px * 0.7), y1],
            [sx + int(h_px * 0.7), y1]
        ], np.int32)
        # Clip to barrier rectangle
        pts_s[:, 0] = np.clip(pts_s[:, 0], x1, x2)
        pts_s[:, 1] = np.clip(pts_s[:, 1], y1, y2)
        cv2.fillPoly(img, [pts_s], (240, 240, 240))

    # Barrier stand legs
    cv2.line(img, (x1 + 10, y2), (x1 + 10, y2 + 10), (50, 50, 50), max(2, int(w_px * 0.02)))
    cv2.line(img, (x2 - 10, y2), (x2 - 10, y2 + 10), (50, 50, 50), max(2, int(w_px * 0.02)))

    return {
        "category": "barrier",
        "class_name": "Barrier",
        "distance_m": float(distance_z),
        "offset_x_m": float(offset_x),
        "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    }


def draw_metric_stop_sign(
    img: np.ndarray,
    distance_z: float,
    offset_x: float,
    intrinsics: dict,
    mounting_height: float = 1.80, # Stop sign center height on pole
    camera_height: float = 0.65,
    pitch_deg: float = 2.5
) -> dict:
    """
    Draw a metric octagonal Stop Sign at (X, Z).
    Real dimensions: H = 0.75m, W = 0.75m.
    """
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    real_d = 0.75 # 0.75m diameter

    d_px = (fy * real_d) / distance_z
    r = int(d_px / 2.0)

    theta = math.radians(pitch_deg)
    # Sign center relative to camera
    y_rel = -(mounting_height - camera_height)
    y_cam = y_rel * math.cos(theta) - distance_z * math.sin(theta)
    z_cam = y_rel * math.sin(theta) + distance_z * math.cos(theta)

    u_center = int((fx * offset_x / z_cam) + cx)
    v_center = int((fy * y_cam / z_cam) + cy)

    # Octagon vertices
    angles = np.linspace(math.pi / 8, 2 * math.pi + math.pi / 8, 9)[:-1]
    pts_oct = []
    for a in angles:
        px = int(u_center + r * math.cos(a))
        py = int(v_center + r * math.sin(a))
        pts_oct.append([px, py])
    pts_oct = np.array(pts_oct, np.int32)

    # Pole down to ground
    y_ground_cam = camera_height * math.cos(theta) - distance_z * math.sin(theta)
    z_ground_cam = camera_height * math.sin(theta) + distance_z * math.cos(theta)
    v_ground = int((fy * y_ground_cam / z_ground_cam) + cy)
    cv2.line(img, (u_center, v_center + r), (u_center, v_ground), (120, 120, 120), max(2, int(r * 0.1)))

    # Red stop sign body
    cv2.fillPoly(img, [pts_oct], (25, 25, 210))
    cv2.polylines(img, [pts_oct], True, (255, 255, 255), max(1, int(r * 0.08)))

    # "STOP" text inside octagon
    font_scale = max(0.2, (d_px / 120.0))
    (tw, th), _ = cv2.getTextSize("STOP", cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, int(font_scale * 2.5)))
    cv2.putText(
        img,
        "STOP",
        (u_center - tw // 2, v_center + th // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        max(1, int(font_scale * 2.5)),
        cv2.LINE_AA
    )

    x1 = u_center - r
    x2 = u_center + r
    y1 = v_center - r
    y2 = v_center + r

    return {
        "category": "traffic sign",
        "sub_category": "stop sign",
        "class_name": "Stop Sign",
        "distance_m": float(distance_z),
        "offset_x_m": float(offset_x),
        "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    }


def generate_all_samples():
    """Generate multi-scenario images, video sequence, and BDD100K JSON annotations."""
    base_dir = Path("data/bdd100k_sample")
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path("data/sample_video")
    video_dir.mkdir(parents=True, exist_ok=True)

    with open("configs/camera_config.yaml", "r") as f:
        cam_cfg = yaml.safe_load(f)["camera"]
    intrinsics = cam_cfg["intrinsics"]
    intrinsics["image_width"] = cam_cfg["image_width"]
    intrinsics["image_height"] = cam_cfg["image_height"]

    samples_meta = []

    # Scenario 1: Close Proximity Navigation (Cone at 1.5m, Stop Sign at 3.2m) - exactly from prompt!
    scene1 = create_realistic_road_scene()
    lbls1 = []
    lbls1.append(draw_metric_cone(scene1, distance_z=1.5, offset_x=-0.45, intrinsics=intrinsics))
    lbls1.append(draw_metric_stop_sign(scene1, distance_z=3.2, offset_x=1.10, intrinsics=intrinsics))
    img1_name = "scene_01_close_cone_stopsign.jpg"
    cv2.imwrite(str(images_dir / img1_name), scene1)
    samples_meta.append({"name": img1_name, "labels": lbls1})

    # Scenario 2: Work Zone Obstacle Avoidance (Barrier at 2.8m, Cone at 4.5m)
    scene2 = create_realistic_road_scene()
    lbls2 = []
    lbls2.append(draw_metric_barrier(scene2, distance_z=2.8, offset_x=0.20, intrinsics=intrinsics))
    lbls2.append(draw_metric_cone(scene2, distance_z=4.5, offset_x=-0.80, intrinsics=intrinsics))
    img2_name = "scene_02_workzone_barrier_cone.jpg"
    cv2.imwrite(str(images_dir / img2_name), scene2)
    samples_meta.append({"name": img2_name, "labels": lbls2})

    # Scenario 3: Complex Multi-Object Intersection (Cone at 2.2m, Barrier at 5.0m, Stop Sign at 7.0m)
    scene3 = create_realistic_road_scene()
    lbls3 = []
    lbls3.append(draw_metric_cone(scene3, distance_z=2.2, offset_x=0.50, intrinsics=intrinsics))
    lbls3.append(draw_metric_barrier(scene3, distance_z=5.0, offset_x=-1.20, intrinsics=intrinsics))
    lbls3.append(draw_metric_stop_sign(scene3, distance_z=7.0, offset_x=1.60, intrinsics=intrinsics))
    img3_name = "scene_03_intersection_multi_object.jpg"
    cv2.imwrite(str(images_dir / img3_name), scene3)
    samples_meta.append({"name": img3_name, "labels": lbls3})

    # Scenario 4: Long-Range Highway Patrol (Cone at 8.0m, Stop Sign at 12.0m)
    scene4 = create_realistic_road_scene()
    lbls4 = []
    lbls4.append(draw_metric_cone(scene4, distance_z=8.0, offset_x=-0.60, intrinsics=intrinsics))
    lbls4.append(draw_metric_stop_sign(scene4, distance_z=12.0, offset_x=1.80, intrinsics=intrinsics))
    img4_name = "scene_04_long_range.jpg"
    cv2.imwrite(str(images_dir / img4_name), scene4)
    samples_meta.append({"name": img4_name, "labels": lbls4})

    # Save BDD100K-compatible JSON annotations
    json_path = base_dir / "bdd100k_sample_labels.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples_meta, f, indent=2)

    print(f"[OK] Generated 4 high-fidelity multi-scenario scenes and BDD100K JSON at {json_path}")

    # Generate Synthetic Video for Optical Flow Tracking (Moving Cone & Approaching Robot)
    video_path = video_dir / "robot_navigation_moving_cone.mp4"
    fps = 30
    duration_sec = 3.0
    num_frames = int(fps * duration_sec)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (1280, 720))

    # Dynamic trajectory: Cone moves laterally while robot approaches (distance closes from 5.0m to 1.8m)
    for i in range(num_frames):
        t = i / float(num_frames)
        dist_z = 5.0 - 3.2 * t                      # Approaching from 5.0m to 1.8m
        offset_x = -1.0 + 1.6 * math.sin(t * math.pi) # Lateral roll across lane

        frame = create_realistic_road_scene()
        draw_metric_cone(frame, distance_z=dist_z, offset_x=offset_x, intrinsics=intrinsics)

        # Distant static stop sign
        draw_metric_stop_sign(frame, distance_z=8.0 - 2.0 * t, offset_x=1.80, intrinsics=intrinsics)

        writer.write(frame)

    writer.release()
    print(f"[OK] Generated synthetic navigation video for Optical Flow at {video_path}")


if __name__ == "__main__":
    generate_all_samples()
