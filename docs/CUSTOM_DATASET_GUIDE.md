# Custom Dataset Onboarding & Integration Guide

This guide details how to integrate, adapt, annotate, calibrate, and fine-tune external datasets (e.g., Roboflow, custom AMR video logs, COCO, KITTI, Cityscapes) into this perception pipeline.

---

## Table of Contents

1. [Supported Dataset Formats](#1-supported-dataset-formats)
2. [Method A: Standard YOLO Annotation Format (Recommended)](#2-method-a-standard-yolo-annotation-format-recommended)
3. [Method B: BDD100K / COCO JSON Format](#3-method-b-bdd100k--coco-json-format)
4. [Method C: Raw Unlabeled Camera Feeds & Video Logs](#4-method-c-raw-unlabeled-camera-feeds--video-logs)
5. [Registering New Obstacle Classes & Physical Priors](#5-registering-new-obstacle-classes--physical-priors)
6. [Camera Calibration for External Rigs](#6-camera-calibration-for-external-rigs)
7. [End-to-End Execution Checklist](#7-end-to-end-execution-checklist)

---

## 1. Supported Dataset Formats

| Format | Source Examples | Adaptation Mechanism |
| :--- | :--- | :--- |
| **YOLO TXT** | Roboflow, Label Studio, CVAT, Ultralytics HUB | Direct ingestion via `data.yaml` |
| **BDD100K JSON** | Official BDD100K, Dataset Ninja | `BDD100KAdapter.convert_bdd100k_json()` |
| **Supervisely JSON**| BDD100K extracted archives (`ann/` & `img/`) | `BDD100KAdapter.convert_dataset_ninja_bdd100k()` |
| **COCO JSON** | MS COCO, OpenImages | Automated keypoint/bbox parser in `dataset_adapter.py` |
| **Raw RGB Images / Video**| USB Cam, RealSense, AMR MP4 stream | `deploy_node.py` direct zero-shot / fine-tuned inference |

---

## 2. Method A: Standard YOLO Annotation Format (Recommended)

When downloading or exporting datasets from annotation tools (e.g., **Roboflow**, **Label Studio**, **CVAT**):

### 2.1 Directory Layout

```text
custom_dataset/
├── images/
│   ├── train/     # e.g., frame_0001.jpg, frame_0002.jpg
│   └── val/       # e.g., val_0001.jpg, val_0002.jpg
├── labels/
│   ├── train/     # e.g., frame_0001.txt, frame_0002.txt
│   └── val/       # e.g., val_0001.txt, val_0002.txt
└── data.yaml
```

Each label `.txt` file contains normalized bounding box rows:
```text
class_id x_center y_center width height
0 0.4523 0.6210 0.0840 0.1950
```

### 2.2 Configure `custom_dataset/data.yaml`

```yaml
path: custom_dataset
train: images/train
val: images/val
test: images/val # optional

names:
  0: Cone
  1: Barrier
  2: Stop Sign
nc: 3
```

### 2.3 Run Transfer Learning with `uv`

```bash
uv run --no-sync python scripts/train.py \
  --data custom_dataset/data.yaml \
  --model yolov8n.pt \
  --epochs 20 \
  --batch 16 \
  --device cpu
```

> **Transfer Learning Rationale:** The trainer automatically freezes the first 10 backbone layers to preserve general feature extraction, fine-tuning only the detection head to your custom dataset.

---

## 3. Method B: BDD100K / COCO JSON Format

If your annotations are stored in hierarchical JSON files:

```python
from src.detector.dataset_adapter import BDD100KAdapter

# 1. Instantiate adapter targeting destination output folder
adapter = BDD100KAdapter(output_dir="data/my_custom_yolo")

# 2. Convert Train Split
adapter.convert_bdd100k_json(
    json_path="path/to/train_labels.json",
    image_src_dir="path/to/train_images",
    split="train",
    max_samples=500
)

# 3. Convert Validation Split
adapter.convert_bdd100k_json(
    json_path="path/to/val_labels.json",
    image_src_dir="path/to/val_images",
    split="val",
    max_samples=100
)

# 4. Generate data.yaml
yaml_path = adapter.generate_yolo_yaml()
print(f"Dataset successfully compiled at: {yaml_path}")
```

Once converted, trigger fine-tuning:
```bash
uv run --no-sync python scripts/train.py --data data/my_custom_yolo/data.yaml
```

---

## 4. Method C: Raw Unlabeled Camera Feeds & Video Logs

To run the complete perception stack (YOLO detection + 1D Kalman sensor fusion + temporal velocity tracking + Bird's-Eye View mapping) on raw AMR footage without retraining:

### 4.1 Process an Image Directory
```bash
uv run --no-sync python deploy_node.py \
  --source path/to/my_robot_images/ \
  --weights weights/robotics_best.pt \
  --config configs/camera_config.yaml \
  --output output/custom_run
```

### 4.2 Process a Video Stream
```bash
uv run --no-sync python deploy_node.py \
  --source path/to/amr_navigation_log.mp4 \
  --weights weights/robotics_best.pt \
  --config configs/camera_config.yaml \
  --output output/custom_video_run
```

Output includes:
- `annotated_*.jpg`: High-contrast HUD badges (`Cone, 1.8m`, `Stop Sign, 4.2m`).
- `bev_*.jpg`: Orthographic top-down local obstacle grid map with metric range rings.

---

## 5. Registering New Obstacle Classes & Physical Priors

If your new dataset contains obstacle categories beyond `Cone`, `Barrier`, and `Stop Sign` (e.g., pedestrians, delivery carts, vehicles), register their physical metric dimensions in [`src/distance_estimation/object_priors.py`](../src/distance_estimation/object_priors.py):

```python
from src.distance_estimation.object_priors import ObjectPriors, ObjectDimension

# Register a pedestrian prior (MUTCD 50th percentile adult)
ObjectPriors.register("person", ObjectDimension(
    height_m=1.70,        # Mean adult pedestrian height
    width_m=0.50,         # Shoulder breadth
    depth_m=0.30,
    min_dist_m=0.3,
    max_dist_m=35.0
))

# Register an automated delivery cart / robot
ObjectPriors.register("robot", ObjectDimension(
    height_m=0.60,
    width_m=0.45,
    depth_m=0.70,
    min_dist_m=0.2,
    max_dist_m=20.0
))
```

> **Why this matters:** The `PinholeDistanceEstimator` and `KalmanDistanceFusion` engines dynamically look up these metric dimensions to compute optical depth $Z = \frac{f_y \cdot H}{h}$ and analytical measurement variance $\sigma^2$.

---

## 6. Camera Calibration for External Rigs

Monocular metric distance estimation depends on calibrated camera intrinsics and vehicle mounting extrinsics. When deploying on a new robot:

### 6.1 Update [`configs/camera_config.yaml`](../configs/camera_config.yaml)

```yaml
camera:
  name: "My_AMR_Front_Camera"
  image_width: 1920      # Camera horizontal resolution
  image_height: 1080     # Camera vertical resolution

  intrinsics:
    fx: 1250.0           # Horizontal focal length in pixels (from checkerboard calibration)
    fy: 1250.0           # Vertical focal length in pixels
    cx: 960.0            # Principal point X (image_width / 2)
    cy: 540.0            # Principal point Y (image_height / 2)

  extrinsics:
    height_m: 0.55       # Camera optical center height above ground (meters)
    pitch_deg: 3.0       # Downward mounting pitch tilt (degrees)
    x_offset_m: 0.20     # Forward mounting offset ahead of robot center (meters)
    y_offset_m: 0.0      # Lateral offset from center line (meters)
```

### 6.2 Quick Calibration Helper (One-Shot Method)

If formal checkerboard calibration is unavailable, calibrate focal length $F$ from a single reference image:
$$F = \frac{h_{\text{pixels}} \cdot D_{\text{known}}}{H_{\text{known}}}$$

```python
from src.geometry.distance import PinholeDistanceEstimator

# Suppose an object of known height 0.70m is placed at exactly 2.0m distance
# and occupies 280 pixels vertically in the camera frame:
focal_px = (280.0 * 2.0) / 0.70  # = 800.0 pixels
print(f"Calculated focal length: {focal_px:.1f} pixels")
```

---

## 7. End-to-End Execution Checklist

- [x] Dataset arranged with `images/` and `labels/` or converted via `BDD100KAdapter`.
- [x] `data.yaml` maps class IDs consistently (`0: Cone`, `1: Barrier`, `2: Stop Sign`).
- [x] Camera calibration parameters verified in `configs/camera_config.yaml`.
- [x] Target class physical heights registered in `object_priors.py`.
- [x] Fine-tuning executed via `uv run --no-sync python scripts/train.py --data <data.yaml>`.
- [x] Validation verified via `uv run --no-sync python scripts/evaluate.py`.
- [x] Real-time inference deployed via `uv run --no-sync python deploy_node.py`.
