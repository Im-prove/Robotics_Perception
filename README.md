# Robotic Object Detection, Distance Estimation & Edge Optimization

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/Package%20Manager-UV-green.svg)](https://github.com/astral-sh/uv)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20ONNX%20Runtime%20%7C%20Ultralytics-orange.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-12%20Passed%20(100%25)-brightgreen.svg)]()

An end-to-end robotics navigation perception pipeline that detects safety-critical navigation objects (**Cones**, **Barriers**, **Stop Signs**), estimates their 3D distance from the robot camera perspective using rigorous geometric formulations, and optimizes model inference for resource-constrained edge hardware.

---

## Table of Contents

- [1. Executive Summary & Architecture](#1-executive-summary--architecture)
- [2. Mathematical Formulations](#2-mathematical-formulations)
  - [A. Known-Object-Dimension Pinhole Model](#a-known-object-dimension-pinhole-model)
  - [B. Calibrated Ground-Plane Contact Geometry](#b-calibrated-ground-plane-contact-geometry)
  - [C. Multi-Hypothesis Sensor Fusion](#c-multi-hypothesis-sensor-fusion)
- [3. Object Detection & Transfer Learning](#3-object-detection--transfer-learning)
- [4. Edge Device Optimization](#4-edge-device-optimization)
  - [Benchmarking Results (Before vs. After)](#benchmarking-results-before-vs-after)
  - [Engineering Analysis](#engineering-analysis)
- [5. Optional Extra Credit Modules](#5-optional-extra-credit-modules)
  - [A. Epipolar Geometry (Stereo Disparity to Depth)](#a-epipolar-geometry-stereo-disparity-to-depth)
  - [B. Homography & Bird's-Eye View (BEV) Ground Projection](#b-homography--birds-eye-view-bev-ground-projection)
  - [C. Optical Flow Tracking & Time-to-Collision (TTC)](#c-optical-flow-tracking--time-to-collision-ttc)
- [6. Project Structure](#6-project-structure)
- [7. Setup & Reproducibility Guide (with `uv`)](#7-setup--reproducibility-guide-with-uv)
- [8. Execution CLI & Scripts](#8-execution-cli--scripts)
- [9. Testing & Verification](#9-testing--verification)
- [10. Submission Information & Author Notes](#10-submission-information--author-notes)

---

## 1. Executive Summary & Architecture

Modern autonomous mobile robots (AMRs), automated guided vehicles (AGVs), and delivery robots require real-time obstacle perception and distance estimation to navigate safely without colliding. While LiDAR is common, camera-only perception provides high-density semantic classification at a fraction of the cost, size, and power budget.

This repository implements a modular, robotics-grade perception engine:

```text
       Camera RGB Frame
             │
             ▼
   ┌───────────────────┐
   │  YOLOv8 Detector  │ ──► [Cone, Barrier, Stop Sign] Bounding Boxes
   └─────────┬─────────┘
             │
             ├──► 1. Pinhole Model (Height Prior H_real) ──────┐
             │                                                 ▼
             ├──► 2. Ground Contact Ray (Height h_cam, Pitch) ─► [Fused 3D Distance]
             │                                                 │ (e.g., 'Cone, 1.5m')
             │                                                 ▼
             ├──► 3. Homography BEV Map ───────────────────────► [2D Navigation Grid]
             │
             └──► 4. Optical Flow Tracker ─────────────────────► [Velocity & TTC Alerts]
```

---

## 2. Mathematical Formulations

### A. Known-Object-Dimension Pinhole Model

Given a pinhole camera with intrinsic calibration matrix:

$$K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

where $f_x, f_y$ are focal lengths in pixels and $(c_x, c_y)$ is the principal point.

By the geometry of similar triangles between the 3D scene and the 2D image plane:

$$\frac{h_{\text{pixels}}}{f_y} = \frac{H_{\text{real}}}{Z}$$

Solving for longitudinal depth $Z$ along the optical axis:

$$Z \approx \frac{f_y \cdot H_{\text{real}}}{h_{\text{pixels}}}$$

Similarly, using known horizontal width $W_{\text{real}}$:

$$Z \approx \frac{f_x \cdot W_{\text{real}}}{w_{\text{pixels}}}$$

Once longitudinal depth $Z$ is computed, the lateral ($X$) and vertical ($Y$) metric coordinates in the camera coordinate frame are:

$$X = \frac{(u_{\text{center}} - c_x) \cdot Z}{f_x}, \quad Y = \frac{(v_{\text{center}} - c_y) \cdot Z}{f_y}$$

The Euclidean 3D distance $D$ from the camera optical center is:

$$D = \sqrt{X^2 + Y^2 + Z^2}$$

#### Calibrated Real-World Metric Priors

| Obstacle Class | Physical Height ($H_{\text{real}}$) | Physical Width ($W_{\text{real}}$) | Standard Reference |
| :--- | :--- | :--- | :--- |
| **Traffic Cone** | **0.70 m** (28 in) | **0.38 m** (15 in) | Standard MUTCD highway cone |
| **Barrier** | **0.90 m** (35 in) | **1.80 m** (71 in) | Type II / Jersey construction barrier |
| **Stop Sign** | **0.75 m** (30 in) | **0.75 m** (30 in) | Standard MUTCD R1-1 octagon |

---

### B. Calibrated Ground-Plane Contact Geometry

For ground-standing obstacles (cones and barriers), the bottom edge of the bounding box represents the physical contact point with the flat ground plane $Y = -h_{\text{cam}}$.

Let $h_{\text{cam}}$ be the camera mounting height above the ground and $\theta_{\text{pitch}}$ be the downward pitch tilt angle. For bottom contact pixel $(u_c, v_{\text{bottom}})$:

1. **Ray Elevation Angle**:

   $$\alpha = \arctan\left(\frac{v_{\text{bottom}} - c_y}{f_y}\right)$$

2. **Total Angle Relative to Ground**:

   $$\psi = \theta_{\text{pitch}} + \alpha$$

3. **Ground Distance**:

   $$Z_{\text{ground}} = \frac{h_{\text{cam}}}{\tan(\psi)} = \frac{h_{\text{cam}}}{\tan\left(\theta_{\text{pitch}} + \arctan\left(\frac{v_{\text{bottom}} - c_y}{f_y}\right)\right)}$$

*Key Advantage*: Invariant to object height truncation or occlusion of the top of the cone!

---

### C. Multi-Hypothesis Sensor Fusion

The system adaptively fuses the pinhole prior distance estimate ($d_p$) and the ground-plane intersection distance estimate ($d_g$):

- **Stop Signs** are elevated signs (mounted on vertical poles) $\rightarrow$ Use pure Pinhole Model.
- **Cones and Barriers** touch the ground $\rightarrow$ Weighted Bayesian fusion:

$$D_{\text{fused}} = \frac{w_p \cdot c_p \cdot d_p + w_g \cdot c_g \cdot d_g}{w_p \cdot c_p + w_g \cdot c_g}$$

where $c_p$ and $c_g$ are geometric confidence scores based on aspect ratio consistency and vertical ground proximity.

---

## 3. Object Detection & Transfer Learning

1. **Pretrained Backbone**: Starts from YOLOv8n (3.15M parameters, 8.7 GFLOPs), optimized for edge efficiency.
2. **Backbone Layer Freezing**: Freezes the first 8 feature-extraction layers to preserve universal edge and texture representations while fine-tuning the neck and detection heads.
3. **BDD100K Adapter**: `BDD100KAdapter` parses standard BDD100K JSON annotations as well as Supervisely/Dataset Ninja formats, extracting `traffic sign`, `stop sign`, `cone`, and `barrier`.
4. **Validation Performance**:
   - **Overall mAP@0.5**: **81.8%**
   - **Overall mAP@0.5:0.95**: **69.7%**
   - **Barrier mAP@0.5**: **99.5%**
   - **Stop Sign mAP@0.5**: **83.5%**
   - **Cone mAP@0.5**: **62.4%**

Annotations are strictly formatted as requested:
```text
Cone, 1.5m
Stop Sign, 3.2m
Barrier, 2.8m
```

---

## 4. Edge Device Optimization

To run efficiently on constrained robotics processors (Raspberry Pi 4/5, NVIDIA Jetson Nano, Intel NUC), we evaluated 4 distinct optimization vectors:

1. **Lightweight Backbone Swapping**: Evaluated next-generation YOLO11n (2.6M params, 6.5 GFLOPs) vs YOLOv8n.
2. **L1 Unstructured Pruning**: Pruned 30% lowest-magnitude weights across 64 Conv2d layers.
3. **ONNX Export**: Cross-platform open graph format for hardware-accelerated runtimes.
4. **Dynamic INT8 Quantization**: Quantized 32-bit floating point weights into 8-bit integers via ONNX Runtime.

### Benchmarking Results (Before vs. After)

Hardware evaluated: **13th Gen Intel Core CPU (x86_64)**:

| Model Variant | Backend | Precision | Device | Mean Latency | P95 Latency | FPS | Model Size | Size Reduction | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv8n Baseline** | PyTorch | FP32 | CPU | 41.19 ms | 49.75 ms | **24.3 FPS** | 6.25 MB | Baseline | **1.00x** |
| **YOLO11n Backbone** | PyTorch | FP32 | CPU | 41.95 ms | 50.87 ms | **23.8 FPS** | 5.35 MB | **14.4% smaller** | **0.98x** |
| **YOLOv8n Pruned (30%)**| PyTorch | Sparsity 30% | CPU | 124.17 ms | 197.24 ms | **8.1 FPS** | 6.25 MB | Sparsity induced | **0.33x** |
| **YOLOv8n ONNX Runtime**| ONNX Runtime | FP32 | CPU | 69.18 ms | 93.21 ms | **14.5 FPS** | 12.23 MB | - | **0.60x** |
| **YOLOv8n Edge Quantized**| ONNX Runtime | **INT8 Dynamic**| CPU | 591.81 ms | 1113.84 ms | **1.7 FPS** | **3.33 MB** | **72.8% smaller** | Storage / RAM |

![Benchmark Comparison Chart](output/benchmark_comparison.png)

### Engineering Analysis

1. **Storage & RAM Footprint**:
   - Dynamic INT8 quantization achieved a **72.8% compression ratio**, shrinking the model from 12.23 MB down to **3.33 MB**. This enables deployment on microcontrollers and embedded devices with strict flash memory constraints (e.g., STM32 / ESP32-S3 AI or Cortex-M).
2. **Latency vs Throughput Dynamics**:
   - On generic desktop CPUs without specialized VNNI hardware extensions, software-level dynamic dequantization introduces per-layer conversion overhead.
   - However, on edge dedicated hardware (such as **NVIDIA Jetson with TensorRT INT8 engines** or **Google Coral EdgeTPU**), INT8 quantization delivers **2x–4x throughput speedup** while cutting memory bandwidth by 4x.
3. **Pruning Insights**:
   - Unstructured L1 pruning zeroes out 30% of weights without requiring retraining. While standard dense CPU BLAS kernels do not accelerate unstructured zeros without sparse BLAS support (e.g., cuSPARSE or DeepSparse), it dramatically improves gzip/tar distribution compression and weight entropy.

---

## 5. Optional Extra Credit Modules

### A. Epipolar Geometry (Stereo Disparity to Depth)

For a rectified stereo camera pair with horizontal baseline $B$ (meters) and focal length $f$ (pixels):

```text
       Left Camera              Right Camera
         (x_L)                     (x_R)
           \                         /
            \       Baseline B      /
             \ <-----------------> /
              \                   /
               \                 /
                \               /
                 ▼             ▼
                   Object Point P (X, Y, Z)
```

1. **Left camera projection**: $x_L = \frac{f \cdot X}{Z} + c_x$
2. **Right camera projection**: $x_R = \frac{f \cdot (X - B)}{Z} + c_x = \frac{f \cdot X}{Z} - \frac{f \cdot B}{Z} + c_x$
3. **Disparity definition**:

   $$d = x_L - x_R = \frac{f \cdot B}{Z}$$

4. **Depth recovery formula**:

   $$Z = \frac{f \cdot B}{d}$$

- `StereoDepthEstimator` utilizes OpenCV's `StereoSGBM` (Semi-Global Block Matching) to generate pixel-level disparity maps and reconstruct metric depth.
- Demo output: `output/extra_credit/demo_stereo_epipolar_depth.jpg`

---

### B. Homography & Bird's-Eye View (BEV) Ground Projection

To plan robot collision-free paths, perspective camera images must be transformed into a metric Bird's-Eye View (top-down ground plane):

$$\begin{bmatrix} u_{\text{bev}} \\ v_{\text{bev}} \\ 1 \end{bmatrix} \sim H_{3 \times 3} \begin{bmatrix} u_{\text{cam}} \\ v_{\text{cam}} \\ 1 \end{bmatrix}$$

`BirdEyeViewMapper`:
- Computes planar homography matrix $H$ using camera intrinsics, height $h_{\text{cam}}$, and pitch tilt $\theta$.
- Warps perspective frames to an orthographic top-down representation.
- Projects all detected obstacles (cones, barriers, stop signs) onto a metric 2D navigation grid with range rings (1m, 2m, 3m, 5m, 8m).
- Demo output: `output/extra_credit/demo_bev_homography_dashboard.jpg`

---

### C. Optical Flow Tracking & Time-to-Collision (TTC)

For dynamic obstacles (e.g., rolling cones, moving barriers):
- `OpticalFlowTracker` utilizes multi-level **Lucas-Kanade pyramidal feature tracking** (`cv2.calcOpticalFlowPyrLK`) inside bounding boxes.
- Estimates 3D approach velocity $v_z = \frac{dZ}{dt}$ (m/s).
- Computes **Time-To-Collision (TTC)**:

  $$\text{TTC} = \frac{Z}{\max(0.1, -v_z)} \quad \text{seconds}$$

- Displays active navigation hazard badges (`CRITICAL TTC < 2.0s`, `CAUTION TTC < 4.0s`).
- Demo output: `output/extra_credit/demo_optical_flow_snapshot.jpg` & `output/extra_credit/demo_optical_flow_ttc.mp4`

---

## 6. Project Structure

```text
Project_Eric_Robotic_ML_assessment/
├── pyproject.toml               # Project metadata & dependencies
├── requirements.txt             # UV-compatible dependency list
├── README.md                    # Detailed documentation and evaluation
├── configs/
│   ├── camera_config.yaml       # Camera intrinsics, extrinsics, stereo & BEV parameters
│   └── default_config.yaml      # Target class metrics, detection thresholds, & fusion weights
├── data/
│   ├── bdd100k_sample/          # Multi-scenario validation images & BDD100K JSON
│   ├── sample_video/            # Navigation video for optical flow tracking
│   └── processed_robotics/      # Fine-tuning YOLO dataset (train/val)
├── dataset/
│   └── BDD100K_88mb/            # BDD100K sample dataset (train/val/ann/img)
├── weights/
│   └── robotics_best.pt         # Fine-tuned transfer learning weights
├── src/
│   ├── detector/
│   │   ├── base_detector.py     # Base abstract detector interface
│   │   ├── yolo_detector.py     # PyTorch & ONNX YOLO inference engine
│   │   ├── dataset_adapter.py   # BDD100K & Supervisely parser & converter
│   │   └── train_transfer.py    # Transfer learning trainer with frozen backbone
│   ├── distance_estimation/
│   │   ├── geometry.py          # Intrinsics/extrinsics, ray casting, 3D projections
│   │   ├── object_priors.py     # Metric dimension database (cones, barriers, stop signs)
│   │   ├── pinhole_estimator.py # Pinhole distance estimation & 3D Euclidean range
│   │   ├── ground_plane_estimator.py # Contact point ray intersection with ground
│   │   └── fusion_estimator.py  # Multi-hypothesis Bayesian sensor fusion
│   ├── edge_optimization/
│   │   ├── quantization.py      # Dynamic INT8 & FP16 ONNX Runtime quantization
│   │   ├── pruning.py           # L1 unstructured & structured channel pruning
│   │   └── benchmark.py         # Automated latency, FPS, and model footprint benchmark
│   ├── extra_credit/
│   │   ├── epipolar_geometry.py # Stereo disparity derivation & depth map generator
│   │   ├── homography_bev.py    # BEV perspective warp & 2D obstacle grid map
│   │   └── optical_flow_tracker.py # Lucas-Kanade tracking, velocity, & TTC alerts
│   └── utils/
│       ├── visualization.py     # Bounding box & distance badge renderer ('Cone, 1.5m')
│       └── metrics.py           # MAE, RMSE, MAPE evaluation metrics
├── experiments/
│   ├── run_inference.py         # Main CLI for image, folder, or video inference
│   ├── run_benchmark.py         # Edge optimization benchmark runner
│   └── run_extra_credit_demo.py # Stereo, BEV, and Optical Flow demo runner
├── scripts/
│   ├── generate_sample_data.py  # Generates metric-accurate test scenes & video
│   └── prepare_and_train.py     # End-to-end dataset generation & fine-tuning
├── tests/
│   ├── test_geometry.py         # Unit tests for camera math & distance formulas
│   ├── test_detector.py         # Unit tests for detector & label formatting
│   ├── test_optimization.py     # Unit tests for quantization & pruning
│   └── test_extra_credit.py     # Unit tests for stereo, BEV, & optical flow
└── output/
    ├── benchmark_results.md     # Markdown benchmark table
    ├── benchmark_comparison.png # Visualization plots
    ├── inference/               # Annotated images & BEV maps
    └── extra_credit/            # Stereo disparity, BEV, and optical flow video
```

---

## 7. Setup & Reproducibility Guide (with `uv`)

This project uses [`uv`](https://github.com/astral-sh/uv), an extremely fast Python package and environment manager.

### 1. Clone the Repository

```bash
git clone <repository_url>
cd Project_Eric_Robotic_ML_assessment
```

### 2. Create and Activate Virtual Environment with `uv`

```bash
# Create a Python 3.11 virtual environment
uv venv .venv --python 3.11

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies with `uv`

```bash
# Install all required packages using uv pip
uv pip install -r requirements.txt
```

---

## 8. Execution CLI & Scripts

### Run Inference on Sample Images

Detects obstacles, estimates 3D distances, renders the BEV map, and annotates bounding boxes strictly as `Cone, 1.5m` or `Stop Sign, 3.2m`:

```bash
# Run on sample image directory
uv run python experiments/run_inference.py --input data/bdd100k_sample/images --weights weights/robotics_best.pt

# Run on a single image
uv run python experiments/run_inference.py --input data/bdd100k_sample/images/scene_01_close_cone_stopsign.jpg --weights weights/robotics_best.pt

# Run on a video file
uv run python experiments/run_inference.py --input data/sample_video/robot_navigation_moving_cone.mp4 --weights weights/robotics_best.pt
```

Outputs are saved to `output/inference/`.

---

### Run Edge Optimization Benchmarks

Benchmarks PyTorch FP32, YOLO11n Lightweight Backbone, 30% Pruned Model, ONNX Runtime, and Dynamic INT8 Quantized Model:

```bash
uv run python experiments/run_benchmark.py
```

Outputs:
- Markdown table: `output/benchmark_results.md`
- Visual plot: `output/benchmark_comparison.png`

---

### Run Extra Credit Demonstrations

Executes Epipolar Stereo Disparity, BEV Homography Grid Map, and Optical Flow Tracking with Time-to-Collision (TTC):

```bash
uv run python experiments/run_extra_credit_demo.py
```

Outputs are saved to `output/extra_credit/`.

---

### Run Transfer Learning Fine-Tuning

To train or re-train the model on target classes with frozen backbone layers:

```bash
uv run python scripts/prepare_and_train.py
```

---

## 9. Testing & Verification

Run the comprehensive unit test suite:

```bash
uv run pytest -v
```

Test coverage includes:
- [x] Pinhole 3D projection and ray casting
- [x] Distance estimation metric accuracy on cones and stop signs
- [x] Ground-plane contact ray intersection math
- [x] Multi-hypothesis sensor fusion
- [x] Exact label string formatting (`Cone, 1.5m`, `Stop Sign, 3.2m`)
- [x] L1 pruning execution and sparsity verification
- [x] Epipolar stereo depth derivation ($Z = \frac{f \cdot B}{d}$)
- [x] Planar homography matrix inversion and projection
- [x] Optical flow tracker velocity and trajectory updates

---

## 10. Submission Information & Author Notes

- **Task**: Object Detection + Distance Estimation for Robotics Navigation
- **Target Classes**: Traffic Cones, Barriers, Stop Signs
- **Distance Models**: Known-Dimension Pinhole Geometry + Ground Plane Intersection + Bayesian Fusion
- **Edge Optimizations**: Dynamic INT8 Quantization (72.8% reduction), L1 Pruning (30% sparsity), YOLO11n Backbone
- **Extra Credit**: Epipolar Stereo Geometry, Homography Bird's Eye View (BEV), Lucas-Kanade Optical Flow with TTC

> **Note on Submission Repository**: As noted in `Problem_Statement_ERIC_Robotics.md`, the assignment instructions mentioned pushing to a personal GitHub repository and also referenced a designated repository. All code is committed to Git version control and is immediately pushable to any remote repository specified.
