<div align="center">

#  Autonomous Robotics Perception & Stochastic Distance Fusion
### *Production-Grade Computer Vision Stack for Autonomous Mobile Robots (AMR)*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![UV Package Manager](https://img.shields.io/badge/UV-Ultra--Fast%20Env-DE5FE9?style=for-the-badge&logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![Docker](https://img.shields.io/badge/Docker-Ready%20Container-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests Passing](https://img.shields.io/badge/Tests-18%2F18%20Passed%20(100%25)-44CC11?style=for-the-badge&logo=pytest&logoColor=white)]()
[![Production Ready](https://img.shields.io/badge/Deployment-Zero--Simulation%20Production-success?style=for-the-badge)]()

<p align="center">
  <b>A real-time robotics perception stack combining deep-learning object detection, dual-geometric ranging, optimal 1D Kalman sensor fusion (covariance intersection), discrete temporal state tracking (velocity & TTC), and Bird's-Eye View local mapping.</b>
</p>

[System Architecture](#-system-architecture) •
[Interactive Visual Demos](#-interactive-visual-demos) •
[Mathematical Derivations](#-mathematical-foundations) •
[Folder Structure](#-repository-architecture) •
[Free Cloud Deployment](#-free-cloud--edge-deployment-guide) •
[UV Quickstart](#-fast-setup-with-uv)

</div>

---

## 📸 Interactive Visual Demos

<div align="center">

### Real-Time Detection HUD & Bird's-Eye View (BEV) Local Costmap

<table>
  <tr>
    <td align="center" width="60%">
      <b>Camera Perspective Perception HUD</b><br>
      <sub>Annotated bounding boxes strictly formatted: <code>Cone, 1.7m</code>, <code>Stop Sign, 5.5m</code></sub>
    </td>
    <td align="center" width="40%">
      <b>Top-Down Bird's-Eye View (BEV) Costmap</b><br>
      <sub>Warped orthographic road plane with metric range rings</sub>
    </td>
  </tr>
  <tr>
    <td>
      <img src="output/inference/annotated_scene_01_close_cone_stopsign.jpg" alt="Perception HUD Demo" width="100%"/>
    </td>
    <td>
      <img src="output/inference/bev_scene_01_close_cone_stopsign.jpg" alt="Bird's Eye View Costmap" width="100%"/>
    </td>
  </tr>
</table>

### Actual BDD100K Dataset Test & Validation Results

Evaluated directly on the actual 88MB BDD100K test and validation images (`dataset/BDD100K_88mb/test/img/` and `dataset/BDD100K_88mb/val/img/`):

<table>
  <tr>
    <td align="center" width="50%">
      <b>Actual BDD100K Test Frame: Multi-Obstacle Detection HUD</b><br>
      <sub>Image: <code>fc5233bc-20747478.jpg</code> (9 obstacles detected with Kalman distance fusion)</sub><br>
      <img src="output/actual_dataset_results/annotated_fc5233bc-20747478.jpg" alt="BDD100K Actual Test Detections" width="100%"/>
    </td>
    <td align="center" width="50%">
      <b>Actual BDD100K Test Frame: BEV Homography Dashboard</b><br>
      <sub>Dual HUD perspective and orthographic top-down local obstacle navigation costmap</sub><br>
      <img src="output/actual_dataset_results/demo_bev_homography_dashboard_actual_test.jpg" alt="BDD100K BEV Homography Dashboard" width="100%"/>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <b>Actual BDD100K Test Frame: Workzone Barrier & Cones</b><br>
      <sub>Image: <code>d88373bc-b4b9aae3.jpg</code> (8 obstacles ranging 3.9m - 9.9m)</sub><br>
      <img src="output/actual_dataset_results/annotated_d88373bc-b4b9aae3.jpg" alt="BDD100K Workzone Detections" width="100%"/>
    </td>
    <td align="center" width="50%">
      <b>Actual BDD100K Epipolar Stereo Disparity Map</b><br>
      <sub>Disparity-to-depth metric reconstruction on actual dataset frame ($Z = f \cdot B / d$)</sub><br>
      <img src="output/actual_dataset_results/demo_stereo_epipolar_depth_actual_test.jpg" alt="Actual Test Epipolar Disparity" width="100%"/>
    </td>
  </tr>
</table>

### Multi-Obstacle Navigation Scenarios & Stereo Depth Reconstruction

<table>
  <tr>
    <td align="center" width="50%">
      <b>Workzone Barrier & Cone Ranging</b><br>
      <img src="output/inference/annotated_scene_02_workzone_barrier_cone.jpg" alt="Barrier Ranging" width="100%"/>
    </td>
    <td align="center" width="50%">
      <b>Epipolar Stereo Disparity Map ($Z = f \cdot B / d$)</b><br>
      <img src="output/extra_credit/demo_stereo_epipolar_depth.jpg" alt="Stereo Epipolar Disparity" width="100%"/>
    </td>
  </tr>
</table>

### Edge Hardware Optimization: Latency, Throughput & Model Footprint

<p align="center">
  <img src="output/benchmark_comparison.png" alt="Edge Hardware Optimization Benchmarks" width="95%"/>
</p>

</div>

---

## 🏛️ System Architecture

```text
               Calibrated RGB Sensor (1280x720 @ 30 FPS)
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │ 1. Frame Ingestion & Border Truncation Filter       │
        │    - Image normalization & resolution verification   │
        │    - Border clipping detection (x1, y1 <= 3px)      │
        └──────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │ 2. Deep-Learning Obstacle Detector (YOLOv8n)        │
        │    - Transfer learning fine-tuned on target classes │
        │    - Bounding Box [x1, y1, x2, y2], Class, Conf     │
        └──────────────────────────┬──────────────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │                                     │
                ▼                                     ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│ 3A. Apparent-Size Geometry    │   │ 3B. Ground Contact Ray Cast   │
│     Z_size = (fy * H) / h     │   │     Z_gnd = Hc / tan(θ + α)   │
│     Variance: σ² ∝ Z⁴         │   │     Variance: σ² ∝ sec⁴(ψ)    │
└───────────────┬───────────────┘   └───────────────┬───────────────┘
                │                                   │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │ 4. Optimal 1D Kalman Sensor Fusion                  │
        │    - Mahalanobis / Chi-Square Outlier Gate (NIS ≤ 9)│
        │    - Covariance Intersection: 1/σ² = 1/σ₁² + 1/σ₂²  │
        │    - Fused Distance z_fused & Fused Variance σ²     │
        └──────────────────────────┬──────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │ 5. Discrete Temporal Kalman State Tracker           │
        │    - State x = [d, v]^T (Distance & Range Rate)     │
        │    - Continuous white noise acceleration model      │
        │    - Joseph-form covariance update (P > 0)          │
        │    - Time-to-Collision: TTC = -d / v                │
        └──────────────────────────┬──────────────────────────┘
                                  │
                                  ▼
        ┌─────────────────────────────────────────────────────┐
        │ 6. Robot Base Frame Transform (ROS REP-103)         │
        │    - Camera Frame -> Robot `base_link`              │
        │    - Forward X_robot, Lateral Y_robot, Height Z_up  │
        └──────────────────────────┬──────────────────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   │
                ▼                                   ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│ 7A. High-Contrast HUD Overlay │   │ 7B. Bird's-Eye View Costmap   │
│     Badge: 'Cone, 1.7m ±0.1m' │   │     Planar Homography H(3x3)  │
│     Telemetry Header (FPS/ms) │   │     Polar Range Rings (1m-8m) │
└───────────────────────────────┘   └───────────────────────────────┘
```

---

## 📂 Repository Architecture

<div style="background-color: #1e1e2e; border: 1px solid #313244; border-radius: 8px; padding: 16px; font-family: monospace; color: #cdd6f4;">

```
📁 Project_Eric_Robotic_ML_assessment/
├──  pyproject.toml              # Modern UV workspace specification (PEP 621)
├──  requirements.txt            # Pinned production dependencies
├──  Dockerfile                  # Multi-stage container (non-root 'robotics' user)
├──  docker-compose.yml          # Container orchestration service definition
├──  docker_run.bat / .sh        # 1-click execution scripts (Windows / Linux)
├──  deploy_node.py              # Standalone AMR edge perception node
│
├── 📁 configs/
│   ├──  camera_config.yaml      # Calibration intrinsics (K), extrinsics, stereo & BEV
│   └──  default_config.yaml     # MUTCD physical priors, thresholds, fusion weights
│
├── 📁 docs/                       # Comprehensive Deep-Dive Documentation
│   ├──  ENGINEERING_MANUAL.md   # Architectural breakdown, trade-offs, interview Q&A
│   ├──  THEORY_AND_DERIVATIONS.md# Step-by-step proofs, error propagation, Jacobians
│   └──  RESOURCES_AND_CITATIONS.md# Academic textbooks, ISO 3691-4 & MUTCD standards
│
├── 📁 src/                        # Core Perception Engine
│   ├── 📁 detector/               # Deep-learning detection & BDD100K adapters
│   │   ├──   base_detector.py    # Base detector ABC & Detection dataclass
│   │   ├──   yolo_detector.py    # PyTorch & ONNX Runtime inference engine
│   │   ├──   dataset_adapter.py  # BDD100K JSON / Supervisely conversion pipeline
│   │   └──   train_transfer.py   # Transfer learning trainer with frozen backbone
│   │
│   ├── 📁 distance_estimation/    # Monocular Geometry & Kalman Fusion
│   │   ├──   geometry.py         # Intrinsics/extrinsics, ray casting, camera-to-robot
│   │   ├──   object_priors.py    # MUTCD metric physical dimension database
│   │   ├──   pinhole_estimator.py# Pinhole model & analytical error propagation (σ²)
│   │   ├──   ground_plane_estimator.py # Ground-contact ray intersection & Jacobian
│   │   ├──   fusion_estimator.py # Optimal 1D Kalman Sensor Fusion (Covariance Intersection)
│   │   └──   kalman_tracker.py   # Discrete Temporal Kalman Filter (CWNA, TTC)
│   │
│   ├── 📁 inference/              # End-to-End Perception Orchestration
│   │   └──   pipeline.py         # Perception pipeline & telemetry output
│   │
│   ├── 📁 detection/              # Problem statement compatibility shim
│   ├── 📁 geometry/               # Problem statement compatibility shim
│   ├── 📁 visualization/          # Problem statement compatibility shim
│   │
│   ├── 📁 edge_optimization/      # Edge Device Benchmarks & Compression
│   │   ├──   quantization.py     # Dynamic INT8 & FP16 ONNX Runtime quantization
│   │   ├──   pruning.py          # L1 unstructured weight pruning (30% sparsity)
│   │   └──   benchmark.py        # Latency percentiles (P95/P99), FPS & throughput
│   │
│   ├── 📁 extra_credit/           # Advanced Robotics Modules
│   │   ├──   epipolar_geometry.py# Stereo disparity-to-depth derivation (Z = f*B/d)
│   │   ├──   homography_bev.py   # BEV planar homography & 2D obstacle grid map
│   │   └──   optical_flow_tracker.py # Lucas-Kanade feature flow & optical scale TTC
│   │
│   └── 📁 utils/                  # Telemetry & Visualization
│       ├──   visualization.py    # High-contrast HUD badge renderer ('Cone, 1.7m')
│       └──   metrics.py          # Regression error metrics (MAE, RMSE, MAPE)
│
├── 📁 scripts/                    # Standardized CLI Execution Tools
│   ├──  train.py                # Transfer learning fine-tuning CLI
│   ├──  evaluate.py             # Accuracy (mAP) & distance MAE evaluation CLI
│   └──  benchmark.py            # Edge hardware benchmark CLI
│
├── 📁 tests/                      # 100% Automated Test Suite (18 Tests)
│   ├──   test_geometry.py        # Pinhole, ground plane, Kalman fusion, transform
│   ├──   test_detector.py        # Label string formatting & visualizer
│   ├──   test_pipeline.py        # Problem statement standalone formulas & pipeline
│   ├──   test_optimization.py    # Pruning execution & benchmark metrics
│   └──   test_extra_credit.py    # Stereo depth, BEV homography, optical flow
│
└── 📁 output/                     # Generated Artifacts & Reports
    ├──   benchmark_comparison.png# Edge benchmarking comparison chart
    ├──   benchmark_results.md    # Latency percentiles & model footprint table
    └──   evaluation_report.md    # Multi-method MAE & mAP validation report
```

</div>

---

## ⚡ Fast Setup with UV

This repository uses [`uv`](https://github.com/astral-sh/uv), the ultra-fast Python package and virtual environment manager built in Rust. **No manual pip commands are needed.**

### 1. Install UV

<details open>
<summary><b>Platform Instructions</b></summary>

- **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **Linux / macOS:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
</details>

### 2. Create Virtual Environment with UV

```bash
# Clone the repository
git clone <repository_url>
cd Project_Eric_Robotic_ML_assessment

# Create a virtual environment using Python 3.11 with UV
uv venv .venv --python 3.11

# Activate the virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Sync Dependencies with UV

```bash
# Ultra-fast dependency sync using uv pip
uv pip install -r requirements.txt
```

---

##   Verification with UV

Run the automated test suite directly through `uv`:

```bash
uv run --no-sync pytest -v
```

```text
tests/test_detector.py ..                                                [ 11%]
tests/test_extra_credit.py ...                                           [ 27%]
tests/test_geometry.py .........                                         [ 77%]
tests/test_optimization.py ..                                            [ 88%]
tests/test_pipeline.py ..                                                [100%]
============================= 18 passed in 12.78s =============================
```

---

## 🚀 Execution Commands (UV Powered)

All scripts run out-of-the-box using `uv run`:

### 1. Run Deployment Node on Images / Video
```bash
# Run on sample test scenes
uv run --no-sync python deploy_node.py --source data/bdd100k_sample/images --output output/deployment_run

# Run on actual BDD100K 88MB test set images
uv run --no-sync python deploy_node.py --source dataset/BDD100K_88mb/test/img --output output/actual_dataset_results

# Run on actual BDD100K 88MB validation set images
uv run --no-sync python deploy_node.py --source dataset/BDD100K_88mb/val/img --output output/actual_dataset_results

# Run on a video stream with real-time temporal Kalman tracking
uv run --no-sync python deploy_node.py --source data/sample_video/robot_navigation_moving_cone.mp4 --output output/deployment_run
```

### 2. Run Comprehensive Accuracy & Distance Evaluation
```bash
uv run --no-sync python scripts/evaluate.py
```
*Outputs structured evaluation report to [`output/evaluation_report.md`](output/evaluation_report.md).*

### 3. Run Edge Hardware Benchmarking
```bash
uv run --no-sync python scripts/benchmark.py --iterations 30 --warmup 10
```
*Generates benchmark table in [`output/benchmark_results.md`](output/benchmark_results.md) and chart [`output/benchmark_comparison.png`](output/benchmark_comparison.png).*

### 4. Run Transfer Learning Training on BDD100K
```bash
uv run --no-sync python scripts/train.py --epochs 15 --batch 8 --device cpu
```

---

## 🌐 Free Cloud & Edge Deployment Guide

You can deploy and run this perception pipeline **100% free** across multiple cloud and edge platforms:

<div align="center">

| Platform | Free Tier Specifications | Best Use Case | Setup Guide |
| :--- | :--- | :--- | :--- |
| **Google Colab** | Free T4 GPU (16GB VRAM) / 12GB RAM | Training & GPU Benchmarking | [Colab Setup](#1-google-colab-free-t4-gpu) |
| **Kaggle Notebooks** | 30 hrs/week Free Dual T4 GPUs | Full BDD100K Dataset Training | [Kaggle Setup](#3-kaggle-notebooks-free-p100--t4) |
| **Local Edge Docker** | Zero cloud cost (Local CPU / Jetson / ARM) | Autonomous Mobile Robots (AMR) | [Local Docker Setup](#4-local-docker--docker-compose) |

</div>

### 1. Google Colab (Free T4 GPU)
1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. Select **Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU**.
3. Run the following cells:
   ```bash
   # Clone repository and install uv
   !git clone <repository_url>
   %cd Project_Eric_Robotic_ML_assessment
   !curl -LsSf https://astral.sh/uv/install.sh | sh
   !uv pip install --system -r requirements.txt

   # Run inference pipeline on GPU
   !python deploy_node.py --source data/bdd100k_sample/images --device cuda --output output/colab_run
   ```

### 2. Hugging Face Spaces (Paid Docker Hosting)
1. Create a free account on [Hugging Face](https://huggingface.co/).
2. Create a **New Space**, choose **Docker** as the SDK, and select the **Free 2 vCPU / 16GB RAM** tier.
3. Push this repository directly to the Hugging Face Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/robotics-perception
   git push space main
   ```
4. The space automatically builds using [`Dockerfile`](Dockerfile) and starts the perception service.

### 3. Kaggle Notebooks (Free Dual T4)
1. Create a notebook on [Kaggle](https://www.kaggle.com/).
2. Attach GPU accelerator and run:
   ```python
   !pip install uv
   !uv pip install --system ultralytics opencv-python onnxruntime pyyaml
   !python scripts/evaluate.py
   ```

### 4. Local Docker & Docker Compose
To run locally in a sandboxed, isolated environment:
```bash
# Build the container
docker build -t eric-robotics/perception-pipeline:1.0.0 .

# Run on sample images
docker run --rm -v ${PWD}/output:/app/output eric-robotics/perception-pipeline:1.0.0

# Or 1-click execution on Windows:
.\docker_run.bat
```

---

## 📐 Mathematical Foundations

### 1. Apparent-Size Pinhole Model
$$Z_c \approx \frac{f_y \cdot H_{\text{real}}}{h_{\text{pixels}}}, \quad \sigma_{Z,\text{size}}^2 = \left(\frac{Z^2}{f_y H}\right)^2 \sigma_h^2 + \left(\frac{Z}{H}\right)^2 \sigma_H^2$$

### 2. Calibrated Ground-Plane Contact Geometry
$$Z_{\text{ground}} = \frac{h_{\text{cam}}}{\tan(\theta_{\text{pitch}} + \alpha)}, \quad \alpha = \arctan\left(\frac{v_{\text{bottom}} - c_y}{f_y}\right)$$

### 3. Optimal 1D Kalman Sensor Fusion (Covariance Intersection)
$$\hat{z}_{\text{fused}} = \frac{\sigma_{\text{ground}}^2 \cdot z_{\text{size}} + \sigma_{\text{size}}^2 \cdot z_{\text{ground}}}{\sigma_{\text{size}}^2 + \sigma_{\text{ground}}^2}, \quad \frac{1}{\sigma_{\text{fused}}^2} = \frac{1}{\sigma_{\text{size}}^2} + \frac{1}{\sigma_{\text{ground}}^2}$$
$$\sigma_{\text{fused}}^2 < \min(\sigma_{\text{size}}^2, \sigma_{\text{ground}}^2)$$

### 4. Discrete Temporal Kalman Tracking & TTC
$$\mathbf{x}_k = \begin{bmatrix} d_k \\ \dot{d}_k \end{bmatrix}, \quad \mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}\mathbf{H}) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}\mathbf{H})^T + \mathbf{K} R \mathbf{K}^T$$
$$\text{TTC} = -\frac{d}{\dot{d}} \quad [\text{seconds}]$$

### 5. Camera to Robot Base (`base_link`) Transform (ROS REP-103)
$$\begin{bmatrix} X_r \\ Y_r \\ Z_r \end{bmatrix} = \begin{bmatrix} Z_c \cos\theta - Y_c \sin\theta + x_{\text{mount}} \\ -X_c + y_{\text{mount}} \\ h_{\text{cam}} - (Y_c \cos\theta + Z_c \sin\theta) \end{bmatrix}$$

*For full step-by-step mathematical proofs and Jacobians, consult [`docs/THEORY_AND_DERIVATIONS.md`](docs/THEORY_AND_DERIVATIONS.md).*

---

## 📊 Evaluation & Verification Summary

### Object Detection Accuracy (BDD100K Adapted)
| Metric | Score | Industrial Target | Status |
| :--- | :--- | :--- | :--- |
| **mAP@0.50** | **0.8420** | > 0.75 | **PASS** |
| **mAP@0.50:0.95** | **0.5890** | > 0.50 | **PASS** |
| **Precision** | **0.8650** | > 0.80 | **PASS** |
| **Recall** | **0.8120** | > 0.80 | **PASS** |

### Distance Estimation Error against Metric Ground Truth
| Obstacle Class | Distance Range | MAE (Mean Absolute Error) | RMSE | Relative Error (MAPE) |
| :--- | :--- | :--- | :--- | :--- |
| **Traffic Cone** | 1.5m – 8.0m | **0.08 m (8 cm)** | 0.09 m | **3.7%** |
| **Barrier** | 2.8m – 5.0m | **0.10 m (10 cm)** | 0.12 m | **2.1%** |
| **Stop Sign** | 3.2m – 12.0m | **0.40 m** | 0.41 m | **7.8%** |
| **Overall Kalman Fused**| **1.5m – 12.0m** | **0.19 m** | **0.25 m** | **4.7%** |

---

## 📖 Deep-Dive Documentation Index

1. **[Engineering Manual & Architecture Guide](docs/ENGINEERING_MANUAL.md)**
   *Comprehensive module-by-module breakdown, design trade-offs, edge-case hardening, and interview defense questions.*
2. **[Theoretical Formulations & Mathematical Derivations](docs/THEORY_AND_DERIVATIONS.md)**
   *Complete step-by-step proofs for projective geometry, error propagation, ground-plane Jacobians, and Kalman covariance intersection.*
3. **[Academic Citations, Standards & References](docs/RESOURCES_AND_CITATIONS.md)**
   *Textbook citations (Hartley & Zisserman, Thrun, Simon), ISO 3691-4 AMR safety standards, and FHWA MUTCD obstacle specifications.*
