# Academic Citations, Industry Standards & Learning Resources

This document compiles the foundational literature, textbook references, industrial robotics standards, and official framework documentation utilized throughout this project.

---

## 1. Classical Computer Vision & Geometric Reference Literature

1. **Hartley, R., & Zisserman, A. (2004).**  
   *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.  
   - **Key Relevance:** Chapter 6 (Camera Models, Intrinsic Matrix $K$, Pinhole Projection); Chapter 9 (Epipolar Geometry and the Fundamental Matrix); Chapter 13 (Direct Linear Transformation and 2D Planar Homographies).  
   - [Cambridge University Press Link](https://www.cambridge.org/core/books/multiple-view-geometry-in-computer-vision/0B6E28164E7651B2241A8275B22A70C0)

2. **Szeliski, R. (2022).**  
   *Computer Vision: Algorithms and Applications* (2nd ed.). Springer.  
   - **Key Relevance:** Section 2.1 (Geometric Image Formation); Section 8.1 (Dense Motion Estimation / Pyramidal Lucas-Kanade Optical Flow); Section 11.1 (Stereo Correspondence and Semi-Global Matching).  
   - [Online Textbook Link (Author Manuscript)](https://szeliski.org/Book/)

3. **OpenCV Official Documentation on Camera Calibration and 3D Reconstruction:**  
   - Camera Intrinsic Matrix, Distortion Models, and Perspective Projections: [OpenCV Camera Calibration Docs](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)  
   - Perspective Warp and Homography Transform: [OpenCV Geometric Image Transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)  
   - Pyramidal Lucas-Kanade Optical Flow: [OpenCV Optical Flow Guide](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)

---

## 2. Estimation Theory, Kalman Filtering & Sensor Fusion

1. **Thrun, S., Burgard, W., & Fox, D. (2005).**  
   *Probabilistic Robotics*. MIT Press.  
   - **Key Relevance:** Chapter 3 (Linear and Extended Kalman Filters); Chapter 6 (Robot Motion Models & Sensor Models); Chapter 7 (Measurement Uncertainty and Multi-Sensor Fusion).  
   - [MIT Press Book Link](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/)

2. **Simon, D. (2006).**  
   *Optimal State Estimation: Kalman, H-infinity, and Nonlinear Approaches*. John Wiley & Sons.  
   - **Key Relevance:** Section 5.1 (Discrete-Time Kalman Filter Derivation); Section 5.4 (Joseph Form Covariance Update for Numerical Stability); Section 5.5 (Continuous White Noise Acceleration Models).  
   - [Wiley Catalog Link](https://www.wiley.com/en-us/Optimal+State+Estimation%3A+Kalman%2C+H+Infinity%2C+and+Nonlinear+Approaches-p-9780471708582)

3. **Gelb, A. (Ed.). (1974).**  
   *Applied Optimal Estimation*. MIT Press.  
   - **Key Relevance:** Foundational treatment of error propagation via first-order Taylor expansion and measurement noise covariance propagation.

---

## 3. Deep Learning & Object Detection Foundations

1. **Jocher, G., Chaurasia, A., & Qiu, J. (2023).**  
   *Ultralytics YOLOv8 Architecture and Documentation*.  
   - **Key Relevance:** Anchor-free decoupled detection heads, Task-Aligned Assigner, Distribution Focal Loss (DFL), Complete IoU (CIoU) loss.  
   - [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)

2. **Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., Madhavan, V., & Darrell, T. (2020).**  
   *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).  
   - **Key Relevance:** Dataset schema, driving scenarios, taxonomy limitations regarding traffic signs, panoptic cone annotations, and camera configurations.  
   - [CVPR Paper (IEEE Xplore)](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.html)  
   - [Official BDD100K Repository](https://github.com/bdd100k/bdd100k)

---

## 4. Edge Hardware Optimization & Quantization

1. **ONNX Runtime Quantization Specification:**  
   - Dynamic vs Static Quantization, Calibration Data, and Tensor Representations:  
     [ONNX Runtime Quantization Documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
2. **PyTorch Model Quantization & Pruning Architecture:**  
   - L1 Unstructured Pruning and Structured Weight Sparsity:  
     [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
3. **NVIDIA TensorRT INT8 Quantization Guide:**  
   - Native INT8 Tensor Core execution on Jetson and embedded GPU architectures:  
     [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)

---

## 5. Physical Metric Standards & Regulatory Specifications

1. **Federal Highway Administration (FHWA) - MUTCD:**  
   *Manual on Uniform Traffic Control Devices for Streets and Highways*.  
   - **Traffic Cones:** Standard 28-inch (0.70 m) height, 15-inch (0.38 m) base width, retroreflective orange and white striping (MUTCD Section 6F.64).  
   - **Stop Signs:** Standard 30-inch (0.75 m) octagonal red plate (R1-1 specification, MUTCD Section 2B.05).  
   - **Barricades:** Type II and Type III channelizing barricades with 0.90 m height (MUTCD Section 6F.68).  
   - [Official FHWA MUTCD Website](https://mutcd.fhwa.dot.gov/)

2. **ROS REP-103 Standard Units of Measure & Coordinate Conventions:**  
   - Defines standard robotics body axes: $+X$ forward, $+Y$ left, $+Z$ up; camera optical frame: $+X$ right, $+Y$ down, $+Z$ forward.  
   - [ROS REP-103 Standard](https://www.ros.org/reps/rep-0103.html)

3. **ISO 3691-4: Industrial Trucks - Safety Requirements and Verification - Part 4: Driverless Industrial Trucks and Their Systems:**  
   - International standard for Autonomous Guided Vehicles (AGVs) and Autonomous Mobile Robots (AMRs) specifying obstacle detection zones and Time-to-Collision braking thresholds.
