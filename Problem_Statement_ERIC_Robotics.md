# Object Detection and Distance Estimation for Robotics Navigation

## 1. Problem Statement

Develop a computer-vision pipeline for robotics navigation that detects navigation-relevant objects in camera images, estimates their distance from the camera, and annotates the results. The implementation should also explore ways to improve inference efficiency on resource-constrained or edge devices.

The target object classes are:

- Traffic cones
- Barriers
- Stop signs

Example annotations:

```text
Cone, 1.5 m
Stop Sign, 3.2 m
```

*The distances above are illustrative examples, not measured results.*

## 2. Objectives

1. **Object detection:** Fine-tune a suitable pretrained model using transfer learning to detect the target classes.
2. **Distance estimation:** Estimate the camera-relative distance to each detection using geometric information, such as camera focal length, pixel measurements, known object dimensions, or calibrated ground-plane geometry.
3. **Edge optimization:** Evaluate at least one practical approach to improve model efficiency, such as a lightweight model, quantization, or pruning.
4. **Performance evaluation:** Measure CPU and GPU inference performance where hardware is available, and report detection accuracy and distance-estimation error where possible.
5. **Clean implementation:** Organize the code into understandable, reusable modules and document the setup and evaluation process.

## 3. Dataset and Training

### Required dataset

Use the BDD100K dataset. Other relevant datasets may be considered as supplements.

### Training approach

- Select a suitable pretrained object-detection model.
- Fine-tune it on the target classes using transfer learning.
- Separate training, validation, and test data where possible.
- Keep the model configuration, class mapping, and annotation format consistent.
- Inspect the dataset labels before training. The standard BDD100K object-detection taxonomy may not provide `cone`, `barrier`, and `stop_sign` as three separate detection classes. Additional annotations or a compatible supplementary dataset may therefore be needed.

## 4. Distance Estimation

Estimate the distance from the camera to each detected object and display the estimate alongside its bounding box.

Possible geometry-based approaches include:

### A. Known-object-size method

For an object with a known physical dimension, an approximate pinhole-camera relationship is:

\[
Z \approx \frac{f_y H}{h}
\]

where:

- `Z` is approximate depth along the camera's viewing direction, in metres.
- `f_y` is the calibrated vertical focal length, in pixels.
- `H` is the known physical object height, in metres.
- `h` is its measured height in the image, in pixels.

This method depends on a suitable known dimension, correct measurements, and camera calibration.

### B. Ground-plane method

For an object point known to touch a flat ground plane, its horizontal distance can be estimated from the camera height, camera pitch, and the calibrated image ray corresponding to that point. This requires a suitable ground-contact point and accurate camera pose information.

Choose a method appropriate to the object and clearly document its assumptions. A distance measured from the camera is not automatically a distance from the robot's base; converting between frames requires the camera-to-robot transform.

## 5. Edge-Device Optimization

Investigate one or more of the following:

- **Lightweight model/backbone:** Start with a small detector suitable for limited hardware.
- **Quantization:** Compare a baseline model with a lower-precision version, if supported by the selected runtime and hardware.
- **Pruning:** Explore structured or unstructured pruning only if time and tooling permit.
- **Runtime/export optimization:** Evaluate a supported inference format or runtime for the target device.

Report actual results rather than assuming an optimization improves performance. Where possible, include before-and-after results for both speed and accuracy.

## 6. Evaluation and Results

Record the configuration used for each experiment so results can be reproduced.

Recommended measurements:

- Detection precision, recall, and mAP where available.
- Distance-estimation mean absolute error (MAE) against measured ground truth.
- CPU FPS and GPU FPS, when both devices are available.
- Model size and input resolution.
- Inference latency and relevant runtime/device details.
- Before-and-after comparisons for optimization experiments.

Use the same test images and comparable settings when comparing model variants. Do not report unmeasured or fabricated results. If a device or metric is unavailable, mark it as not tested.

## 7. Suggested Modular Project Structure

```text
robot-navigation/
├── data/
│   └── robot_nav.yaml
├── datasets/                 # Usually excluded from Git if large
├── src/
│   ├── detection/
│   │   └── detector.py
│   ├── geometry/
│   │   ├── camera.py
│   │   └── distance.py
│   ├── inference/
│   │   └── pipeline.py
│   └── visualization/
│       └── annotate.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── benchmark.py
├── tests/
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```

This structure is a suggestion, not a mandatory layout. Adapt it to any starter repository or submission conventions provided with the assignment. Avoid committing large datasets, secrets, or generated files unless specifically required.

## 8. Deliverables Checklist

- [ ] Fine-tuned detector for the requested target classes.
- [ ] Dataset configuration and explanation of target-class annotations.
- [ ] Inference pipeline that detects and annotates objects.
- [ ] Geometry-based camera-relative distance estimator.
- [ ] Camera calibration inputs and distance-estimation assumptions documented.
- [ ] Example output images or video.
- [ ] Evaluation results for detection and, where possible, distance estimation.
- [ ] CPU/GPU FPS measurements for available hardware.
- [ ] At least one optimization experiment, if feasible within the deadline.
- [ ] Before-and-after comparison when optimization is attempted.
- [ ] Reproducible setup and run instructions.
- [ ] Code committed and submitted to the correct repository.

## 9. Optional Extra Credit

These topics are optional and should be attempted only after the core pipeline is working:

- **Epipolar geometry:** Derive the stereo disparity–depth relationship and explain its assumptions.
- **Homography/perspective transform:** Transform a planar road region into a bird's-eye view.
- **Optical flow:** Track object motion across frames, for example to maintain cone tracks over time.

These extensions do not replace the required detector, distance estimation, or performance evaluation.

## 10. Deadline and Submission Instructions

The assignment states that work must be completed within **two days of receiving the assignment email**.

The provided text contains conflicting repository instructions:

- The email says to upload work to a repository on the candidate's personal GitHub account and reply with the candidate's full name, GitHub username, and repository link.
- The assignment brief says that submissions to personal repositories will not be reviewed and instructs candidates to push to the designated repository provided for the assignment.

**Clarify the required repository with the sender before final submission.** Until confirmed, keep your code in version control and avoid assuming which destination is authoritative.

## 11. Technical Notes and Limitations

- Detection and distance estimation are separate tasks: a correct class prediction does not guarantee an accurate distance.
- A monocular RGB image does not, by itself, determine absolute metric depth without additional geometric constraints, known dimensions, calibration, or another source of depth information.
- Camera calibration and physical camera mounting measurements are important for defensible distance estimates.
- BDD100K's standard detection labels may need supplementation or relabelling to meet the exact target taxonomy.
- Real-time performance depends on the model, input resolution, runtime, and target hardware. Record the actual environment for each benchmark.

---

## Assignment at a Glance

**Build:** A modular object-detection and distance-estimation pipeline for cones, barriers, and stop signs.

**Train with:** Transfer learning from a pretrained detector, using BDD100K and any needed additional annotations/data.

**Estimate distance with:** Camera calibration and geometry.

**Optimize:** Explore a lightweight model, quantization, pruning, or runtime optimization.

**Evaluate:** Detection quality, distance error, model size, and CPU/GPU FPS where available.

**Submit:** Documented source code and results, pushed to the repository confirmed by the assignment sender.
