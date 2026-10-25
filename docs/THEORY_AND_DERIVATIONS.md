# Theoretical Formulations & Mathematical Derivations

This document provides rigorous, step-by-step mathematical proofs and derivations for every geometric, stochastic, and kinematic model implemented in the autonomous robotics perception stack.

---

## Table of Contents

1. [Projective Pinhole Camera Model](#1-projective-pinhole-camera-model)
2. [Analytical Error Propagation for Apparent-Size Depth](#2-analytical-error-propagation-for-apparent-size-depth)
3. [Ground-Plane Ray-Casting & Coordinate Geometry](#3-ground-plane-ray-casting--coordinate-geometry)
4. [Analytical Variance of Ground-Plane Projection](#4-analytical-variance-of-ground-plane-projection)
5. [Optimal 1D Kalman Sensor Fusion (Covariance Intersection)](#5-optimal-1d-kalman-sensor-fusion-covariance-intersection)
6. [Discrete Temporal Kalman Filter (State Estimation & Tracking)](#6-discrete-temporal-kalman-filter-state-estimation--tracking)
7. [Camera-to-Robot Coordinate Frame Transformation (ROS REP-103)](#7-camera-to-robot-coordinate-frame-transformation-ros-rep-103)
8. [Epipolar Geometry & Stereo Disparity-to-Depth Derivation](#8-epipolar-geometry--stereo-disparity-to-depth-derivation)
9. [Planar Homography Matrix for Bird's-Eye View (BEV)](#9-planar-homography-matrix-for-birds-eye-view-bev)
10. [Lucas-Kanade Pyramidal Optical Flow & Time-to-Collision (TTC)](#10-lucas-kanade-pyramidal-optical-flow--time-to-collision-ttc)

---

## 1. Projective Pinhole Camera Model

A standard pinhole camera projects a 3D point $\mathbf{P}_c = [X_c, Y_c, Z_c]^T$ in camera optical coordinates onto a 2D image pixel $\mathbf{p} = [u, v]^T$ via central perspective projection.

### 1.1 The Camera Intrinsic Matrix $K$

The intrinsic matrix $K \in \mathbb{R}^{3 \times 3}$ is defined as:

$$K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

Assuming zero skew ($s = 0$):

$$\tilde{\mathbf{p}} = \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim K \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = \begin{bmatrix} f_x X_c + c_x Z_c \\ f_y Y_c + c_y Z_c \\ Z_c \end{bmatrix}$$

De-homogenizing by dividing by $Z_c$:

$$u = \frac{f_x X_c}{Z_c} + c_x, \quad v = \frac{f_y Y_c}{Z_c} + c_y$$

### 1.2 Longitudinal Depth from Known Physical Height

Consider a physical obstacle of known metric height $H$ whose vertical extent spans from $Y_{\text{top}}$ to $Y_{\text{bottom}} = Y_{\text{top}} + H$ in camera coordinates:

$$v_{\text{bottom}} - v_{\text{top}} = h_{\text{pixels}} = \left(\frac{f_y Y_{\text{bottom}}}{Z_c} + c_y\right) - \left(\frac{f_y Y_{\text{top}}}{Z_c} + c_y\right) = \frac{f_y (Y_{\text{bottom}} - Y_{\text{top}})}{Z_c} = \frac{f_y H}{Z_c}$$

Solving directly for $Z_c$ along the optical axis:

$$\boxed{Z_c \approx \frac{f_y \cdot H}{h_{\text{pixels}}}}$$

Similarly, using known horizontal width $W$ and measured pixel width $w_{\text{pixels}}$:

$$\boxed{Z_c \approx \frac{f_x \cdot W}{w_{\text{pixels}}}}$$

### 1.3 Full 3D Euclidean Range Recovery

Once $Z_c$ is obtained:

$$X_c = \frac{(u_{\text{center}} - c_x) \cdot Z_c}{f_x}, \quad Y_c = \frac{(v_{\text{center}} - c_y) \cdot Z_c}{f_y}$$

The Euclidean 3D distance $D$ from the camera optical center is:

$$\boxed{D = \sqrt{X_c^2 + Y_c^2 + Z_c^2}}$$

---

## 2. Analytical Error Propagation for Apparent-Size Depth

Depth estimation is subject to measurement noise in pixel localization ($\sigma_h$) and physical object manufacturing variations ($\sigma_H$). We model these uncertainties using first-order Taylor expansion error propagation.

### 2.1 Jacobian Derivation

Let $Z = g(h, H) = \frac{f_y H}{h}$. The partial derivatives are:

$$\frac{\partial Z}{\partial h} = -\frac{f_y H}{h^2} = -\frac{Z}{h} = -\frac{Z^2}{f_y H}$$

$$\frac{\partial Z}{\partial H} = \frac{f_y}{h} = \frac{Z}{H}$$

### 2.2 Total Variance Formulation

Assuming pixel detection error $\sigma_h$ and prior dimension variation $\sigma_H$ are uncorrelated ($\operatorname{Cov}(h, H) = 0$):

$$\sigma_{Z,\text{size}}^2 = \left(\frac{\partial Z}{\partial h}\right)^2 \sigma_h^2 + \left(\frac{\partial Z}{\partial H}\right)^2 \sigma_H^2$$

Substituting the partial derivatives:

$$\boxed{\sigma_{Z,\text{size}}^2 = \left(\frac{Z^2}{f_y H}\right)^2 \sigma_h^2 + \left(\frac{Z}{H}\right)^2 \sigma_H^2}$$

### 2.3 Physical Interpretation

- $\left(\frac{Z^2}{f_y H}\right)^2 \propto Z^4$: Measurement variance scales with the **fourth power of distance**. At 2m, a 1-pixel error causes negligible depth error ($\approx 0.01\text{m}$); at 12m, the identical 1-pixel error causes significant depth error ($\approx 0.5\text{m}$).
- When a bounding box is clipped by the image boundary ($x_1 \le 3$, $y_1 \le 3$, $x_2 \ge W-3$, or $y_2 \ge H-3$), $h$ represents only a partial object, artificially inflating $Z$. The system scales $\sigma_{Z,\text{size}}^2$ by a penalty factor of $5.0\times$ to reflect this degradation.

---

## 3. Ground-Plane Ray-Casting & Coordinate Geometry

For objects standing on the ground (traffic cones, barriers), the bottom of the bounding box $(u_c, v_{\text{bottom}})$ coincides with the contact interface on the ground plane.

### 3.1 Camera Orientation & Extrinsics

Let the camera be mounted at height $h_{\text{cam}} > 0$ above the ground, tilted downward by pitch angle $\theta > 0$.
The ground plane in world coordinates is defined by $Y_w = 0$, with the camera origin at $[0, h_{\text{cam}}, 0]^T$.

### 3.2 Ray Projection

The normalized ray vector corresponding to bottom pixel $(u_c, v_{\text{bottom}})$ in camera frame is:

$$\mathbf{r}_c = \begin{bmatrix} \frac{u_c - c_x}{f_x} \\ \frac{v_{\text{bottom}} - c_y}{f_y} \\ 1 \end{bmatrix}$$

The optical ray depression angle $\alpha$ relative to the camera optical axis is:

$$\alpha = \arctan\left(\frac{v_{\text{bottom}} - c_y}{f_y}\right)$$

The total angle $\psi$ relative to the horizontal ground plane is:

$$\psi = \theta + \alpha = \theta + \arctan\left(\frac{v_{\text{bottom}} - c_y}{f_y}\right)$$

### 3.3 Intersection with Ground Plane

By right-triangle trigonometry on the vertical plane:

$$\tan(\psi) = \frac{h_{\text{cam}}}{Z_{\text{ground}}} \implies \boxed{Z_{\text{ground}} = \frac{h_{\text{cam}}}{\tan(\psi)} = \frac{h_{\text{cam}}}{\tan\left(\theta + \arctan\left(\frac{v_{\text{bottom}} - c_y}{f_y}\right)\right)}}$$

---

## 4. Analytical Variance of Ground-Plane Projection

Ground plane estimation is sensitive to vertical pixel localization ($v$), vehicle pitch vibrations ($\theta$), and dynamic suspension height variations ($h_{\text{cam}}$).

### 4.1 Partial Derivatives

Let $Z = f(v, \theta, h_{\text{cam}}) = \frac{h_{\text{cam}}}{\tan(\psi)}$, where $\psi = \theta + \alpha(v)$ and $\alpha(v) = \arctan\left(\frac{v - c_y}{f_y}\right)$.

1. **With respect to angle $\psi$:**
   $$\frac{\partial Z}{\partial \psi} = -\frac{h_{\text{cam}} \sec^2(\psi)}{\tan^2(\psi)}$$

2. **With respect to vertical pixel $v$:**
   $$\frac{\partial \alpha}{\partial v} = \frac{1}{1 + \left(\frac{v - c_y}{f_y}\right)^2} \cdot \frac{1}{f_y} = \frac{\cos^2(\alpha)}{f_y}$$
   $$\frac{\partial Z}{\partial v} = \frac{\partial Z}{\partial \psi} \cdot \frac{\partial \alpha}{\partial v} = -\frac{h_{\text{cam}} \sec^2(\psi)}{\tan^2(\psi)} \cdot \frac{\cos^2(\alpha)}{f_y}$$

3. **With respect to pitch angle $\theta$:**
   $$\frac{\partial Z}{\partial \theta} = \frac{\partial Z}{\partial \psi} \cdot \frac{\partial \psi}{\partial \theta} = -\frac{h_{\text{cam}} \sec^2(\psi)}{\tan^2(\psi)}$$

4. **With respect to camera height $h_{\text{cam}}$:**
   $$\frac{\partial Z}{\partial h_{\text{cam}}} = \frac{1}{\tan(\psi)} = \frac{Z}{h_{\text{cam}}}$$

### 4.2 Total Ground Variance

$$\boxed{\sigma_{Z,\text{ground}}^2 = \left(\frac{\partial Z}{\partial v}\right)^2 \sigma_v^2 + \left(\frac{\partial Z}{\partial \theta}\right)^2 \sigma_\theta^2 + \left(\frac{\partial Z}{\partial h_{\text{cam}}}\right)^2 \sigma_{h_{\text{cam}}}^2}$$

### 4.3 Singularity Safeguard

As $\psi \to 0$ (optical ray approaches the horizon):
$$\lim_{\psi \to 0} \tan(\psi) = 0 \implies \lim_{\psi \to 0} \frac{\partial Z}{\partial \psi} = \infty \implies \sigma_{Z,\text{ground}}^2 \to \infty$$
To prevent numeric divergence:
$$\text{If } \psi \le 0.75^\circ \quad \text{or} \quad \tan(\psi) < 10^{-4} \implies \text{Reject ground hypothesis (return None)}$$

---

## 5. Optimal 1D Kalman Sensor Fusion (Covariance Intersection)

We model the combination of the two independent physical depth measurements as a Maximum Likelihood / Minimum Mean Square Error (MMSE) estimation problem.

### 5.1 Problem Statement

Let the unknown true distance be $z \in \mathbb{R}$. We possess two unbiased measurements with Gaussian error distributions:

$$z_1 = z_{\text{size}} = z + \epsilon_1, \quad \epsilon_1 \sim \mathcal{N}(0, \sigma_1^2)$$
$$z_2 = z_{\text{ground}} = z + \epsilon_2, \quad \epsilon_2 \sim \mathcal{N}(0, \sigma_2^2)$$

Assuming measurement errors are independent ($\mathbb{E}[\epsilon_1 \epsilon_2] = 0$).

### 5.2 Derivation via Weighted Least Squares (BLUE)

We seek a linear unbiased estimator $\hat{z} = w_1 z_1 + w_2 z_2$ such that $\mathbb{E}[\hat{z}] = z$, which enforces:

$$w_1 + w_2 = 1 \implies w_2 = 1 - w_1$$

The variance of the fused estimator is:

$$\sigma_{\text{fused}}^2 = \operatorname{Var}(w_1 z_1 + (1 - w_1) z_2) = w_1^2 \sigma_1^2 + (1 - w_1)^2 \sigma_2^2$$

To minimize $\sigma_{\text{fused}}^2$, set the derivative with respect to $w_1$ to zero:

$$\frac{d(\sigma_{\text{fused}}^2)}{d w_1} = 2 w_1 \sigma_1^2 - 2 (1 - w_1) \sigma_2^2 = 0$$
$$w_1 (\sigma_1^2 + \sigma_2^2) = \sigma_2^2 \implies \boxed{w_1 = \frac{\sigma_2^2}{\sigma_1^2 + \sigma_2^2}}$$
$$w_2 = 1 - w_1 = \frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2}$$

### 5.3 Optimal Fused State & Fused Covariance

$$\boxed{\hat{z}_{\text{fused}} = \frac{\sigma_{\text{ground}}^2 \cdot z_{\text{size}} + \sigma_{\text{size}}^2 \cdot z_{\text{ground}}}{\sigma_{\text{size}}^2 + \sigma_{\text{ground}}^2}}$$

Substituting $w_1$ and $w_2$ into the variance equation:

$$\sigma_{\text{fused}}^2 = \left(\frac{\sigma_2^2}{\sigma_1^2 + \sigma_2^2}\right)^2 \sigma_1^2 + \left(\frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2}\right)^2 \sigma_2^2 = \frac{\sigma_1^2 \sigma_2^4 + \sigma_1^4 \sigma_2^2}{(\sigma_1^2 + \sigma_2^2)^2} = \frac{\sigma_1^2 \sigma_2^2 (\sigma_1^2 + \sigma_2^2)}{(\sigma_1^2 + \sigma_2^2)^2}$$

$$\boxed{\sigma_{\text{fused}}^2 = \frac{\sigma_{\text{size}}^2 \cdot \sigma_{\text{ground}}^2}{\sigma_{\text{size}}^2 + \sigma_{\text{ground}}^2} \iff \frac{1}{\sigma_{\text{fused}}^2} = \frac{1}{\sigma_{\text{size}}^2} + \frac{1}{\sigma_{\text{ground}}^2}}$$

### 5.4 Proof that Uncertainty is Strictly Reduced

$$\sigma_{\text{fused}}^2 = \sigma_1^2 \cdot \left(\frac{\sigma_2^2}{\sigma_1^2 + \sigma_2^2}\right) < \sigma_1^2 \quad \left(\text{since } \frac{\sigma_2^2}{\sigma_1^2 + \sigma_2^2} < 1\right)$$
$$\sigma_{\text{fused}}^2 = \sigma_2^2 \cdot \left(\frac{\sigma_1^2}{\sigma_1^2 + \sigma_2^2}\right) < \sigma_2^2$$

$$\boxed{\sigma_{\text{fused}}^2 < \min(\sigma_{\text{size}}^2, \sigma_{\text{ground}}^2)}$$

$\blacksquare$

---

## 6. Discrete Temporal Kalman Filter (State Estimation & Tracking)

To smooth measurement fluctuations across video streams and derive kinematic metrics (approach velocity and Time-To-Collision), each tracked object is assigned a discrete-time Kalman Filter.

### 6.1 State Space & Kinematic Model

The state vector at time step $k$ is:

$$\mathbf{x}_k = \begin{bmatrix} d_k \\ v_k \end{bmatrix} \in \mathbb{R}^2$$

where $d_k$ is distance (meters) and $v_k = \dot{d}_k$ is the relative range rate (m/s).

Assuming a **Continuous White Noise Acceleration (CWNA)** model driven by zero-mean Gaussian acceleration noise $w(t) \sim \mathcal{N}(0, q)$:

$$\dot{\mathbf{x}}(t) = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} \mathbf{x}(t) + \begin{bmatrix} 0 \\ 1 \end{bmatrix} w(t)$$

Discretizing with sampling interval $\Delta t = t_k - t_{k-1}$:

$$\mathbf{F} = \exp\left(\begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} \Delta t\right) = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}$$

### 6.2 Process Noise Covariance $\mathbf{Q}$

$$\mathbf{Q} = \int_0^{\Delta t} \mathbf{F}(\tau) \begin{bmatrix} 0 \\ 1 \end{bmatrix} q \begin{bmatrix} 0 & 1 \end{bmatrix} \mathbf{F}(\tau)^T d\tau = q \int_0^{\Delta t} \begin{bmatrix} \tau \\ 1 \end{bmatrix} \begin{bmatrix} \tau & 1 \end{bmatrix} d\tau$$

$$\boxed{\mathbf{Q} = q \begin{bmatrix} \frac{\Delta t^3}{3} & \frac{\Delta t^2}{2} \\ \frac{\Delta t^2}{2} & \Delta t \end{bmatrix}}$$

where $q \approx 1.2\text{ m}^2/\text{s}^3$ represents maneuverability process spectral density.

### 6.3 Measurement Model

At each frame, vision geometry provides the fused distance $z_k$:

$$z_k = \mathbf{H} \mathbf{x}_k + \nu_k, \quad \nu_k \sim \mathcal{N}(0, R)$$
$$\mathbf{H} = \begin{bmatrix} 1 & 0 \end{bmatrix}, \quad R = \sigma_{\text{fused}}^2$$

### 6.4 Prediction Equations

$$\hat{\mathbf{x}}_{k|k-1} = \mathbf{F} \hat{\mathbf{x}}_{k-1|k-1}$$
$$\mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{Q}$$

### 6.5 Measurement Update (Joseph Form)

1. **Innovation Residual:**
   $$y_k = z_k - \mathbf{H} \hat{\mathbf{x}}_{k|k-1} = z_k - \hat{d}_{k|k-1}$$

2. **Innovation Covariance:**
   $$S_k = \mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + R = P_{11,k|k-1} + R$$

3. **Optimal Kalman Gain:**
   $$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}^T S_k^{-1} = \frac{1}{S_k} \begin{bmatrix} P_{11} \\ P_{21} \end{bmatrix}$$

4. **Updated State Estimate:**
   $$\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k y_k$$

5. **Joseph-Form Covariance Update:**
   Standard covariance update $\mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}$ is prone to loss of symmetry and positive definiteness due to floating-point truncation. We implement the Joseph stabilized formulation:
   $$\boxed{\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H})^T + \mathbf{K}_k R \mathbf{K}_k^T}$$

### 6.6 Time-to-Collision (TTC)

When $v_k < -0.10\text{ m/s}$ (obstacle is closing in on the robot):

$$\boxed{\text{TTC} = -\frac{d_k}{v_k} \quad [\text{seconds}]}$$

---

## 7. Camera-to-Robot Coordinate Frame Transformation (ROS REP-103)

In robotics software stacks, sensors do not share coordinates with the robot chassis. We map between the **Camera Optical Frame** and the **Robot Base Frame (`base_link`)** following the **ROS REP-103 standard**:

- **Camera Optical Frame $\mathcal{F}_c$:** $+X_c$ points Right, $+Y_c$ points Down, $+Z_c$ points Forward along optical axis.
- **Robot Base Frame $\mathcal{F}_r$ (`base_link`):** $+X_r$ points Forward, $+Y_r$ points Left, $+Z_r$ points Up from the ground contact footprint.

```text
       Camera Frame (Fc)                     Robot Base Frame (Fr)
             +Z (forward)                          +X (forward)
              ^                                         ^
              │                                         │
              │                                         │
              └───► +X (right)          +Y (left) ◄─────┘
              │                                         │
              ▼                                         ▼
             +Y (down)                                 +Z (up)
```

### 7.1 Coordinate Rotation & Translation

Let the camera be mounted at:
- Elevation $h_{\text{cam}}$ above ground plane
- Forward offset $x_{\text{mount}}$ ahead of robot center
- Lateral offset $y_{\text{mount}}$ relative to robot center
- Downward pitch angle $\theta$ (rotation around camera $X_c$ axis)

A point $[X_c, Y_c, Z_c]^T$ in $\mathcal{F}_c$ transforms to $[X_r, Y_r, Z_r]^T$ in $\mathcal{F}_r$ via:

$$\begin{bmatrix} X_r \\ Y_r \\ Z_r \end{bmatrix} = \begin{bmatrix} \cos\theta & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}_{\text{pitch}} \begin{bmatrix} 0 & 0 & 1 \\ -1 & 0 & 0 \\ 0 & -1 & 0 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} + \begin{bmatrix} x_{\text{mount}} \\ y_{\text{mount}} \\ h_{\text{cam}} \end{bmatrix}$$

Expanding explicitly:

$$\boxed{X_r = Z_c \cos\theta - Y_c \sin\theta + x_{\text{mount}}}$$
$$\boxed{Y_r = -X_c + y_{\text{mount}}}$$
$$\boxed{Z_r = h_{\text{cam}} - (Y_c \cos\theta + Z_c \sin\theta)}$$

### 7.2 Inverse Transformation (Robot to Camera)

$$\Delta X = X_r - x_{\text{mount}}, \quad \Delta Y = Y_r - y_{\text{mount}}, \quad \Delta Z = Z_r - h_{\text{cam}}$$

$$X_c = -\Delta Y$$
$$Y_c = -\Delta Z \cos\theta - \Delta X \sin\theta$$
$$Z_c = \Delta X \cos\theta - \Delta Z \sin\theta$$

---

## 8. Epipolar Geometry & Stereo Disparity-to-Depth Derivation

Consider an ideal rectified stereo rig with parallel optical axes separated by horizontal baseline $B$ (meters) with identical focal length $f$ (pixels).

```text
        Left Optical Center                       Right Optical Center
               O_L                                       O_R
                │ <---------------- B ----------------->  │
                │                                         │
         Left Image Plane                          Right Image Plane
         ┌──────────────┐                          ┌──────────────┐
         │     x_L      │                          │     x_R      │
         └──────┬───────┘                          └──────┬───────┘
                 \                                       /
                  \                                     /
                   \                                   /
                    \                                 /
                     ▼                               ▼
                               Scene Point P(X, Y, Z)
```

### 8.1 Geometric Projections

1. **Left Camera Image Plane:**
   $$x_L = f \cdot \frac{X}{Z} + c_x$$

2. **Right Camera Image Plane:**
   In the right camera coordinate frame, the point coordinates are $[X - B, Y, Z]^T$:
   $$x_R = f \cdot \frac{X - B}{Z} + c_x = \frac{f X}{Z} - \frac{f B}{Z} + c_x$$

3. **Disparity Definition ($d$):**
   $$d = x_L - x_R = \left(\frac{f X}{Z} + c_x\right) - \left(\frac{f X}{Z} - \frac{f B}{Z} + c_x\right) = \frac{f \cdot B}{Z}$$

### 8.2 Depth Recovery

Solving for depth $Z$:

$$\boxed{Z = \frac{f \cdot B}{d}}$$

---

## 9. Planar Homography Matrix for Bird's-Eye View (BEV)

To generate an orthographic 2D top-down ground map, we model the perspective mapping between the camera image plane and the ground plane as a projective planar homography $H \in \mathbb{R}^{3 \times 3}$.

### 9.1 Homography Equation

For ground plane points $\mathbf{P}_{\text{ground}} = [X_g, Z_g, 1]^T$ and camera image coordinates $\mathbf{p}_{\text{cam}} = [u, v, 1]^T$:

$$\begin{bmatrix} u_{\text{bev}} \\ v_{\text{bev}} \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} u_{\text{cam}} \\ v_{\text{cam}} \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u_{\text{cam}} \\ v_{\text{cam}} \\ 1 \end{bmatrix}$$

### 9.2 Direct Linear Transformation (DLT)

Given 4 non-collinear correspondence pairs $(\mathbf{x}_i, \mathbf{x}'_i)$, each point pair contributes two linearly independent equations:

$$\mathbf{a}_{x,i}^T \mathbf{h} = 0, \quad \mathbf{a}_{y,i}^T \mathbf{h} = 0$$

where $\mathbf{h} = \operatorname{vec}(\mathbf{H})$. Solving $\mathbf{A} \mathbf{h} = \mathbf{0}$ via Singular Value Decomposition (SVD):

$$\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T \implies \mathbf{h} = \mathbf{V}_{:,9} \quad (\text{eigenvector corresponding to smallest singular value})$$

---

## 10. Lucas-Kanade Pyramidal Optical Flow & Time-to-Collision (TTC)

For dynamic obstacles, feature motion across frames $I(x, y, t)$ satisfies the **Optical Flow Brightness Constancy Constraint**:

$$I(x + \delta x, y + \delta y, t + \delta t) \approx I(x, y, t)$$

### 10.1 Optical Flow Gradient Equation

Expanding by Taylor series:

$$I_x u + I_y v + I_t = 0$$

where $I_x = \frac{\partial I}{\partial x}, I_y = \frac{\partial I}{\partial y}, I_t = \frac{\partial I}{\partial t}$, and $[u, v]^T = \left[\frac{dx}{dt}, \frac{dy}{dt}\right]^T$.

Over an $n \times n$ window ($n=15$), this yields an overdetermined system $\mathbf{A} \mathbf{u} = \mathbf{b}$:

$$\begin{bmatrix} I_x(p_1) & I_y(p_1) \\ \vdots & \vdots \\ I_x(p_k) & I_y(p_k) \end{bmatrix} \begin{bmatrix} u \\ v \end{bmatrix} = -\begin{bmatrix} I_t(p_1) \\ \vdots \\ I_t(p_k) \end{bmatrix}$$

The least-squares solution is:

$$\begin{bmatrix} u \\ v \end{bmatrix} = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{b} = \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}^{-1} \begin{bmatrix} -\sum I_x I_t \\ -\sum I_y I_t \end{bmatrix}$$

### 10.2 TTC Derivation from Scale Expansion

For an obstacle of height $H$ observed at focal length $f$, image size $h(t) = \frac{f H}{Z(t)}$. Differentiating with respect to time:

$$\frac{dh}{dt} = -\frac{f H}{Z^2} \frac{dZ}{dt} = -\frac{h}{Z} \dot{Z}$$

Rearranging:

$$\boxed{\text{TTC} = -\frac{Z}{\dot{Z}} = \frac{h(t)}{\frac{dh}{dt}}}$$

This allows Time-to-Collision to be directly verified both from **filtered state range rate** ($\text{TTC} = -d / \dot{d}$) and independently from **optical image expansion rate** ($\text{TTC} = h / \dot{h}$).
