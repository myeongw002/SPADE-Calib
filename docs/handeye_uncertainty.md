# Full-recording Hand–Eye (7-DoF) and uncertainty analysis

**Implementation phase:** after trajectory screening PR #6, before geometry-aware Scene Selection.
This module fits **one common rigid Extrinsic** using **all synchronized GLIM/COLMAP trajectories**. It does not perform Camera–LiDAR plane/depth calibration, and it does not infer clock offset or physical sensor uncertainty automatically.

## Quick start

```bash
pip install -e '.[research,dev]'
spade-calib handeye \
  --config configs/local-recording.yaml \
  --output outputs/handeye.json \
  --bootstrap-resamples 96 --bootstrap-block-s 20
```

When the original image/PCD folders are not available, use the trajectory-only mode. No PCD is needed for **trajectory Hand–Eye estimation**:

```bash
spade-calib handeye \
  --glim /private/traj_lidar.txt \
  --sfm /private/camera_poses.csv \
  --camera-to-lidar-offset-ns 0 \
  --output outputs/handeye.json \
  --bootstrap-resamples 96 --bootstrap-block-s 20
```

The complete recording mode uses all **registered image poses** backed by available RGB images, even if no raw PCD is near that camera timestamp; ignoring unmatched LiDAR frames would erroneously throw away usable GLIM camera trajectory pairs.

The result includes **both** `T_lidar_camera` and its inverse `T_camera_lidar`. For projection into the RGB camera, use `T_camera_lidar`. The output assumes the provided GLIM and COLMAP conventions from `docs/scene_selection.md`.

## Mathematical definition

Let `A_i = inv(T_world_lidar(t_i)) @ T_world_lidar(t_j)` and `B_i = inv(T_sfm_world_camera(t_i)) @ T_sfm_world_camera(t_j)`, where SfM translations are not metric. The unknowns are `X = T_lidar_camera` and positive `s` (metric metres per SfM unit):

`A_i X = X B_i(s)`.

The rotation residual in **radians** and translation residual in **metres** are

```
r_R_i = Log[ (R_A_i R_X)^T (R_X R_B_i) ]
r_t_i = (R_A_i - I) t_X + t_A_i - R_X (s t_B_i)
```

Rotation is updated via local *left* perturbation `R_X(delta) = Exp(delta_phi) R_X0`, translation via `t_X = t_X0 + delta_t`, and scale via `s = s0 exp(delta_log_s)`. `scipy.optimize.least_squares` minimizes a robust loss over the whitened 6-component residuals. The rotation and translation whitening sigmas (defaults 0.5 deg and 0.05m for each relative-motion residual) are **assumptions**, not measured GLIM or COLMAP pose noise; changing them changes the reported model-based covariance. The initial `R_X,s,t_X` come from PR #6's closed-form provisional fit.

## Joint Jacobian / weak directions

The **7-dimensional Jacobian** is centrally differentiated at the converged solution in the coordinate chart `[left_delta_phi(rad), delta_t(m), delta_log_s]`. The information surrogate is `H = J^T J`. It **includes rotation–translation–scale cross-terms** unlike PR #6's block diagonal screening proxy.

For meaningful singular-vector comparisons with mixed units, the Jacobian columns are normalized by reference values, by default 1 degree for angular perturbation, 0.1m for each Translation axis, and 0.1 for log scale. SVD gives rank and weak *joint parameter directions*; its relative singular-value thresholds are diagnostics, **not probabilistic noise estimates**.

If the 7-parameter system is numerically rank deficient, the API **refuses to return finite covariance or standard deviations** rather than silently using a pseudoinverse. Any regularized or prior-conditioned covariance would have to identify that prior explicitly.

## Model-based covariance (conditional, not sensor ground truth)

Under the **strong additional assumption** that the whitened relative-motion residuals are independent, correctly scaled and locally linear Gaussian observations,

`Cov_joint ≈ (J^T J)^{-1}`.

The uncertainty of the six Extrinsic parameters must marginalize the unknown SfM scale. The proper covariance is the top-left 6×6 block of the *inverse full joint H* (`Cov_joint[:6,:6]`). Algebraically this equals the inverse of the Schur complement of the scale block, not the inverse of `H[:6,:6]`. Individual rotation/translation tangent-coordinate standard deviations come from the diagonal, and the complete 6×6 covariance and 6×6 correlation matrix are retained.

**Important:** Relative Motion pairs share trajectory poses; they are correlated, and GLIM/COLMAP per-pose cross-covariances are not available here. We do not adjust for these dependencies in the model-based covariance, and robust-loss influence is not represented by a derived sandwich estimator. Consequently **this covariance is an assumed-noise local sensitivity estimate, not calibrated physical 1-sigma accuracy**. Full-rank status is not proof of well-conditioned, unbiased or precise Calibration.

## Temporal block bootstrap (empirical subset stability)

Optionally divide the sequence of motion pair **start times** into contiguous blocks (default 20s). Resample those blocks with replacement and **refit all seven parameters**; report the sample covariance of local 6-DoF perturbations plus log-scale differences. This respects some local temporal correlation, unlike naive i.i.d. pair resampling. Every failed or rank-deficient refit is counted, and bootstrap covariance is withheld if too few succeed.

This quantifies variation with respect to the observed trajectory segments, not the sensor's physical stochastic covariance. Shared systematic errors of GLIM/SfM, camera intrinsics errors, clock misalignment, scene mismatch and unknown true Extrinsic may not appear in bootstrap variability. Bootstrap results are not automatically a valid probabilistic prior for final Geometry Optimization.

## Downstream use and present limits

- `handeye.optimization`: convergence, robust objective, relative-rotation and translation residual distribution.
- `observability_and_model_covariance`: rank, normalized singular values, weak 7-parameter directions, conditional 7×7 and marginalized 6×6 covariance, per-DOF tangent standard deviations, correlations and largest combined Extrinsic uncertainty direction.
- `bootstrap`: empirical 7×7 and 6×6 covariance and weak modes, valid/failed refits, assumptions.
- **No scene-wise or geometry-based information matrices computed** by this module; no final Plane/Depth Calibration and no Prior inserted into the final optimizer yet.
- Actual prediction accuracy requires independent calibrated Extrinsic reference or held-out geometric measurement. Comparison of two trajectory-fit transforms is only a stability check.

## Local historical recording validation (not in public repository)

With user-supplied GLIM `traj_lidar(2).txt` and pycolmap `camera_poses(2).csv`, 4,747 camera/GLIM poses synchronized, 477 relative-motion pairs fitted at a 2-s baseline and 1-s pair stride, with positive SfM metric scale 9.22244. Model-based Jacobian numerical rank 7/7 under specified scaling/threshold; its weakest **Extrinsic uncertainty direction** is primarily LiDAR Z Translation (after scale marginalization), as is the bootstrap result. Rotation fit median residual ~0.453deg. A full 20-s, 96-draw temporal block bootstrap gave 24 temporal blocks and substantial uncertainty in LiDAR Z Translation.

A separate **first-half / second-half split** yields substantially different LiDAR Z Translation estimates even though the Jacobian is numerically full rank: full ≈ 0.0185m, first ≈ 0.2557m, second ≈ -0.1471m. This illustrates that rank alone is insufficient for confidence. See private local report for complete values. No private source data is committed.

## Safety/engineering notes

- If GLIM/SfM timestamps are not synchronized, the model can yield plausible but wrong Extrinsic. Clock offset is never estimated here.
- If the sensor attachment changes during recording, a common Extrinsic model is invalid.
- The rotation tangent and translation components are expressed in the LiDAR frame of `T_lidar_camera`; do not interpret them as independent Euler angle uncertainties or transform them to Camera frame without covariance transport.
- Scene geometry, camera depth and LiDAR reflectance are not used; there is no assurance that weak DOFs are recoverable from the selected scenes.