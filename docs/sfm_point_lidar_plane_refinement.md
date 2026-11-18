# SfM point -> finite LiDAR plane patch, joint 7-DoF prototype

This is a **GT-free research baseline**, not verified automatic calibration.
Unlike the DA2 depth pipeline, camera 3D points come from COLMAP tracks.
For KITTI Odometry the LiDAR cloud must be `accumulate-kitti` output in the
**target LiDAR sensor frame**. Never use a cloud expressed in a global map
frame with `T_camera_lidar` directly.

## Measurement and transform

- `X = T_camera_lidar`: `p_C_metric = R_CL p_L + t_CL`.
- `s > 0`: metres per COLMAP SfM unit.
- Camera observation `p_C_sfm` is **unscaled Camera-frame XYZ** for the target image.
- LiDAR patch `n_L.T p_L + d_L = 0`, unit normal, metric metres.
- Residual in LiDAR metres:

  `r = n_L.T R_CL.T (s p_C_sfm - t_CL) + d_L`.

The joint 7-DoF estimator also uses the existing motion equation
`B(s) X = X A`, with the **same** GLIM and COLMAP trajectories used to obtain
the initial Hand-Eye estimate. Optimized variables are left Camera rotation,
additive Camera translation and log SfM scale. A bounded robust least-squares
solver holds the *provisional matches fixed* within a single solve.

Motion and plane residuals are normalized with assumed noise scales. Each
plane patch receives equal total normalization to prevent dense ground from
automatically dominating; this is a heuristic, **not** a calibrated covariance.
The local rank report is a sensitivity diagnostic, **not** an uncertainty guarantee.

## 1. Prepare matches (KITTI frame 105 example)

```bash
spade-calib prepare-sfm-planes \
  --handeye outputs/kitti07_ground/handeye.json \
  --colmap-model /home/myungw00/PROJECT/pycolmap/output_kitti07_ov5/sparse/0 \
  --image-name 000105.png --frame-index 105 \
  --camera-json outputs/kitti07_ground/camera2_intrinsics.json \
  --cloud outputs/kitti07_ground/accumulated_000105.npz \
  --kitti-times /path/to/data_odometry_poses/dataset/sequences/07/times.txt \
  --max-planes 16 \
  --output outputs/kitti07_ground/sfm_plane_matches_105.npz
```

Outputs two files: `sfm_plane_matches_105.npz` and the same-stem `.json`
report. The NPZ stores point-to-patch residual data; the JSON includes
source and target-frame checks, per-patch supports, scan IDs, and an explicit
`automatic_provisional_not_ground_truth` status. The COLMAP model **must**
contain `images.bin/txt` and `points3D.bin/txt` 3D tracks. A GT-fixed pose-only
model with no tracks is not sufficient.

The cloud target stamp is checked against `times.txt`, and the image filename
must correspond to the **same** target frame. Rigid KITTI accumulation lacks
per-return timestamps/deskew; this is a known geometric error source.

## 2. Refine jointly

```bash
spade-calib refine-sfm-planes \
  --handeye outputs/kitti07_ground/handeye.json \
  --matches outputs/kitti07_ground/sfm_plane_matches_105.npz \
  --glim /home/myungw00/PROJECT/glim/result_kitti07_16/odom_lidar.txt \
  --sfm /home/myungw00/PROJECT/pycolmap/output_kitti07_ov5/sparse/0 \
  --kitti-times /path/to/data_odometry_poses/dataset/sequences/07/times.txt \
  --allow-provisional \
  --output outputs/kitti07_ground/sfm_plane_refinement_105.json
```

The `--sfm` model MUST be the same model used to prepare matches. For
unreviewed matches the explicit `--allow-provisional` acknowledgement is
mandatory. Refine will refuse to silently combine different Hand-Eye scales,
models, or frame identities. The output contains `handeye.T_camera_lidar`
for compatibility with `spade-calib evaluate-kitti-handeye` (evaluation only).

```bash
spade-calib evaluate-kitti-handeye \
  --result outputs/kitti07_ground/sfm_plane_refinement_105.json \
  --calib /path/to/KITTI/07/calib.txt --camera-index 2
```

**Never use KITTI calib.txt to generate point-plane matches or fit extrinsics.**

## Matching limits and failure policy

- LiDAR patch extraction: voxel sampling, dominant + orientation-aware local-PCA RANSAC, finite connected support,
  PCA refit, non-collinear surface support. No semantic/dynamic rejection.
- Initial correspondence: project unscaled SfM Camera XYZ through current
  `s` and `T_camera_lidar` into the target LiDAR frame; use signed plane
  distance AND distance to actual finite patch support, uniqueness margin,
  and minimum matched-track count.
- Match report includes a **normal-rank proxy**. If this is 1 (e.g. only
  horizontal ground patches), point-plane residuals CANNOT independently
  constrain all translation directions. A full-rank joint motion+plane
  Jacobian does not prove that the added plane measurements improved all 7 DoF.
- Matching is a **hypothesis**, not physically verified correspondence. Far
  walls, parallel facades, dynamic objects and incorrect initial extrinsics
  can give wrong matches. Review patch projections and compare with GT *only
  after fitting*. Matching uses one target image/scan accumulation, while
  the motion loss uses the full sequence. Do not count the same SfM track as
  independent measurements across several target frames.
- Repeated LiDAR scans share GLIM pose errors; normal FIM/information and
  covariance would be overconfident if scan/track correlations were ignored.
- No claim of physical accuracy is made by a residual reduction. Benchmark
  multiple scenes and initial extrinsic perturbations.

## Smoke tests performed

- Deterministic synthetic 7-DoF recovery; finite patch support rejection;
  multiple planar surfaces; duplicate SfM observation rejection.
- Real archived KITTI07 frame 105 (11 rigid-accumulated scans, 330370 points,
  1279 COLMAP metric anchors): extraction and provisional matching ran.
  With `max_planes=16`, the prototype generated 358 candidate matches from
  7 patches, but the matched patch-normal rank proxy was **1**. This is
  **not** evidence of improved camera-LiDAR calibration. It points to a
  need for additional directional surfaces / better scan selection.
- Full real KITTI07 joint-solver accuracy is **not yet measured** here: the
  original GLIM trajectory, full COLMAP sparse tracks and KITTI times.txt
  are not included in the provided data bundle.

## Orientation-aware extraction (updated prototype)

The previous repeated maximum-inlier global RANSAC spent most of the 16 patch
slots on parallel near-horizontal surfaces. The default extraction now:

1. Fits **one** dominant reference plane with global RANSAC.
2. Estimates local PCA normals on voxel-sampled LiDAR points; only locally
   planar seeds generate additional hypotheses.
3. Prioritizes normals differing at least 25 degrees from the dominant plane,
   with additional priority for directions novel relative to *all* accepted
   patches. The orientation rule changes proposal search, **not** the
   inlier threshold or final validity test.
4. Accepts only finite connected support, double PCA refit, 2D width and
   plane RMSE checks. Falls back to unrestricted RANSAC if no novel patches
   are present. Does **not** invent perpendicular planes.

Defaults may be tuned explicitly:

```bash
spade-calib prepare-sfm-planes ... \
  --max-planes 16 \
  --diverse-normal-min-angle-deg 25 \
  --novel-normal-min-angle-deg 25
```

To reproduce the non-orientation-aware baseline, add `--no-orientation-aware`.

**KITTI07 frame 105 local smoke test:** same 1,305,598-point 11-scan cloud;
30,000 voxel-sampled points; 16 extracted patches. The updated version produced
15 patches differing by >=25 degrees from its dominant plane (many are
near-vertical and some oblique). The candidate matcher accepted **432** SfM
point/plane assignments on **5** patches, with normal rank *proxy* **3**.
The prior user result had 224 matches on 2 predominantly horizontal patches,
normal rank proxy 1. These comparisons concern **candidate availability only**:
no new full real 7-DoF joint result has been obtained, and some automatic
matches are likely incorrect. The GT-based checks were performed only after
matching as diagnostics, not to choose plane proposals or optimize extrinsics.
The local orientation tests reject known wall-free clouds and compare a
synthetic ground-dominated scene against the old strategy.