# KITTI07 Metric Ground Extraction → Correspondence → Hand–Eye Refinement

## What is implemented

`spade-calib prepare-ground` is a new front-end for **the existing**
`spade-calib refine-ground` 3-DoF/6-DoF solver.

Algorithm provenance:
- Camera: archived `COMPA_D1_ROIFree_Plane_Experiment_20261107/experiment.py`,
  adapted from local PCA → depth discontinuity/planarity filtering → seeded
  image-connected region growing → PCA refit → overlap NMS.
- LiDAR: same archive's repeated 3-point RANSAC → PCA refit, with an
  explicit expected-up and positive below-sensor ground height prior.
- Matching: archived `COMPA_D1_CameraGuided_20261107/run_camera_guided.py`,
  adapted from Hand–Eye initial projection → dilated Camera plane mask →
  independent metric Plane-depth gate → normal-guided LiDAR RANSAC →
  3D connected component → offset/normal/support gates.
- Optimization: existing `calibration/solver.py`: Hand–Eye Motion residual
  retained, SfM scale fixed, identical matches/pairs for **Rotation fixed
  Translation 3-DoF** and **Rotation+Translation 6-DoF**.
- Optional `calib.txt` is used **after fitting** to evaluate KITTI GT errors.
  This command does not read KITTI GT to create matches.

This is a structural, configurable port of the old experiments, **not**
a byte-identical recreation of their D1 outputs.

## Inputs and frame contract

You need:
1. `handeye.json` from `spade-calib handeye` using KITTI07
   `odom_lidar.txt`, `camera_trajectory_tum.txt`, `times.txt`.
2. An accumulated LiDAR NPZ created by `spade-calib accumulate-kitti`,
   including `xyz`, `frame_id=kitti_lidar_frame_N`,
   `reference_timestamp_ns`, `frame_ids`, `motion_model=rigid`.
3. Dense Camera **metric optical-Z depth** from **the exact same frame N**,
   as `.npy` float H×W. It may have been obtained from DA2 by an externally
   validated SfM-anchor metric alignment. Raw DA2 inverse depth, global
   affine-ambiguous SfM depth, ray range, and KITTI GT/LiDAR-projected
   depth are **not valid inputs** for this experiment.
4. Camera JSON `K`/distortion with width, height and depth type:

```json
{
  "K": [[500.0, 0.0, 600.0], [0.0, 500.0, 180.0], [0.0, 0.0, 1.0]],
  "distortion": [],
  "width": 1241,
  "height": 376,
  "depth_type": "camera_z_m"
}
```

**The JSON above is an illustrative syntax example, NOT KITTI07
calibration.** Set values from your particular KITTI07 rectified Camera 2
`P2[:3,:3]` and the actual depth/image dimensions, with distortion zero
only for the rectified optical frame. Camera optical coordinates are
`x right, y down, z forward`. LiDAR up is assumed `+[0,0,1]` in the
target LiDAR frame. Ground lies below the LiDAR at a positive plane offset
(default 0.35–3.5 m). The current front-end is specialized to KITTI
Odometry rigid accumulation; no per-return deskew or ring timing exists in
KITTI .bin files.

To prevent accidental temporal misalignment, the CLI requires
`--frame-index N --depth-frame-index N` and verifies the NPZ
`frame_id` and `reference_timestamp_ns` against `times.txt`.
This verifies **declared** frame metadata; it cannot infer whether an
incorrectly named depth file depicts the same scene.

## Step A: Existing KITTI accumulation

```bash
spade-calib accumulate-kitti \
  --lidar-dir /data/kitti07/velodyne \
  --glim /data/kitti07/odom_lidar.txt \
  --kitti-times /data/kitti07/times.txt \
  --start-frame 100 --stop-frame 110 --target-frame 105 \
  --output outputs/accumulated_105.npz
```

The frame range/target 100–110/105 is a **user-choice example**, not
a prevalidated KITTI07 ground-quality window. The SfM metric depth must
come from the Camera 2 image at frame **105**.

## Step B: Plane extraction & matching alone

```bash
python -m pip install -e '.[research,dev]'
spade-calib prepare-ground \
  --handeye outputs/handeye07.json \
  --cloud outputs/accumulated_105.npz \
  --metric-depth /data/kitti07/metric_depth/000105.npy \
  --camera-json /data/kitti07/camera2_intrinsics.json \
  --kitti-times /data/kitti07/times.txt \
  --frame-index 105 --depth-frame-index 105 \
  --output-dir outputs/ground_105
```

Output:
- `ground_105/ground_matches.json`: the **candidate** pair in the existing
  `GroundMatch` schema. `verified_by_independent_review=false` and
  `source_type=automatic_provisional_camera_guided_ground`.
- `ground_105/ground_diagnostics.json`: Camera plane counts, LiDAR RANSAC
  inlier/RMSE, guided matching attempts, rejection reasons, tentative match
  metrics and source/frame metadata.

If no candidate passes, diagnostics are written and the command fails; it
does **not** invent ground coordinates or fall back to a GT plane.

## Step C: One-command extraction + both optimization modes

```bash
spade-calib prepare-ground \
  --handeye outputs/handeye07.json \
  --cloud outputs/accumulated_105.npz \
  --metric-depth /data/kitti07/metric_depth/000105.npy \
  --camera-json /data/kitti07/camera2_intrinsics.json \
  --kitti-times /data/kitti07/times.txt \
  --frame-index 105 --depth-frame-index 105 \
  --glim /data/kitti07/odom_lidar.txt \
  --sfm /data/kitti07/camera_trajectory_tum.txt \
  --calib /data/kitti07/calib.txt --camera-index 2 \
  --run-refinement --allow-provisional \
  --output-dir outputs/ground_105
```

The explicit `--allow-provisional` flag acknowledges that the auto
correspondence is still a **hypothesis**, not independently annotated GT.
This experiment runs exactly two ablations on the same detected pair:
`fixed_rotation` (3-DoF Translation) and `optimize_rotation` (6-DoF).
The output `ground_refinement.json` includes initial/final plane residuals,
Jacobian rank, boundary-hit flags, and, if `--calib` was supplied, the
independently evaluated baseline/fixed/full KITTI GT errors.

## What is NOT yet done

- **DA2 → metric Camera Z-depth alignment** is not in this pipeline.
  It must be established independently from SfM anchors and validated;
  blindly multiplying by the Hand–Eye SfM trajectory scale is insufficient
  because monocular depth prediction may carry a distinct scale/shift.
- Automatically extracted plane correspondence is **not** physical GT,
  and the initial Hand–Eye Extrinsic influences which LiDAR patch is selected.
  Validate masks and matches visually and on held-out scenes.
- A single ground plane constrains translation only along its normal; the
  remaining translation axes are constrained by the Hand–Eye motion model
  and can remain poorly conditioned. Successful optimizer convergence
  does not imply a reliable Calibration improvement.
- RANSAC, prior-normal, depth/offset, connected-component and region
  parameters are ported engineering heuristics, not measurement noise
  estimates. KITTI07 may require parameters validated on training scenes.
- Do not tune thresholds, weights or scene selection against evaluation
  GT then claim an independent test result.
- If the camera depth is biased, its plane offset is biased even if the
  LiDAR ground is geometrically accurate. Report Camera-depth diagnostics.

The outputs are suitable for controlled A/B experimentation, not yet a
validated all-scene production Calibration method.
