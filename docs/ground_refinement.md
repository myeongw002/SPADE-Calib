# Hand-Eye + metric Ground-plane refinement

**Status:** Implemented standalone optimizer and comparison CLI. The current
repository does not yet automatically extract camera metric ground planes or
verify camera/LiDAR plane correspondences for KITTI07. Independent measured
ground pairs must be provided by the caller. No real KITTI07 calibration
improvement is claimed merely because these functions are implemented.

## Scope

Inputs:
- Existing `spade-calib handeye` JSON result, with LiDAR->Camera `T_camera_lidar`.
- The **same** GLIM and SfM trajectories / KITTI frame-index mapping as Hand-Eye.
- One or more independently measured, physically matched camera/LiDAR **metric**
  ground planes, each expressed in the local sensor frames at the **same scene time**.
- Optional KITTI `calib.txt` used **after** optimization for evaluation only.

The saved Hand-Eye SfM metric scale is fixed in both optimization modes:

| Mode | Variables | Active residuals |
| --- | --- | --- |
| `fixed_rotation` | Camera translation (3) | Hand-Eye translation + rotation (constant), ground offset |
| `optimize_rotation` | Camera left-rotation (3) + translation (3) | Hand-Eye motion, ground normal + offset |

Ground normal residual and plane-offset residual come from the archived
`COMPA_handeye_ground_loss_20261107/run_handeye_ground_joint.py`
convention, not its original 8-parameter DA2 inverse-depth solver:

```text
    p_C = R_CL p_L + t_CL
    n_L^C = R_CL n_L
    d_L^C = d_L - (R_CL n_L)^T t_CL
    r_n   = (n_L^C x n_C) / sigma_normal_rad
    r_d   = (d_L^C - d_C) / sigma_offset_m
```

A camera plane normal may be reversed together with its offset. Its sign
is chosen ONCE using the initial Hand-Eye rotation before optimization,
and the initial angle to the paired LiDAR ground must pass the configured
gate (default 45 degrees). This sign gate does not establish physical
correspondence validity.

Hand-Eye `B(s) X = X A` residuals are **retained in the objective**, not
merely used as an initializer. Residuals are normalized by the number of
motion pairs/ground matches to prevent sample-count-only reweighting.
Default noise scales and relative weights are *heuristics*. The same
motion pairs and input ground matches are used by both ablation arms.

## Ground JSON interface (illustrative geometry ONLY)

```json
{
  "schema_version": 1,
  "convention": "T_camera_lidar",
  "matches": [
    {
      "id": "example_only_not_a_kitti_measurement",
      "lidar": {
        "frame": "lidar_target",
        "normal": [0.0, 0.0, 1.0],
        "offset_m": -1.5
      },
      "camera": {
        "frame": "camera",
        "normal": [0.0, -1.0, 0.0],
        "offset_m": -1.7
      },
      "sigma_normal_deg": 3.0,
      "sigma_offset_m": 0.15
    }
  ]
}
```

The above numbers are **not KITTI07 ground observations**, and must not be
used to claim real-world accuracy. Ground planes use the unit-normal,
signed offset form `n.T @ p + d = 0`, with **metre** offsets in their
respective sensor frames. Camera ground should come from measured metric
depth, not unscaled monocular DA2 or a world-frame SfM plane. KITTI GT
extrinsic must not be used to fabricate corresponding planes. If a
LiDAR cloud is accumulated in the reference frame of frame 384, its
corresponding camera ground observation must also be from frame 384.

## CLI (KITTI07 example)

```bash
python -m pip install -e '.[research,dev]'

# handeye.json must have been generated with the same trajectory inputs.
spade-calib refine-ground \
  --handeye outputs/handeye.json \
  --glim /data/kitti07/odom_lidar.txt \
  --sfm /data/kitti07/camera_trajectory_tum.txt \
  --kitti-times /data/kitti07/times.txt \
  --ground-matches /data/kitti07/verified_ground_pairs.json \
  --calib /data/kitti07/calib.txt --camera-index 2 \
  --output outputs/ground_refinement.json
```

Omit `--calib` to run without KITTI reference. Do not reuse the
illustrative JSON as a real measurement. `--handeye-weight`,
`--ground-weight`, `--max-translation-delta-m`,
`--max-rotation-delta-deg` and `--max-initial-normal-angle-deg`
are explicit experimental settings; sweep weights/limits on **training**
data rather than tuning against KITTI test GT.

The output contains `baseline`, `fixed_rotation`,
`optimize_rotation` and (if `--calib` is supplied)
`kitti_evaluation_only` with the **same** KITTI GT baseline/fixed/full
rotation and translation errors. It also includes:
- Initial/final per-match normal angle and offset residual.
- Motion-pair count and weighted motion residual norm change.
- Ground-only and joint Jacobian singular values/rank, with explicit
  parameter-normalization units.
- `bound_hit` flags for suspect solutions.

## Limitations / safeguards

- A **single ground plane constrains translation along its normal**;
  ground-only offset rank is at most 1 for fixed-R 3-DoF optimization.
  Ground normal provides up to 2 independent rotation directions; its
  normal-axis rotation is unobservable from that plane alone.
- Hand-Eye can constrain remaining DOFs but planar UGV motion remains
  ill-conditioned; numerical optimizer convergence is not accuracy proof.
- Measurement sigmas here are empirical weights, not calibrated
  sensor-noise distributions; motion pairs share poses and are correlated.
- The relative-motion pairs are rebuilt from passed trajectories and checked
  against the saved **pair count**, not byte-for-byte identical content.
  Preserve/hash source data for exact experiment reproduction.
- Actual Ground Plane correspondence extraction, DA2 metric-depth fitting,
  outlier detection and non-ground plane refinement are separate missing
  pipeline stages, not silently approximated by this solver.
- Source archive's original optimizer additionally refits camera ground
  when inverse-depth scale/shift changes. That original depth adaptation
  is deliberately **not included** here because SfM scale and camera
  metric planes are fixed in this ablation.
- The output reports per-axis GT errors in Camera coordinates, not in the
  LiDAR/vehicle axes. Consider X/Y/Z separately when concluding improvement.
