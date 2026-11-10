# Trajectory Observability and Scene Selection (phase 5)

This module selects **candidate temporal windows** from a whole recording; it does
**not** estimate final LiDAR–camera calibration. It is built on the recording
indexer introduced in PR #5, and preserves historical D1 legacy readers.

## Standard command (full source recording)

```bash
spade-calib select-scenes --config configs/local-recording.yaml \
  --window-s 30 --window-stride-s 15 \
  --pair-baseline-s 2 --pair-stride-s 1 \
  --max-scenes 4 --output outputs/scene_selection.json
```

This reads the standard schema-v2 recording config, indexes real RGB/PCD file
paths, matches each image to its COLMAP pose and LiDAR frame using the explicit
clock policy, and interpolates the GLIM trajectory to camera timestamps. A
candidate is eligible only if at least 10 raw PCD scan timestamps occur in its
window and the maximum time gap, **including the scene boundaries**, is at
most 1.5 s. These values can be customized through `SelectionSettings` in the
Python API. The indexer does not decode the PCD point data at this stage.

## Trajectory-only mode (e.g. historical user uploads)

```bash
spade-calib select-scenes --glim /private/traj_lidar.txt \
  --sfm /private/camera_poses.csv --output outputs/trajectory_only.json
```

This mode deliberately does **not** assert that the required RGB and raw PCD
files exist for selected windows. Each selected row sets
`files_verified_by_index=false`; a scene cannot yet be run end-to-end.

## Pose and time conventions

- GLIM: `timestamp_seconds tx ty tz qx qy qz qw`, with `T_world_lidar` and
  translation in **meters**.
- COLMAP/pycolmap: `images.bin`, `images.txt` or `camera_poses.csv`; COLMAP
  `qvec/tvec` is `T_camera_from_sfm_world` with translation in **unknown SfM units**.
- A camera timestamp in `image_name` is an integer nanosecond epoch timestamp;
  `camera_to_lidar_offset_ns` is added to it to query GLIM, and **not estimated**.
- Relative motions are `A = inverse(T_world_lidar(t_i)) @ T_world_lidar(t_j)`
  and `B = inverse(T_sfm_world_camera(t_i)) @ T_sfm_world_camera(t_j)`.
- The provisional transform obeys `A X = X B(s)` with `X=T_lidar_camera`.
  A global rotation is fitted from relative rotation vectors, and a provisional
  positive metric scale `s` from least-squares translation constraints. The
  fit is **diagnostic only**, not a validated extrinsic.

## Observability proxy: what it means and what it does NOT mean

For each relative-motion pair, let `Ra`, `ta` denote LiDAR relative motion and
`Rb`, `tb` denote camera/SfM relative motion. With provisional rotation `R_X`
and common metric scale `s`, the linear translation-and-log-scale constraint is

`(Ra - I) * t_X - R_X * (s * tb) + ta = 0`.

The **rotation-excitation block** uses `(Ra-I)` and the **translation/scale
block** uses `[(Ra-I)*L, -R_X*(s*tb)]`, where `L=1m` by default makes the
translation perturbation dimensionless. The reported 7x7 block-diagonal
information `H=(J^T J)/num_pairs` is a **motion-diversity surrogate**. It is
**not a statistically calibrated Fisher Information Matrix** of the final
plane/depth residual, has no sensor noise weights, omits rotational-to-
translation cross-blocks, and depends on the choice of `L` and log-scale
parameterization. DOF weakness is reported as 3D/4D eigenvectors and normalized
singular values, not as a claim that an individual Euler axis is unobservable.

Candidate scenes are fixed physical-time windows with a configurable stride,
relative motion baseline and subsampling. Candidates must have enough motion
pairs. A deterministic greedy D-optimal-style score maximizes
`log det(I + sum(H_scene)/regularizer)` while preventing temporal overlap.
The regularizer is **heuristic**, so scene ranking must be verified by
geometry, matching and downstream calibration. An evenly sampled fixed pair
budget is used to reduce preference for simply longer recordings. Overlap of
relative-motion observations means these are NOT i.i.d. Fisher samples.

## Limitations and next gates

- This module uses *all* available trajectory poses to estimate its provisional
  rotation and SfM scale, so selection is **not independent held-out evaluation**.
- Observability is not confirmed by camera/LiDAR common field of view, planar
  support, return density, ROI or depth quality. Those checks belong in the
  next geometry-validation stage.
- Scene boundaries are metadata timestamps; full per-return LiDAR deskew
  and pose interpolation checks occur in the subsequent Refinement pipeline.
- Failed/partial COLMAP registration cannot be imputed. Multi-model COLMAP
  directories must be disambiguated upstream.
- No per-Scene Calibration, automatic initial Hand-Eye for downstream use,
  or full end-to-end Runner is implemented.