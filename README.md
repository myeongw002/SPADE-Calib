# SPADE-Calib

**Sparse Point-cloud Accumulation and Depth-Enhanced LiDAR–Camera Calibration**

Targetless LiDAR–Camera calibration research code using accumulated sparse LiDAR Geometry, camera metric depth and plane correspondences.

> **Status:** Installable Python package, config-driven D1 legacy readers and **numerically reproduced historical D1 Matching / plane-only 6/8-DoF Calibration**. Selected D1 Joint 6-DoF / Point-to-Plane PGO solvers are ported and numerically checked; Camera/Calibration solver modules remain unported.

## Installation

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
~~~

For numeric/vision dependencies also install with pip install -e '.[research,dev]'.

## Inspect D1 inputs

Extract private data **outside** the repository. Copy configs/examples/d1.yaml and set dataset.root to the extracted inputs directory.

~~~bash
spade-calib inspect --config path/to/local-d1.yaml \
  --output outputs/d1_manifest.json
~~~

This inspects point clouds, SfM/DA2, camera plane files and Hand–Eye inputs. It does not estimate Calibration.

## Reproduce original D1 plane-only Calibration

With an authorized **trusted original** 2026-11-07 experiment ZIP:

~~~bash
python scripts/reproduce_d1.py \
  --archive /private/COMPA_D1_RefinedPCD_CameraGuided_Calibration_20261107.zip \
  --output outputs/d1_reproduction_report.json
~~~

Original legacy scripts are executed in a temporary private directory and all Matching, support indices, results CSV, and 4-fold Extrinsic matrices are compared to their saved archival references. This tool **executes included Python source: do not use untrusted archives**.

Local results: **Joint 6 matches / 44 Calibration rows; PGO 5 matches / 40 Calibration rows. All numeric differences 0.0.** The private ZIP is not included in GitHub; public CI uses synthetic comparison tests. See docs/d1_reproduction.md.


## D1 LiDAR Refinement (phase 4 — selected methods only)

**Joint 6-DoF** and **Point-to-Plane PGO** optimize from raw D1 LiDAR scans and GLIM trajectory without loading a refined reference PCD. The algorithms were ported from the recovered COMPA_D1_Refinement_Methods_B_Fixed_20261106 source, retaining only two selected methods, with a shared GLIM per-return deskew.

~~~bash
pip install -e '.[research,dev]'
python scripts/prepare_d1_refinement.py --raw-archive /private/d1_distance.zip --method Joint --output outputs/d1_joint
python scripts/prepare_d1_refinement.py --raw-archive /private/d1_distance.zip --method PGO --output outputs/d1_pgo
~~~

Add `--reference-archive /private/d1_calibration.zip` for numerical regression only: reference point clouds are compared **after** optimization and are never passed into either solver.

Local D1 (10 scans, 215,504 returns) maximum same-return deviations against archived calibration XYZ: **Joint 0.713 mm, PGO 0.557 mm**. Both optimize over 22 scan pairs. Free-space ghost filtering is optional and OFF by default. No C1, sequential, or iterative GICP+PGO method is exposed.

The D1 preset includes a historical ROI exclusion and anchor scan 5. PGO uses a regularized **small-angle relative pose graph**, not a nonlinear Lie-group optimizer. This numerical reproduction does not independently demonstrate physical calibration accuracy. See [LiDAR refinement](docs/lidar_refinement.md) for limitations.

## Whole-recording data loader and indexer

This new `schema_version: 2` interface indexes **all** RGB images, raw timestamped LiDAR scans, GLIM trajectory and pycolmap/COLMAP registered camera poses from recording paths, before selecting D1/D2/etc. It does **not** run DA2, GLIM, Calibration or scene observability analysis. The existing legacy D1 `spade-calib inspect` (`schema_version: 1`) remains intact.

~~~bash
cp configs/examples/recording.yaml configs/local-recording.yaml
# Edit dataset.root and the four source paths in local-recording.yaml.
spade-calib index --config configs/local-recording.yaml --output outputs/recording_index.json
~~~

It supports timestamp-nanosecond filenames or explicit CSV timestamp maps, explicit camera-LiDAR clock offset, nearest-scan pairing with tolerance, raw PCD lazy loading, and COLMAP `images.bin`/`images.txt` or `camera_poses.csv`. **SfM scale and camera–LiDAR clock offset are never inferred.** Only metadata is indexed; unregistered images and unmatched frames are reported rather than silently discarded. See [Whole-recording input contract](docs/recording_index.md).

## Whole-recording Trajectory Observability and Scene Selection (phase 5)

The `spade-calib select-scenes` CLI takes the existing **schema_version 2 full-recording config**, synchronizes registered pycolmap camera poses with GLIM timestamps using the explicit recorded clock offset, proposes fixed-duration candidate scenes, calculates a documented **motion observability proxy** for Hand–Eye plus unknown SfM scale, and greedily selects **non-overlapping** complementary windows. It preserves the `inspect` and `index` commands.

~~~bash
pip install -e '.[research,dev]'
spade-calib select-scenes --config configs/local-recording.yaml \
  --window-s 30 --window-stride-s 15 --pair-baseline-s 2 \
  --max-scenes 4 --output outputs/selected_scenes.json
~~~

For legacy trajectories without all original RGB/PCD files, run **trajectory-only** mode. It explicitly marks selected windows as **not validated runnable scenes**:

~~~bash
spade-calib select-scenes --glim /private/traj_lidar.txt \
  --sfm /private/camera_poses.csv --output outputs/trajectory_only.json
~~~

For complete recordings, a candidate needs continuous raw LiDAR scan timestamp coverage (default >=10 scans, max 1.5s gap including boundaries); RGB/SfM registration and raw PCD paths are indexed. This stage does NOT compute final extrinsic, plane correspondences, or a statistically calibrated 6/8-DoF FIM. Observability metrics depend on normalization and regularization and must be checked against final geometry/calibration. See [trajectory Scene selection](docs/scene_selection.md).

## Full-recording 7-DoF Hand–Eye and uncertainty analysis

`spade-calib handeye` fits a **single** `T_lidar_camera` + SfM metric scale from the entire synchronized GLIM/COLMAP trajectory, then reports full-coupling Jacobian observability, an **assumed-noise model-based** 7×7 joint covariance and its 6×6 marginalized Extrinsic covariance, per-DOF tangent standard deviations and coupled weak directions. Optional temporal block bootstrap estimates **recording subset stability**, not true sensor covariance. It does not run calibration on camera/LiDAR planes or turn the estimated covariance into a Prior automatically.

~~~bash
pip install -e '.[research,dev]'
spade-calib handeye --config configs/local-recording.yaml --output outputs/handeye.json --bootstrap-resamples 96
# Historical trajectory-only data (raw PCD and RGB unavailable):
spade-calib handeye --glim /private/traj_lidar.txt --sfm /private/camera_poses.csv --output outputs/handeye.json --bootstrap-resamples 96
~~~

See [Full-recording Hand–Eye and uncertainty analysis](docs/handeye_uncertainty.md).

## KITTI vs custom timestamps (shared solver)

Dataset-specific filename/time rules are handled in input adapters. Hand–Eye,
observability, scene selection and future geometry stages use a shared pose
representation and **do not branch on dataset names**.

- **Custom recordings:** use nanosecond image/PCD filenames, or declare
  `image_timestamps.csv` / `lidar_timestamps.csv` in `recording.yaml`.
  The full-recording Hand–Eye command now respects these explicit mappings.
- **KITTI Odometry:** keep `000000.png` and `times.txt`. An explicit frame-index
  adapter matches GLIM *odom rows* and registered COLMAP frames without treating
  old synthetic 0.1-second GLIM timestamps as actual KITTI times.

~~~bash
spade-calib handeye --glim /data/kitti07/odom_lidar.txt \\
  --sfm /data/kitti07/colmap/sparse/0 \\
  --kitti-times /data/kitti07/times.txt \\
  --output outputs/kitti07_handeye.json

# Pose-only historical exports are also supported when every frame is present:
spade-calib handeye --glim /data/kitti07/odom_lidar.txt \\
  --sfm /data/kitti07/camera_trajectory_tum.txt \\
  --kitti-times /data/kitti07/times.txt \\
  --output outputs/kitti07_tum_handeye.json
~~~

The public, projection-ready Extrinsic is **`T_camera_lidar`**
(LiDAR → Camera). The current 7-DoF Hand–Eye internal optimization still uses
its inverse `T_lidar_camera`; both are included in the result JSON. Uncertainty
coordinates are currently in the LiDAR tangent frame and must be transformed
before claiming Camera-axis uncertainty.

This phase covers trajectory inputs and timestamp matching. KITTI raw `.bin`
clouds are **not** accepted by the existing PCD refinement loader; they require
a separate raw-cloud adapter before full point-cloud refinement. See
[dataset adapter and timestamp contract](docs/dataset_adapters.md).

## Intended Pipeline

    SfM + GLIM -> Hand–Eye Initialization
    LiDAR scans -> Deskew -> Pose Refinement -> Accumulated PCD
    Camera + DA2 + SfM -> Metric Depth -> ROI-free Camera Plane Extraction
    Camera and LiDAR Geometry -> Normal Diversity + Guided Correspondences
    Initial Extrinsic + observations -> Calibration -> Evaluation

The archived 2026-11-07 baseline is **plane-only**. The newer scene-wide + plane + SfM Joint Optimizer remains unrecovered.

## Documentation

- docs/d1_reproduction.md — Verified local reproduction and limitations
- docs/lidar_refinement.md — Selected LiDAR refinement solvers, reproduction and limits
- docs/recording_index.md — Whole-recording dataset metadata and timestamp mapping
- docs/scene_selection.md — Trajectory observability proxy, selection and failure gates
- docs/handeye_uncertainty.md — Full-trajectory Hand–Eye, joint covariance and bootstrap assumptions
- docs/legacy_adapters.md — Implemented D1 input readers
- docs/coordinate_systems.md — Frames, depth and plane conventions
- docs/data_formats.md — Provisional data schema
- docs/legacy_inventory.md — Experiment script inventory
- docs/migration_plan.md — Future development gates

## Limitations and permissions

Output reproducibility does not prove Extrinsic accuracy or correct physical plane correspondences. Camera masks are not independent of the original SfM depth alignment.

The raw ZIP, point clouds, imagery and private outputs are not tracked. The repository currently has no software license; public visibility alone is not permission to reuse private source or data.
