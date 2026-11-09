# SPADE-Calib

**Sparse Point-cloud Accumulation and Depth-Enhanced LiDAR–Camera Calibration**

Targetless LiDAR–Camera calibration research code using accumulated sparse LiDAR Geometry, camera metric depth and plane correspondences.

> **Status:** Installable Python package, config-driven D1 legacy readers and **numerically reproduced historical D1 Matching / plane-only 6/8-DoF Calibration**. LiDAR D1 A/B/C1 modules are now available, but historical Joint/PGO production and Camera/Calibration solver modules remain unported.

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

Selected refiners: **Joint 6-DoF** and **Point-to-Plane PGO**. There are no B/C1, interpolated correction, or iterative GICP+PGO variants in the executable pipeline. The common GLIM per-return Deskew and optional target-scan Free-space Filter remain.

**Current capability is historical D1 result replay and per-scan rigid-correction recovery, NOT restoration of the original Joint/PGO optimization algorithms.** Original archived refined XYZ are required until the source optimizers have been recovered or reimplemented and independently checked.

~~~bash
python scripts/prepare_d1_refinement.py \
  --raw-archive /private/d1_distance.zip \
  --reference-archive /private/d1_calibration.zip \
  --method Joint --output outputs/d1_joint

python scripts/prepare_d1_refinement.py \
  --raw-archive /private/d1_distance.zip \
  --reference-archive /private/d1_calibration.zip \
  --method PGO --output outputs/d1_pgo
~~~

On original D1 (10 scans; 215,504 points), post-hoc per-scan rigid fitting to the archived output has a worst-case residual of about **0.0072 mm**. This does **not** measure optimizer reproducibility. See [LiDAR refinement notes](docs/lidar_refinement.md) for limitations.

## Intended Pipeline

    SfM + GLIM -> Hand–Eye Initialization
    LiDAR scans -> Deskew -> Pose Refinement -> Accumulated PCD
    Camera + DA2 + SfM -> Metric Depth -> ROI-free Camera Plane Extraction
    Camera and LiDAR Geometry -> Normal Diversity + Guided Correspondences
    Initial Extrinsic + observations -> Calibration -> Evaluation

The archived 2026-11-07 baseline is **plane-only**. The newer scene-wide + plane + SfM Joint Optimizer remains unrecovered.

## Documentation

- docs/d1_reproduction.md — Verified local reproduction and limitations
- docs/legacy_adapters.md — Implemented D1 input readers
- docs/coordinate_systems.md — Frames, depth and plane conventions
- docs/data_formats.md — Provisional data schema
- docs/legacy_inventory.md — Experiment script inventory
- docs/migration_plan.md — Future development gates

## Limitations and permissions

Output reproducibility does not prove Extrinsic accuracy or correct physical plane correspondences. Camera masks are not independent of the original SfM depth alignment.

The raw ZIP, point clouds, imagery and private outputs are not tracked. The repository currently has no software license; public visibility alone is not permission to reuse private source or data.
