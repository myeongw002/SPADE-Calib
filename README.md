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


## D1 LiDAR Refinement (phase 4)

The D1 raw PCD + GLIM A Per-point Deskew, 2026-11-06 B scan-pair 6-DoF Refinement and C1 time-interpolated correction are available as importable modules. Optional target-scan Free-space Filtering is **disabled by default** and is a heuristic, not ground-truth Ghost Point Removal.

~~~bash
pip install -e '.[research,dev]'
python scripts/prepare_d1_refinement.py --archive /private/d1_distance.zip --output outputs/d1_lidar --variant A
# Alternative: --variant B or --variant C1; add --ghost-filter ONLY for ablation
~~~

A reproduced the stored D1 A PCD with exact coordinate equality (215,504 returns) in the local environment. B pose corrections agree within 0.081 mm translation-component difference to saved experiment reference. **B/C1 are NOT historical Joint/PGO.** Read [LiDAR Refinement](docs/lidar_refinement.md) for reproducibility gates and limitations.

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
