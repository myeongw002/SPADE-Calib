# SPADE-Calib

**Sparse Point-cloud Accumulation and Depth-Enhanced LiDAR–Camera Calibration**

Research code for targetless LiDAR–camera extrinsic calibration using
accumulated sparse LiDAR geometry, camera metric-depth and plane
correspondences.

> **Research status:** project bootstrap and legacy D1 **input adapters** are
> available. Calibration, SfM reconstruction, plane extraction, matching and
> LiDAR-refinement algorithms are **not yet ported** as reusable modules.

## Install

Python 3.10+:

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    pytest -q

For optional numerical/vision dependencies:

    pip install -e ".[research,dev]"

## Inspect existing D1 inputs

Unzip the experiment data to a **local path outside this repository**. Copy
[the example dataset config](configs/examples/d1.yaml), then edit
dataset.root to the extracted inputs folder. Run:

    spade-calib inspect --config path/to/local-d1.yaml --output outputs/d1_manifest.json

This reads legacy Joint/PGO XYZ, DA2 arrays, SfM anchors, camera planes/masks
and Hand–Eye matrices; checks their basic validity and writes an input
manifest. It does **not** perform calibration.

Run optional private D1 artifact-shape tests with:

    SPADE_D1_CONFIG=/path/to/local-d1.yaml pytest -q

## Planned architecture

    SfM + GLIM -> Hand–Eye initialization
    LiDAR + trajectory -> deskew -> pose refinement -> accumulated cloud
    Camera image + DA2 + SfM -> metric depth -> ROI-free camera planes
    Camera planes + LiDAR cloud -> normal-diversity guided correspondences
    Geometry + initial transform -> calibration -> held-out evaluation

Older D1 experiments implemented plane-only 6/8-DoF optimization. The later
scene-wide + plane + SfM optimizer is a **separate, not-yet-recovered version**.

## Documentation

- [Legacy D1 adapters](docs/legacy_adapters.md)
- [Coordinate systems](docs/coordinate_systems.md)
- [Data interchange schema](docs/data_formats.md)
- [Recovered code inventory](docs/legacy_inventory.md)
- [Migration plan](docs/migration_plan.md)

## Research limitations

Existing scene-depth consistency results do not establish independent
ground-truth extrinsic accuracy. Plane matching is not guaranteed to identify
the same physical surface. The original Hand–Eye transform and LiDAR PCD frame
must be checked before using them together.

## Data / licensing

No raw recordings, point clouds, SfM reconstructions, original experiment
ZIPs, or private output are tracked. Publication and licensing rights have not
yet been decided; **the repository has no software license at present**.
