# SPADE-Calib

**Sparse Point-cloud Accumulation and Depth-Enhanced LiDAR–Camera Calibration**

SPADE-Calib is an **in-development research framework** for targetless LiDAR–camera extrinsic calibration. The intended pipeline combines sparse LiDAR scan accumulation, camera metric-depth alignment, camera-guided geometric correspondences and extrinsic optimization.

> **Status — project bootstrap:** only the foundational data models, documentation and tests are implemented. The original D1 experimental scripts have **not** yet been ported or published. Do not interpret the proposed pipeline as a validated end-to-end implementation.

## Research pipeline (planned)

```text
SfM trajectory + LiDAR trajectory --> Hand–Eye initialization
LiDAR scans + trajectory ---------> per-point deskew -> refined accumulated cloud
Camera image + depth + SfM -------> metric depth -> ROI-free camera planes
Accumulated cloud + camera planes + T_init
                                --> camera-guided plane correspondences
Accumulated cloud + metric depth + planes + SfM anchors
                                --> extrinsic calibration -> evaluation
```

Existing D1 experiments implemented plane-based 6/8-DoF optimization. A later, currently unrecovered experiment introduced scene-wide depth consistency in addition to plane and SfM residuals. These are **different experimental generations** and will be migrated separately.

## Current contents

- `src/spade_calib/io/models.py`: validated, in-memory camera, transform, point-cloud and plane representations.
- `docs/coordinate_systems.md`: frame convention and depth semantics.
- `docs/data_formats.md`: provisional on-disk interchange format.
- `docs/architecture.md`: module boundaries and status.
- `docs/legacy_inventory.md`: 16 recovered legacy Python scripts, grouped by experiment.
- `docs/migration_plan.md`: staged migration and reproduction gates.
- `tests/`: synthetic, data-free unit tests.

No raw datasets or unpublished experiment ZIP files are committed.

## Development setup

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

Optional research dependencies:

```bash
python -m pip install -e ".[research,dev]"
```

There is **no runnable `spade-calib` CLI yet**. Configuration samples are design drafts rather than executable experiments.

## Reproducibility and claims

A numeric decrease in scene-depth residual is **not** ground-truth extrinsic accuracy. The D1/D2/D3 results in the research history require a documented dataset split, identical evaluation support, and independent extrinsic reference before calibration accuracy can be claimed.

See [migration plan](docs/migration_plan.md) for the next engineering tasks.

## Publication and licensing

Code/data publication rights are being reviewed. **No license has been selected yet**; public visibility does not imply a license to reuse the repository's content.
