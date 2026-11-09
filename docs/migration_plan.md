# SPADE-Calib migration milestones

## Milestone 0 — bootstrap (merged PR #1)

Installable Python package, frame and plane models, coordinate conventions, synthetic unit tests and legacy code inventory.

## Milestone 1A — D1 input adapters (merged PR #2)

Read Joint/PGO XYZ, DA2, SfM anchors, camera-plane CSV/NPZ and Hand–Eye CSV/YAML via config-driven loaders; inspect file hashes and validity.

## Milestone 1B — numerical reproduction (this PR)

- Unpack the **private original** D1 Calibration archive into an isolated workspace.
- Run unmodified camera-guided Matching and plane-only 6/8-DoF Calibration for Joint and PGO.
- Compare matching IDs, support indices, per-row metrics and 4-fold Extrinsic matrices.
- Document archived ZIP SHA and software environment; exclude private data from public Git.
- Synthetic comparator unit tests and optional full private-archive regression.

**Gate met locally on 2026-11-09:** Joint matched 6 planes and 44 calibrated Extrinsics; PGO 5 planes and 40 Extrinsics. All reproduced original archive outputs exactly. Each new platform must verify independently.

## Milestone 2 — modular algorithms (next)

- Port original ROI-free local-PCA Camera Plane Extraction.
- Port GLIM Deskew and Accumulation; do not conflate B/C1 and exact Joint/PGO PCD generation.
- Port Normal Diversity and camera-guided matching with explicit search budgets and seeds.
- Port historical plane-only 6/8-DoF optimizer without changing its Loss Function.
- Preserve D1 regression metrics through the refactor and add synthetic geometry tests.

**Gate:** newly ported modules reproduce the same numerical outputs, not merely the legacy scripts.

## Milestone 3 — unified runner

Config-driven full pipeline, stage outputs, SHA/config caching, logs, artifacts and dataset-independent execution.

## Milestone 4 — research-method restoration

Implement scene-wide depth residual, visibility/occlusion handling, 6/8-DoF ablations, Normal Diversity under equal budgets and D2/D3 common-support transfer evaluation. Analyze Plane Reliability and Extrinsic–Depth Coupling.

Metric-depth agreement alone is not independent ground-truth calibration accuracy.
