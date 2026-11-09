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

## Milestone 2 — modular algorithms (in progress)

### Phase 4 — two selected LiDAR refinement candidates

- **Retained only** Joint 6-DoF Scan Pose Optimization and Point-to-Plane PGO.
- Shared GLIM per-return Deskew, timestamp/scan-ID preservation and optional Target-scan Free-space Filter.
- Implemented reading of saved historical Joint/PGO D1 XYZ and recovery/verification of 10 scan-wise rigid transforms (maximum pointwise fit error about 7 micrometers). Exact final XYZ are retained.
- Removed standalone B/C1 estimator, interpolated C1 correction and its tests/configuration from the selected pipeline. Do not confuse the historic B/C1 ablation with final methods.
- **Open gate:** Obtain or independently implement original Joint/P2Plane-PGO solver; run on D1/D2/D3 raw scans and compare original fixed-reference clouds, corrections and held-out metrics. SVD pose recovery from final XYZ is not solver reproduction.

### Remaining extraction

- ROI-free Camera Plane Extraction.
- Normal Diversity and Camera-guided Plane Matching.
- Plane-only 6/8DoF Calibration Solver.
- Regression gates for an independently implemented joint/PGO optimizer.

## Milestone 3 — unified runner

Config-driven full pipeline, stage outputs, SHA/config caching, logs, artifacts and dataset-independent execution.

## Milestone 4 — research-method restoration

Implement scene-wide depth residual, visibility/occlusion handling, 6/8-DoF ablations, Normal Diversity under equal budgets and D2/D3 common-support transfer evaluation. Analyze Plane Reliability and Extrinsic–Depth Coupling.

Metric-depth agreement alone is not independent ground-truth calibration accuracy.
