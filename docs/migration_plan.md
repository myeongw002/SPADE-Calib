# SPADE-Calib migration milestones

## Milestone 0 — bootstrap (merged PR #1)

Installable Python package, frame and plane models, coordinate conventions, synthetic unit tests and legacy code inventory.

## Milestone 1A — D1 input adapters (merged PR #2)

Read Joint/PGO XYZ, DA2, SfM anchors, camera-plane CSV/NPZ and Hand–Eye CSV/YAML via config-driven loaders; inspect file hashes and validity.

## Milestone 1B — numerical reproduction (merged PR #3)

- Unpack the **private original** D1 Calibration archive into an isolated workspace.
- Run unmodified camera-guided Matching and plane-only 6/8-DoF Calibration for Joint and PGO.
- Compare matching IDs, support indices, per-row metrics and 4-fold Extrinsic matrices.
- Document archived ZIP SHA and software environment; exclude private data from public Git.
- Synthetic comparator unit tests and optional full private-archive regression.

**Gate met locally on 2026-11-09:** Joint matched 6 planes and 44 calibrated Extrinsics; PGO 5 planes and 40 Extrinsics. All reproduced original archive outputs exactly. Each new platform must verify independently.

## Milestone 2 — modular algorithms (in progress)

### Phase 4 — two selected LiDAR Refinement solvers (D1 implemented)

- **Joint 6-DoF:** robust point-to-plane matching, IRLS, scan-level 6-DoF corrections from `compa_B_C1_experiment.py` (B correction only; C1 is not selected).
- **P2Plane PGO:** robust pairwise 6-DoF relative correction estimates and anchored small-angle pose graph from `retest_pgo.py`.
- Both share GLIM per-return deskew, original scan/return IDs, and optional post-refinement target-scan free-space consistency filtering (OFF by default).
- Actual D1 optimizer execution: 10 scans; 215,504 returns; 22 scan pairs. Maximum same-return difference from archived final XYZ: **Joint 0.7134 mm / PGO 0.5571 mm**.
- Previous post-hoc Kabsch transform recovery is removed from the operational pipeline. The archive may be read only after optimization to calculate regression statistics.
- D1 ROI and wall exclusion are historically fixed and documented; PGO uses a linearized small-angle pose graph, not a nonlinear SE(3) graph objective.

**Remaining gates:** independently validate D2/D3 with raw scans; generalize archive/ROI config; quantify point-support and scene transfer. D1 PCD matching does not prove independent calibration accuracy.

### Remaining extraction

- ROI-free Camera Plane Extraction.
- Normal Diversity and Camera-guided Plane Matching.
- Plane-only 6/8-DoF Calibration Solver.


## Milestone 2B — common recording inputs and indexing (PR #5, stacked on PR #4)

- New schema_version 2 loads full recording paths (RGB, raw PCD, GLIM, COLMAP), retaining schema_version 1 legacy D1 inspector.
- Deterministic timestamps and nearest LiDAR/image mapping with explicit offset and tolerance; no guessed synchronization.
- Read registered COLMAP camera poses; never assume SfM metric scale or matching world frames.
- Lazy raw PCD scan loading with per-return times and provenance; metadata JSON manifest.
- **Not yet implemented:** observability analysis, scene segmentation/selection, camera calibration parsing, an automatic scene runner, and complete end-to-end Calibration.

## Milestone 2C — Trajectory Observability and Scene Selection (PR #6)

- Synchronize GLIM and COLMAP registered camera trajectories at timestamps, no implicit world-frame or clock-offset alignment.
- Propose fixed-duration scene windows, compute 3x3 rotation-excitation and 4x4 translation + unknown-SfM-scale Jacobian information proxies.
- Document the 7x7 matrix as **a motion-excitation surrogate, not the final plane/depth FIM**, and report weak eigen-directions rather than naming independent Euler angles.
- Select up to K non-overlapping windows using deterministic greedy log-det gain.
- In whole-recording mode, require raw LiDAR timestamp coverage; trajectory-only mode remains explicitly non-executable without images/scans.
- Actual archived 2026-10-08 trajectories: 4,747 synchronized poses, 30 candidate windows, 4 selected trajectory windows; the archived 3×10 raw LiDAR PCDs do not fully cover any 30-second selected scene.
- Still needed: Geometry-based scene validation, refinement on selected windows, end-to-end Hand–Eye and plane/depth Calibration, independent held-out calibration accuracy evaluation.

## Milestone 2D — Whole-trajectory Hand–Eye and conditional Extrinsic uncertainty (PR #7)

- Optimize `T_lidar_camera` (6-DoF) and positive SfM metric scale (1-DoF) jointly on all synchronized GLIM/COLMAP relative motions.
- Use the full 7x7 residual Jacobian with rotation/translation/scale coupling; report numerical rank, conditioned weak modes and tangent-frame per-DOF diagnostics.
- If rank=7, calculate the **conditional assumed-noise** joint covariance and correctly marginalized 6x6 Extrinsic covariance; reject finite covariance for rank-deficient fits.
- Optionally refit 20-second temporal blocks (bootstrap) to quantify within-recording subset stability; report rejections and rank failures.
- Local historical trajectory experiment: 4,747 synchronized poses, 477 relative motions, fitted SfM scale ≈9.2224; LiDAR Z Translation has the largest normalized Extrinsic uncertainty; first/second trajectory half Z Translation results differ by ≈0.403 m. Do not equate conditional covariance or bootstrap to ground-truth accuracy.
- **Not included:** geometric scene information, a calibrated pose-noise model, a statistically validated prior, automatic timing synchronization or final depth/plane calibration.

## Milestone 3 — unified runner

Config-driven full pipeline, stage outputs, SHA/config caching, logs, artifacts and dataset-independent execution.

## Milestone 4 — research-method restoration

Implement scene-wide depth residual, visibility/occlusion handling, 6/8-DoF ablations, Normal Diversity under equal budgets and D2/D3 common-support transfer evaluation. Analyze Plane Reliability and Extrinsic–Depth Coupling.

Metric-depth agreement alone is not independent ground-truth calibration accuracy.
