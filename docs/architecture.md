# Architecture and implementation status

The repository has two distinct purposes: **preserve/reproduce older D1 experiments** and **implement a reusable calibration library**. An unverified experimental result must not silently become an API assumption.

## Logical flow

1. Trajectories: SfM and GLIM -> synchronized relative motions -> Hand–Eye initial T and SfM scale.
2. LiDAR: raw scans + timestamps + trajectory -> deskew -> optional scan-pose refinement -> accumulated point cloud.
3. Camera: image + DA2 raw prediction + SfM depth anchors -> metric-depth alignment -> ROI-free local-plane extraction.
4. Correspondence: ranked camera planes -> projected LiDAR search -> RANSAC/connected-component gates -> candidate plane matches.
5. Calibration: initial transform + scene/depth/plane/SfM constraints -> 6-DoF or 8-DoF optimization.
6. Evaluation: held-out data, fixed common LiDAR support, cross-scene projection and independent extrinsic reference (if available).

## Code boundaries

- `io/`: typed models and serialization; no optimization.
- `lidar/`: registration and accumulation; does not read camera files.
- `camera/`: depth and plane construction; does not read LiDAR files.
- `correspondence/`: accepts objects, not fixed filenames or scene IDs.
- `calibration/`: residuals, parameterizations and solver.
- `evaluation/`: never trains, refits or selects hyperparameters on test data.

Pipeline orchestration belongs in a future CLI/runner, not module import side effects.

## Current state

Only `io/models.py` and synthetic unit tests exist as reusable package code. No end-to-end pipeline or CLI is available. Original experiments are inventoried in [legacy_inventory.md](legacy_inventory.md) but have not been published.

## Known research constraints

- The current Camera Depth representation is **camera-Z metric depth**, not radial range.
- A matching gate passing is **not** proof of a physical plane correspondence.
- Scene-depth consistency is not independently verified 6-DoF extrinsic accuracy.
- Current Normal Diversity ranks camera plane normals; it is not a full extrinsic FIM.
- Optimizing 8 variables (extrinsic + depth scale/shift) may yield coupled/weakly observable directions.

See [migration_plan.md](migration_plan.md) for ordered engineering work.
