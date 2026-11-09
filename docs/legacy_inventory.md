# Recovered legacy Python scripts — dependency inventory

This inventory is based on four D1 experiment archives supplied for migration. **16 distinct script roles** were identified (some exported files are duplicates/aliases). Files below are *not* yet checked into this public repository because the original scripts include absolute paths, D1-specific constants, and experiment-specific data assumptions. The status refers to the uploaded code, not to implemented SPADE-Calib modules.

## A. LiDAR deskew / B-C1 refinement archive (3 scripts)

| Script | Main inputs | Outputs / role | Migration issue |
| --- | --- | --- | --- |
| `deskew_ablation_compa.py` | Raw LiDAR scans, GLIM trajectory, point timestamps | deskewed clouds and ablation utilities | Requires original supporting files for some ablations |
| `compa_B_C1_experiment.py` | deskewed scan data, wall/anchor support, D1 source IDs | per-scan corrections, A/B/C1 clouds, metrics | D1-specific anchor and wall ROI; correction solver differs from the earlier Joint/PGO versions |
| `compa_B_C1_wall_anchor_check.py` | A/B/C1 point clouds and wall support | fixed-wall validation | D1-specific wall reference |

## B. ROI-free plane experiment archive (4 scripts)

| Script | Main inputs | Outputs / role | Migration issue |
| --- | --- | --- | --- |
| `experiment.py` | DA2 metric depth, camera calibration, GLIM scans/trajectory, extrinsic candidate | local-PCA camera plane CSV/masks; early accumulated PCD | Absolute path to an extrinsic file; scan-wise accumulation rather than final deskew pipeline |
| `hard_gate.py` | independent plane match candidates | geometric-gate filter results | exploratory thresholds |
| `vertical_probe.py` | initial LiDAR planes and camera geometry | ground-excluded vertical plane candidates | D1-specific gates |
| `prepare_output.py` | ROI-free experiment outputs | overlays and report images | assumes fixed output layout |

## C. Initial camera-guided archive (4 scripts)

| Script | Main inputs | Outputs / role | Migration issue |
| --- | --- | --- | --- |
| `run_camera_guided.py` | camera plane CSV/masks, raw GLIM accumulation, original Hand–Eye T | camera-guided plane candidates, geometric rank diagnostics | static D1 ID, runtime file paths; rankings do not limit total search |
| `independent_baseline.py` | accumulated PCD and camera planes | independent RANSAC/matching baseline | search budgets and ground handling differ from guided method |
| `anchor_diagnostics.py` | SfM sparse metric anchors, camera plane masks | per-plane anchor statistics | quality diagnostic, not calibrated uncertainty |
| `inspect.py` | original Hand–Eye and depth/point cloud | projection/depth inspection | hard-coded input locations |

An exported alias of `anchor_diagnostics.py` is not counted separately.

## D. Refined PCD plane calibration archive (5 scripts)

| Script | Main inputs | Outputs / role | Migration issue |
| --- | --- | --- | --- |
| `run_camera_guided_refined.py` | `Joint_xyz.npy`/`PGO_xyz.npy`, plane masks, Hand–Eye T | refined-PCD plane candidates, support IDs, matching diagnostics | hard-coded scene ID; environment-variable method choice |
| `run_calibration_refined.py` | matched planes, raw DA2, SfM anchors, initial extrinsic | plane-based 6/8-DoF optimization, results and extrinsic matrices | **No scene-wide depth residual in optimizer**; manually selected plane IDs |
| `verify_refined.py` | PCDs, calibration results, depth, line geometry | held-out projection/depth diagnostics | D1-centric; requires previously generated inputs |
| `extra_diagnostics.py` | optimization output and plane correspondences | Jacobian/observability diagnostics | not a verified joint 8-DoF sufficiency test |
| `make_report.py` | experiment CSV/JSON and visualization | aggregated experiment report | dependent on fixed result layout |

## Missing or not recovered

- Original full Hand–Eye + monocular SfM scale solver (only initial transform/scale results recovered).
- Reusable DA2 inference, SfM reconstruction and metric-alignment pipelines.
- Exact original Joint/GICP+PGO pose-refinement source; PCD outputs are present.
- Latest scene-wide + plane + SfM joint-optimization source (described in the conversation but archive unavailable).
- Latest D2/D3 common-support transfer evaluation source.

## Integration notes

- Do **not** concatenate the scripts or import them directly: many execute file loading and work at module import time.
- Preserve legacy outputs first, then migrate one stage and compare hashes/numeric metrics.
- `run_camera_guided.py` and the separately uploaded `run_camera_guided(2).py` are the same previously inspected algorithm.
- Plane masks are at 320x180 while original RGB/depth data are 1280x720; projection mapping and distortion must remain consistent.
- The older plane-only objective is **not** the latest scene-wide objective.
