# D1 LiDAR refinement — exactly two selected methods

## Selected candidates

Only **Joint 6-DoF Scan Pose Optimization** (method ID `Joint`) and **Point-to-Plane Pose Graph Optimization** (method ID `PGO`) are retained. The final selection follows the original D1/D2/D3 comparisons. Prior A/B/C1 ablation variants, C1 interpolation, sequential refiners and iterative GICP+PGO are **not supported refinement methods** in this module/CLI.

The common front-end is GLIM **per-return Deskew**, and a **target-frame Free-space Consistency Filter** remains optional, OFF by default. Neither is a third pose-refinement candidate.

## What is implemented and what is not

- `scan_io.py`: strict D1 VLP-16 PCD reader, original per-return offsets, scan/ring/source-return identity, private D1 ZIP manifest and GLIM trajectory.
- `trajectory.py`: interpolated `T_world_lidar` from GLIM, quaternion Slerp, no extrapolation.
- `deskew.py`: transform every return to the anchor LiDAR frame with `inv(T_world_lidar_anchor) @ T_world_lidar_return`.
- `refinement.py`: **load the stored final `Joint_xyz.npy` or `PGO_xyz.npy`**; recover ten per-scan post-hoc 6-DoF rigid correction matrices using Kabsch, compare fitted coordinates against saved reference; preserve exact stored XYZ and source-return indices.
- `ghost_filter.py`: optional target-scan Free-space Contradiction test. It is NOT a validated dynamic-object/ghost classifier.

**Not implemented:** an executable reproduction of the original Joint and P2Plane-PGO optimization algorithms. The private 2026-11-07 calibration ZIP contains their **output point clouds**, not the source routines/pose-graph edges/optimizer states. Fitting a rigid transform from original and final XYZ is **not** running either optimizer. The recovered corrections cannot be used as estimators on an unseen sequence.

## Coordinates and reconstruction

All 10 scans were deskewed into Target LiDAR Frame at anchor scan 5. For each source scan \(s\), estimate a proper rigid transform from original Deskew points \(x_{si}\) and final reference points \(y_{si}\) by

\[
\min_{R_s\in SO(3),\,t_s\in\mathbb R^3}
\sum_i \lVert R_s x_{si}+t_s-y_{si}\rVert^2.
\]

This is a **post-hoc Kabsch fit**, not an alternative refinement method. It verifies per-scan correction provenance and is deliberately withheld if source/reference point counts or array ordering differ. Exact reference XYZ is passed downstream, not the slightly rounded approximation.

### Actual D1 reference verification

From the original 10 VLP-16 scans, GLIM per-return Deskew reproduces the archived A cloud at all **215,504** point IDs (exact equality in the earlier local environment). Each selected reference has 215,504 points, in the same inferred source-return order. Per-scan rigid fitting produced:

| Method | Number of fitted scan transforms | Largest per-point residual from stored refined PCD |
| --- | ---: | ---: |
| Joint | 10 | approximately 0.0000071 m (0.0071mm) |
| PGO | 10 | approximately 0.0000072 m (0.0072mm) |

The source identity/order assumption is strongly supported by the micrometer-scale fits, but cannot replace the original optimizer code or explicit index mapping. These numbers reflect stored output representation, **not calibration ground-truth accuracy**.

## Run with the private D1 archives

~~~bash
python -m pip install -e '.[research,dev]'
python scripts/prepare_d1_refinement.py \
  --raw-archive /private/rosbag2_2026_10_08-15_05_54_1791439637016960512_distance.zip \
  --reference-archive /private/COMPA_D1_RefinedPCD_CameraGuided_Calibration_20261107.zip \
  --method Joint --output outputs/d1_joint

python scripts/prepare_d1_refinement.py \
  --raw-archive /private/rosbag2_2026_10_08-15_05_54_1791439637016960512_distance.zip \
  --reference-archive /private/COMPA_D1_RefinedPCD_CameraGuided_Calibration_20261107.zip \
  --method PGO --output outputs/d1_pgo
~~~

To check filtering as a separate ablation, add `--ghost-filter` explicitly. Input ZIPs and raw/refined PCDs must stay out of the public repository. The output contains `d1_<method>_refined.npz`, `d1_<method>_recovered_poses.npz` and a JSON report with source SHA-256 and rigid-fit residuals.

### Tests

Public CI runs synthetic reader, deskew, reference-replay and Free-space tests. Private tests can additionally compare both original methods with the archive:

~~~bash
SPADE_D1_RAW_ZIP=/private/raw.zip \
SPADE_D1_REFERENCE_ZIP=/private/calibration.zip \
python -m pytest -q tests/test_d1_refinement_private.py
~~~

## Important evaluation limitations

1. Applying Free-space Filtering changes the LiDAR point set. Cross-method Projection/Calibration comparisons should use common point IDs or explicitly document differing support.
2. Target-free-space inconsistency is one-sided: a point behind the target surface is not rejected, and no target observation means no judgment. A rejected point is not necessarily a true Ghost Point.
3. The same archived outputs must not be used as both fitting observations and *independent* reference accuracy measures. Exact reproduction of the saved XYZ is not proof of better extrinsics.
4. Restoration of the original Joint and P2Plane-PGO optimizers and D1/D2/D3 end-to-end reproduction remain open. Do not describe phase 4 as full solver reconstruction.