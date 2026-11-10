# D1 LiDAR Refinement — selected Joint 6-DoF / Point-to-Plane PGO

## Scope and primary source

Two selected algorithms only, ported from the **actual optimization source** in the user-provided `COMPA_D1_Refinement_Methods_B_Fixed_20261106.zip`:

- Joint: `COMPA_BC1_multimethod_test/COMPA_D1_B_C1_Repro_Package_20261106/scripts/compa_B_C1_experiment.py`, functions `sample_model`, `match_pairs`, `solve_step`, `ptransform` (B correction only; C1 interpolation is not a supported method).
- P2Plane PGO: `COMPA_BC1_multimethod_test/retest_pgo.py`, functions `pair_estimation`, `graph_solve`, `to_center`; source scans and matched pairs are shared with Joint.
- Common GLIM per-return deskew from `deskew_ablation_compa.py` and strict binary D1 scan ZIP reader.

Historical Joint/PGO source was *newly recovered*. These solvers are **not GLIM's internal scan optimization**. Both use one rigid correction per GLIM-deskewed scan, anchor scan 5 fixed; PGO uses *small-angle* relative pose differences and an anchored graph solve, not a fully nonlinear Lie-group pose graph.

## Executable pipeline (no refined reference required)

~~~bash
python -m pip install -e '.[research,dev]'
python scripts/prepare_d1_refinement.py --raw-archive /private/rosbag2_2026_10_08-15_05_54_1791439637016960512_distance.zip --method Joint --output outputs/d1_joint
python scripts/prepare_d1_refinement.py --raw-archive /private/rosbag2_2026_10_08-15_05_54_1791439637016960512_distance.zip --method PGO --output outputs/d1_pgo
~~~

Optional numerical regression comparison is performed **after** solving, never as solver input:

~~~bash
python scripts/prepare_d1_refinement.py --raw-archive /private/d1_distance.zip --reference-archive /private/COMPA_D1_RefinedPCD_CameraGuided_Calibration_20261107.zip --method Joint --output outputs/d1_joint_checked
~~~

Optional `--ghost-filter` rejects accumulated source returns that lie in current target-scan free space; **OFF by default**. Point IDs and scan IDs survive in output NPZ. The filter does not label or remove all real-world ghost points.

## D1 numerical checks against archived calibration inputs

D1: 10 raw scans, 215,504 valid returns, fixed anchor scan 5, pairset 22, matching 750 per pair (16,500 correspondences per Joint iteration). The archived PCD is read only after the optimizer terminates. Results in the reproducibility environment (2026-11-10):

| Solver | Maximum same-index pointwise 3D deviation | Mean deviation | 95th percentile |
|---|---:|---:|---:|
| Joint 6-DoF | 0.7134 mm | 0.02695 mm | 0.12954 mm |
| P2Plane PGO | 0.5571 mm | 0.00591 mm | 0.03635 mm |

The exact original numeric result is **not bitwise identical**. The 0.557 mm PGO maximum is dominated by one scan; no claim of true physical pose accuracy is made. If a 1mm tolerance is selected for historical output regression, it is a research/software acceptance threshold, **not** a guarantee of calibration accuracy.

## Implementation details

1. `scan_io` reads the original 22-byte binary VLP-16 PCD with original scan ID, ring, time offset and return position; `trajectory` interpolates `T_world_lidar` by Slerp; `deskew` transforms points per return into scan-5 LiDAR coordinates.
2. Joint: voxel-sample points, estimate target normals with 24-neighbor PCA, match 22 neighboring/reference scan pairs with normal cosine and distance gates, solve 54 correction parameters (10 scans, fixed scan 5) by robust point-to-plane linearized IRLS with Tikhonov and adjacent-scan smoothness, 5 outer iterations.
3. PGO: pairwise robust bounded 6-DoF least squares on **frozen** point-to-plane correspondences; solve relative-pose graph with bounded robust confidence and weak priors; apply one corrected pose per scan.
4. `T_refined_lidar_from_deskewed_lidar` is stored in target-LiDAR coordinates, with camera extrinsic optimization occurring later. The optimized D1 data is not obtained by post-hoc Kabsch fitting to a known refined output.
5. D1 training excludes a historical wall ROI `[10.3,14.9] x [4.3,9.4] x [-1.2,4.6]` and crops the scan to an additional D1-specific region. This is a documented *experiment setup*, not generally valid Calibration logic. `--disable-d1-roi` changes this setup for exploratory scenes.

## Limits / remaining gates

- Validate D2 and D3 with their own raw scans + trajectories and compare historical source outputs. The provided CLI currently parses D1 archive layout; generic scene loader is future work.
- The within-scene metric is not independently measured ground truth. There is no independent extrinsic or scan-pose GT.
- Small-angle relative PGO is not exactly SE(3) geodesic residual optimization.
- Neither method uses 3D ghost removal in the pose optimizer itself; it is a separate ablation, disabled by default and applied after refinement.
- Changing point sampling, ROI bounds, randomness or SciPy/Numpy versions can change pointwise outputs. The original 2026-11-06 numerical script is preserved in the private source archive.