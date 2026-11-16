# KITTI07 GT-pose Reference Accumulation (Oracle Baseline)

This is an evaluation baseline, **not a camera–LiDAR extrinsic calibration algorithm**. It compares scan-selection strategies using KITTI odometry GT-pose accumulation (oracle reference), GLIM pose accumulation, and optionally GLIM+Joint. KITTI GT poses are not error-free surface ground truth.

## Dependencies and input files

Install with: python -m pip install -e '.[research,dev]'

Required:
- KITTI Odometry sequences/07/velodyne/*.bin (use the same physical beam configuration as the GLIM trajectory)
- KITTI Odometry sequences/07/times.txt
- KITTI Odometry sequences/07/calib.txt (Tr = LiDAR -> rectified camera 0)
- KITTI Odometry poses/07.txt (camera 0 -> world, 3x4 per frame)
- GLIM odom_lidar.txt (Nx8: timestamp, xyz, qxyzw, with verified row-index mapping)

From the repository root:

    python scripts/experiments/run_oracle_accumulation.py \
      --velodyne /path/to/sequences/07/velodyne \
      --times /path/to/sequences/07/times.txt \
      --odom /path/to/glim/odom_lidar.txt \
      --poses /path/to/poses/07.txt \
      --calib /path/to/sequences/07/calib.txt \
      --window 012 --target-frame 898 \
      --half-frames 5 --radii-m 5 --match-scan-budget \
      --selection-source glim --refiner none \
      --roi 0 40 -20 20 -3 6 --voxel-m 0.5 \
      --out results/kitti07_oracle_012

ROI is **mandatory**, in the **target LiDAR frame**, and is held fixed for all methods. The example ROI is illustrative and not a validated camera-visible region. Inspect the target scene, define the desired ROI, then sweep voxel size e.g. 0.25/0.5/1.0 m.

Optional Joint refinement: add --refiner joint --refine-iterations 4 --refine-voxel-m 0.16 --refine-max-samples 2500.

Selection controls:
- --selection-source glim: fixed-radius and distance-budget policy from estimated trajectory, as in the existing pilot.
- --selection-source gt: idealized selection policy using GT positions. Report it separately.
- --match-scan-budget: closest N poses with N equal to the Fixed Frame scan count, **NOT** a fixed-radius policy. Use this to control data budget.

The two KITTI windows are 005 (37.5-52.5 s, target 384) and 012 (90-105 s, target 898). GLIM row j is assumed to correspond to KITTI frame j + pose_index_offset, **not** nearest recorded GLIM timestamp. This is dataset-specific and must be verified for new logs.

## Pose conventions

T_W_L[i] = T_W_C0[i] @ T_C0_L
T_Lk_Li = inverse(T_W_L[k]) @ T_W_L[i]

Use target-frame relative poses for the GLIM and GT comparisons (the two odometry trajectories need not share a world frame). The Joint solver returns per-scan rigid corrections in the target LiDAR frame, so T_Lk_Li_joint = correction[i] @ T_Lk_Li_glim.

## Outputs

- reference_voxels.npz: GT-pose voxel union from **all eligible candidate scans** in the same scene/window/ROI.
- summary.csv: each strategy x oracle/glim/(optional joint), scan counts, voxel statistics, pose error aggregates, same-return displacement.
- per_scan_pose_errors.csv: target-relative rotation error in degrees and translation error in meters for each selected scan; aggregate excludes target anchor.
- manifest.json: exact selected frames, pose-index mapping, CLI arguments and caveats.
- <strategy>_joint_corrections.npy: optional saved Joint correction matrices.

Definitions:
- reference_recall = occupied voxels common to estimate and GT-pose scene reference / reference voxel count.
- reference_precision = common voxels / estimated occupied voxel count.
- paired_rmse_m and paired_p95_m = displacement of **identical original returns** transformed with estimated versus GT-relative poses (GT-ROI mask held fixed); neither is independent surface reconstruction error.
- pose_rot_*_deg and pose_trans_*_m = per-scan target-relative error of the estimated transform against KITTI GT (target anchor excluded from aggregates).

## Interpretation limits

1. Reference includes the evaluated scan points, so voxel metrics are **not** held-out accuracy or independent scene completeness. Oracle recall is a sampling-coverage proxy, not proof of surface reconstruction.
2. KITTI Odometry .bin has no per-return timestamps: no deskew. Dynamic objects and motion distortion can thicken surfaces even using GT poses.
3. Compare matching scan budgets. An 11-vs-20 comparison mixes strategy with number of observations.
4. Misalignment can increase voxel count. Higher occupancy does not prove higher geometric accuracy or calibration observability.
5. An extrinsic error claim requires later evaluation against independent camera-LiDAR extrinsic GT; camera-visible ROI/physical plane association are out of scope here.

## Tests

    python -m pytest tests/test_oracle_accumulation.py -q

Synthetic tests exercise coordinate composition, ROI voxel logic, equal scan budget and the CLI; they do not replace a real KITTI07 experiment.


## Joint vs PGO from exactly the same GLIM scans

To run both refiners independently on each selected scan set, replace --refiner none in the command above with:

    --refiner both --refine-iterations 4 --pgo-iterations 4 --pgo-pair-max-nfev 38

Or use --refiner pgo / --refiner joint for one solver. Default --refiner none keeps the original oracle + GLIM baseline. Both Joint and PGO receive the **same original, unrefined GLIM accumulation**; PGO is not initialized from a Joint result.

PGO uses spade_calib.lidar.pgo_solver.solve_pgo, with a PGOSettings instance whose pairs use the same JointSettings sampling and correspondence thresholds as Joint. Crucially, d1_roi=False (KITTI07, NOT the D1-specific physical ROI). --refine-voxel-m and --refine-max-samples affect pair sampling for both solvers; --refine-iterations controls Joint optimization only; --pgo-iterations controls graph IRLS and --pgo-pair-max-nfev controls pairwise least-squares fitting.

The summary.csv stage column becomes oracle, glim, joint, pgo for --refiner both. per_scan_pose_errors.csv also includes joint and pgo. Each scan strategy emits <strategy>_joint_corrections.npy, <strategy>_pgo_corrections.npy, and matching <strategy>_{joint,pgo}_diagnostics.json files. These correction matrices map points already in the GLIM target LiDAR frame into the refined target frame. For each scan, the evaluated raw LiDAR-to-target transform is T_refined = T_correction @ T_GLIM_relative; do not treat T_correction alone as a raw scan pose.

The PGO implementation is the repository's pairwise point-to-plane + small-angle graph optimizer; it is not GLIM's internal back-end. Pairwise graph diagnostics are training residuals, not GT accuracy. The existing joint-vs-pgo tests are synthetic; perform real KITTI07 verification and check connected matching constraints before drawing conclusions.
