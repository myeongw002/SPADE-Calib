# B4: multi-scale Hand-Eye + ground only

Use the previously saved segment reconstruction output. No new accumulation,
LiDAR refinement, local BA or all-plane point-to-plane calibration is needed.
The old optional --refine-extrinsic experiment remains a legacy baseline;
do not use it for B4. Run the new script instead.

```bash
python -m pip install -e '.[research,dev]'
pytest -q tests/test_segmented_ground.py
python scripts/experiments/run_segment_ground.py \
  --run outputs/segment07_trial1 \
  --out outputs/ground07_B4 \
  --gt-calib /media/myungw00/256GB_SSD/kitti/data_odometry_calib/sequences/07/calib.txt \
  --oracle
```

Repeat for 09/10. The original run's config.json supplies GLIM, SfM and KITTI
times.txt paths. Override with --glim, --sfm or --sequence if files moved.
Original full pose inputs are necessary; the reconstruction ZIP alone does
not contain the motion matrices. Reconstructed image names, timestamps,
partition, selected pair rows/counts, anchor and archived HE residual cost
are checked. No pair re-gating or segmentation rerun.

A0 (GLIM rigid) is default. A1 can be tested separately using --arm A1;
it intentionally keeps the original GLIM motion pairs for comparison
and is not a pose-coupled joint fit. Original run directories are read-only.

## Ground selection

Use the saved original metric SfM points.npz and LiDAR cloud_voxel.ply, not
the previous all-plane matches. Restrict to an anchor-camera forward corridor
(default depth 3–25m, lateral +/-3m) and below each sensor (0.35–3.5m).
Reuse the existing height/normal constrained RANSAC independently with
camera up -Y and LiDAR up +Z. Match horizontal 1.5m grid cells via HE and
refit each sensor's own points in overlapping cells. Reject insufficient
overlap, thin spatial extent, high plane thickness and large initial
normal/offset mismatch. These are heuristic geometry-only gates, NOT
verified road labels: sidewalk/road confusion, curved/sloped roads,
accumulation drift and SfM depth bias remain possible. Never use GT to
extract or select ground. Review exported per-segment ground PLY.

## B4 objective

B(s)X=XA, shared 6DoF X across every segment, one scale per segment.
Reuse the exact saved HE pair mask and existing motion residual.
E = mean over motion pairs of sum of componentwise robust whitened HE
residuals + lambda_G * mean over accepted segments of sum of robust
whitened ground residuals. Robustify before averaging/weighting. Normal
difference n_C - R n_L, offset residual d_C * (s/s_ref) - d_L + (R n_L)^T t.
Both planes are positive-up and camera d_C is already metric at s_ref.
Default HE scales come from previous fit; ground assumed noise scales are
3deg / 0.15m (not calibrated covariances). Default weights 0.1, 1, 10 are
a sensitivity sweep, not tuned on GT or statistically optimal.

B4-a fixes scales by default. Add --refit-scales to additionally run B4-b
with log-scale shift bounded to +/-0.15. Extrinsic bounds are componentwise
+/-5deg and +/-1m around the saved HE. All runs use the same accepted ground
supports and motion pairs. Rank uses normalized unrobustified local
Jacobian, including marginalized translation singular values. Do not
interpret rank as physical accuracy. Missing ground or rank deficiency
is reported as failure, not hidden by regularization.

Explicit --gt-calib adds evaluation of the frozen ground supports at GT
and calibration error reporting only. --oracle additionally uses GT as
initial X, with identical HE-centered bounds, original scales and fixed
correspondences. This is a separate diagnostic, not a deployable result;
GT-start drift includes HE bias as well as ground bias. The optimizer
does not use GT for the normal B4 runs. Never choose weights on test GT.

## Outputs

summary.json: archived/recomputed HE cost, baseline, rejected and accepted
ground candidates, all B4 results, optional GT evaluation and oracle.
Per accepted segment: ground_camera.ply, ground_lidar.ply and
ground_lidar_in_camera_HE.ply. Compare axis-wise errors, ground residuals
at GT, HE-only versus joint sensitivity, scale changes, bound hits and
HE residual changes. No claim of improvement before real-data evaluation.
