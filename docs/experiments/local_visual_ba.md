# H: local visual BA + Hand-Eye / Scene Consistency

Diagnostic, not a production calibration pipeline. GLIM and camera intrinsics
are fixed. In each archived scale segment, selected registered frames and
actual COLMAP image feature observations define a local BA. Original landmarks
are copied into each anchor frame, scaled by archived metric scale s0.
Landmarks that cross segments are independent local copies; no global BA or
boundary continuity is claimed. Gauge: anchor pose fixed and largest absolute
nonzero initial baseline translation component fixed; true metric scale is
an independent ratio alpha=s/s0. Pose increments are bounded, landmarks
free, depth barrier and robust reprojection. Uses PINHOLE, SIMPLE_PINHOLE,
SIMPLE_RADIAL, RADIAL, OPENCV intrinsics as stored by COLMAP; fail on others.

Scene uses exactly the frozen G voxel-support point/plane moment factors.
H1 and H2 share identical selected cameras, observations, ground tracks,
initial state, bounds and block normalization; HE residual uses existing HE
sigmas, scene uses sigma_scene. These weights are exploratory, not calibrated
covariances. Ground uses B4 fixed correspondence and frozen camera normal;
its offset changes with mean displacement of selected observed ground
landmarks along that normal, then scales by alpha. This is a local
fixed-normal approximation, NOT a plane refit. Original B4 support is
identified by XYZ with only float32 tolerance, then associated to actual
landmark IDs. At least 6 selected ground tracks are required per segment.
Other tracks are proposed per view, unioned, and all their eligible
observations retained. A 3px initial reprojection gate, 2-view and selected
parallax gate, connected camera graph and >=12 obs/view are required.
These are necessary but not sufficient for local identifiability.

Arms:
- Hfixed_HE / Hfixed_scene: original poses/landmarks, X and scales only.
- H0_visual: local visual-only BA, no extrinsic/scale update.
- H0_then_HE / H0_then_scene: H0 poses/landmarks frozen, X and scales.
- H1_joint_HE / H2_joint_scene: joint BA, shared X, per-segment scales,
  start from the SAME original state (not H0).

Sparse finite-difference Jacobian (structural sparsity) and LSMR/TRF;
not explicitly parallel, no GPU, no dense global Jacobian. Ground centroid
introduces extra FD colors. No claims of full rank or global convergence.
Check solver status, bounds, reprojection/cheirality, pose shifts, scales,
GT extrinsic and projections. H0 visual observations are training data,
not a held-out test. H0_then results depend on H0 status even if converged
themselves. Joint objective reductions do not prove calibration improvement.
No GT in optimization or selection.

```bash
KITTI=/media/myungw00/256GB_SSD/kitti
SEQ=07
python scripts/experiments/run_scene_consistency.py \
  --run "outputs/segment${SEQ}_trial1" \
  --out "outputs/scene${SEQ}_H_factors" \
  --windows 5 --center-stride 5 --support-mode voxel --local-voxel .18 \
  --prepare-only
python scripts/experiments/run_local_visual_ba.py \
  --run "outputs/segment${SEQ}_trial1" \
  --ground-run "outputs/ground${SEQ}_B4c" \
  --scene-factors "outputs/scene${SEQ}_H_factors" \
  --out "outputs/local_ba${SEQ}_H" \
  --window 5 --scene-kind point \
  --gt-calib "$KITTI/data_odometry_calib/sequences/${SEQ}/calib.txt" \
  --image-dir "$KITTI/data_odometry_color/sequences/${SEQ}/image_2"
```

First run uses a reduced visual graph (stride 5, 40 track proposals/view,
12 ground track cap). If support checks fail, prepare factors with stride 3
or 1 in a fresh folder, not a GT-selected frame subset. If max_nfev is reached,
report non-convergence; do not silently count it as success. Save summary.json
and per-arm segment NPZ (refined local pose/landmark arrays, observation IDs
and pixels, reference and optimized scales). For actual metric anchor-local
positions multiply stored ref_metric positions by scale_est/scale_ref.
Projection is evaluation-only on original A0 map/single scan.


## Retry after duplicate COLMAP observation error

Multiple measured features in one image may reference the same landmark.
After the original depth and 3 px gates, keep exactly one measured
observation per (image, landmark), using minimum original reprojection
error and earliest row on ties. No pixels are synthesized or averaged.
The cleaned observation graph is fixed across all H arms and the removal
counts are written per segment and frame. This does not prove which
feature is physically correct and may favor the original SfM solution.

Run `bash scripts/experiments/run_h_all.sh` from the project root in the
active environment. It checks existing factor archives (summary, settings,
NPZ contents and shapes); reuses valid ones, creates a new unique directory
instead of overwriting incomplete factors or BA outputs, and runs 07, 09,
10 sequentially even if one fails. Nonzero final exit status indicates
at least one failed sequence. Override KITTI, CENTER_STRIDE or VOXEL with
environment variables. Output locations and failures are printed in logs.
