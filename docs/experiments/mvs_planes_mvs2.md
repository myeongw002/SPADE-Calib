# MVS2 — Ground and normal-aware finite plane extraction

Input: MVS1 cloud_anchor_camera_m.ply, already in metric anchor camera axes.
No SfM/MVS rerun, extrinsic/scale refinement, LiDAR guidance, or GT.
Retain finite nonzero input normals; normalize signs only in the absolute-dot
gate. Common voxel representatives (.10 m), deterministic cap 40000, range
1..65m; this is sampled support, not full-resolution surface segmentation.

G0/G1: distance-only / distance+normal gated triplet RANSAC on common camera
ROI x ±3m, z 3..25m, y .35..3.5m; tilt <=15deg to camera up [0,-1,0],
positive up offset .35..3.5m. Same seed. Camera up is an approximate
gravity prior, not estimated IMU gravity. G1 used for shared ground removal
(distance AND sign-invariant normal compatibility over entire sample, not
semantic ground or connected ground outside ROI). If G1 fails, stop.
P0/P1: identical remaining sample, distance-only / normal-aware scoring,
exclude normals within 25deg of approximate up to focus on nonground
constraints. Both use triple hypotheses, connected finite patch (existing
12-neighbor graph with .65m radius), TLS refit and monotone trimming.
Threshold .12m, 20deg normal gate, min 100 sampled points and .5m robust
extent in both tangent axes, 512 hypotheses per extraction, max 12 patches.
Top distinct hypotheses screened (up to 12). No guarantee of optimal
RANSAC result or physical planar accuracy. Normal alignment is partly
enforced by P1; lower normal dispersion is not independent validation.
P0/P1 have same starting seed but later remaining sets/hypotheses differ.
No explicit directional quota: report diversity before designing selection.

Run in existing SPADE environment with Open3D:
    python scripts/experiments/run_mvs_planes.py \
      --ply outputs/mvs07_pycolmap_seg00/segment_00/pycolmap/cloud_anchor_camera_m.ply \
      --out outputs/mvs07_MVS2_seg00

summary.json reports support counts, RMSE, distance/normal percentiles,
robust tangent extent, centroid, offset, and equal-plane-weight singular
values of N/sqrt(K) with/without shared G1 (translation-only proxy given
fixed rotation/scale and correct correspondences). Weak direction from
right singular vector is not unique if nullspace dimension >1. Rank gate
5% largest singular value. Inspect G0/G1.ply, P0/P1/colored_planes.ply,
individual plane PLY. Source indices in sampled.npz; arm indices refer
to that sample. Gray unassigned, green ground-compatible, other colors
patches. All thresholds are exploratory. Compare physically identical
planes against independent LiDAR or GT later; do not optimize extrinsic yet.
