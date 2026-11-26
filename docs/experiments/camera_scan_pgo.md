# Camera-initialized LiDAR registration PGO pilot

Original segmented HE archive, fixed scale and segment-local anchor camera
pose, original SfM rotations fixed in primary arms. Use selected matched
poses every 5 plus anchor and last; candidate edges at node gaps 1 and 2.
Camera metric relative poses from original SfM on all candidate edges;
LiDAR observations are raw synthetic16 scan-to-scan point-plane ICP
initialized by X^-1 C_i^-1 C_j X. Independent GLIM-seeded repeats are
diagnostic only. Primary camera-seeded acceptance frozen before fitting.
No ground, line, BA or scale free variable. No new visual tracks.
Registration uses voxel/normals from existing joint solver. Rank-deficient
directions retained as zero weights via local normalized SVD. Equal edge
mass, robust point-plane information proxy not covariance. No deskew,
dynamic rejection or guaranteed correspondence. Reject iteration limit,
insufficient matches/overlap, correction limit, rank<3. Failed edges
reported. If any segment has no accepted edges, stop. Camera tether
connects all nodes even if registration graph is disconnected.

Compare initial, registration X only, pose translation only, joint;
GLIM X only/joint on identical accepted edges and factor weights (local
registration chart applied at GLIM measurement as a matched weighting
control, not GLIM covariance). Optional --full-pose adds both 6DoF arms.
Fixed original scale in camera tether; corrected positions can change
local lengths. Robust residuals are componentwise soft-L1 before
normalizing each family by edge count. Default weight1 and existing
camera sigmas, heuristic not likelihood. Bounds ±5deg/±1m extrinsic,
±2m translation/±5deg node correction. Same-state raw marginalized
information, fixed graph. Not proof of physical observability.

```bash
KITTI=/media/myungw00/256GB_SSD/kitti
python scripts/experiments/run_camera_scan_pgo.py \
 --run outputs/segment07_trial1 \
 --gt-calib "$KITTI/data_odometry_calib/sequences/07/calib.txt" \
 --gt-poses "$KITTI/data_odometry_poses/poses/07.txt" \
 --out outputs/pgo07_camera_scan
zip -r pgo07_camera_scan.zip outputs/pgo07_camera_scan
```

Pilot 07 first. Source original GLIM/COLMAP and LiDAR paths from config;
override --sequence/--glim/--sfm/--lidar-dir if moved. Use same 16ch data.
GT loaded only after fits. summary.json, registration_edges.json,
graph.npz. Check registration GT relative error and seed disagreement
before judging graph; if registration worse than GLIM or joint does not
improve X, don't proceed to larger graph or new pose freedom. No holdout,
single sequence pilot. Do not claim novelty.
