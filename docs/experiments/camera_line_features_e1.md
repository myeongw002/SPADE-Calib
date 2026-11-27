# E1 — Camera-only geometric line filtering

Purpose: isolate camera candidate quality before investigating unstable LiDAR line extraction. E0 remains unchanged. E1 reads original rectified images and only the left 3x3 of the requested KITTI projection (P2 by default). No LiDAR point cloud, line, pose, ground estimate, Extrinsic, GT transform, or earlier output ZIP is required. The default eight frames are historical E0 anchor IDs for visual comparability; override with --frames for a different independently chosen set. They are not an independent held-out set.

## Run

From repository root in the existing research virtual environment. Official ELSED (pyelsed) is required unless using a camera-only detector cache. See docs/experiments/edge_detector_pilot.md for ELSED and docs/da2_metric_depth.md for DA2 installation.

    KITTI=/media/myungw00/256GB_SSD/kitti
    python scripts/experiments/run_camera_line_features.py \
        --image-dir "$KITTI/data_odometry_color/sequences/07/image_2" \
        --calib "$KITTI/data_odometry_calib/sequences/07/calib.txt" \
        --out outputs/edge07_E1_rgb

With raw relative DA2 inference (adjust checkpoint path):

    KITTI=/media/myungw00/256GB_SSD/kitti
    python scripts/experiments/run_camera_line_features.py \
        --image-dir "$KITTI/data_odometry_color/sequences/07/image_2" \
        --calib "$KITTI/data_odometry_calib/sequences/07/calib.txt" \
        --da2-repo ~/PROJECT/Depth-Anything-V2 \
        --checkpoint /path/to/depth_anything_v2_vitl.pth \
        --encoder vitl --device cuda \
        --out outputs/edge07_E1_depth

Optionally add --elsed-dir outputs/edge07_E1_rgb to reuse its detector arrays. Cache must include frame_XXXXXX/original.png and elsed_segments.npz; image pixels are checked. Use --depth-dir /path/to/raw_predictions instead of --checkpoint to read raw float HxW .npy named 000069.npy etc.; external provenance is not independently verified. Do not use colorized PNG or metric-aligned depth as raw relative predictions. DA2 inference reloads the model per frame using the existing infer_da2 API. No depth model download occurs. Use --frames 69 189 ... and --budget 10 to control evaluation.

## Comparison

All in-bounds nondegenerate ELSED lines >=40 px are the descriptive all arm. Ranked arms use the same pool and same per-frame budget (up to 10); when depth is enabled, exclude lines with fewer than five valid bilateral sample stations from **all ranked arms**. Report full eligible and common pool counts separately. No hidden matching or ground filtering.

- length, random (fixed seed), normal (greedy E-optimal normal diversity).
- jump, crease (rank bilateral relative-depth profile discontinuity scores).
- depth (max of clipped jump/crease times valid fraction).
- fusion (normal diversity with provisional .25 + .75*depth support weight).

With normalized image line l and K, n = normalize(K^T l), use outer products of (n_x,n_z) and mean minimum eigenvalue. This is an image proxy, not the actual Extrinsic Jacobian or covariance. Profiles fit both sides over 3–9 px, 17 stations along middle 80% of line, sign-coherent median normalized by local profile range with 3% global percentile-range floor. Crease is a relative-depth slope-change cue, not a metric normal. ELSED position is unchanged. No depth-only line detector in E1. No hard geometry-confidence threshold yet; fixed-budget ranking allows comparison before tuning thresholds.

## Review

Each frame has original.png, raw ELSED arrays (reusable), optional relative_depth.npy/JSON and depth visualization, individual selection overlays (original detector IDs printed), comparison.png, summary.json. Root features.csv contains all eligible lines, review.csv the union of ranked selections, with no duplicate line per frame. geometry_label, localization_label and notes are blank. Suggested geometry labels: jump, crease, other_static_geometry, appearance_only, dynamic, uncertain. Suggested localization labels: aligned, misaligned, uncertain. Judge against original RGB and where necessary other views, not against the model depth score; hide arm membership during review to reduce bias. A high score is not a ground-truth label. For each arm compare manually confirmed static physical geometry fraction at equal budget, localization, coverage and normal diversity; report uncertain/unreviewed separately. Full recall requires exhaustive independent annotations and multi-view repeatability requires physical tracks; neither is implemented. Do not infer LiDAR matchability or calibration improvement from E1.

Synthetic unit/integration tests cover projection normals, depth affine invariance, jump/crease vs smooth profiles, empty candidates, equal-budget selection, and cached-camera CLI. Actual DA2 and real KITTI E1 results require running locally.
