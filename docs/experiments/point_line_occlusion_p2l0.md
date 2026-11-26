# P2L-0: Camera-view occlusion-point matching pilot

Galibr-inspired, not its exact implementation. No 3D line extraction or fit.
Frozen E1 jump-dominant lines (jump >= .4, jump >= crease, valid fraction >= .5),
no per-frame top-k. Use only segment anchors with accepted B4-c ground planes.
Same LiDAR input and line pool in all arms. Ground plane masks near/below
ground points as foreground candidates; **all** points participate in z-buffer
and ground can be the observed far side. No object segmentation or exact
camera visibility. Use original voxel returns, not interpolated edge points.
Grid cells 4px; keep nearest optical Z; search up to 2 cells cardinally for
far-side and opposite foreground support. Unknown cells alone cannot create
edges. Direction compatibility is a coarse axis proxy (45 deg). Unique
nearest finite image segment within gate, second-best distance margin 4px;
one point per 8px along-line bin, >=3 bins and >=25% coverage. All thresholds
are provisional. No optimizer. No automatic physical identity validation.

## Run

From repo root, existing research venv. The runner uses threadpoolctl to limit
BLAS threads (install it if absent: python -m pip install threadpoolctl).

    KITTI=/media/myungw00/256GB_SSD/kitti
    python scripts/experiments/run_point_line_occlusion.py \
        --gt-calib "$KITTI/data_odometry_calib/sequences/07/calib.txt" \
        --out outputs/p2l07_A0

This reads outputs/segment07_trial1, outputs/ground07_B4c and
outputs/edge07_E1_depth directly; no ZIP needed. The default A0 (or ground
summary arm) is the long segment 18cm voxel cloud, not the short local W21
cloud. To test existing ghost-filtered W21 5cm data instead:

    python scripts/experiments/run_point_line_occlusion.py \
        --cloud-data outputs/lines07_M0_ghost \
        --gt-calib "$KITTI/data_odometry_calib/sequences/07/calib.txt" \
        --out outputs/p2l07_W21

M0 cloud preparation only (if missing), no annotation or line extraction:

    python scripts/experiments/run_crossmodal_lines.py prepare \
        --run outputs/segment07_trial1 \
        --out outputs/p2l07_cloud_W21 \
        --image-dir "$KITTI/data_odometry_color/sequences/07/image_2" \
        --intrinsics-calib "$KITTI/data_odometry_calib/sequences/07/calib.txt" \
        --initial-summary outputs/ground07_B4c/summary.json \
        --initial-arm B4c_w0.1 --window 21 --voxel .05 --ghost-filter

Then use --cloud-data outputs/p2l07_cloud_W21. No change to existing M0
archives. --gt-calib is optional; B4-c proposals do not use GT. GT_oracle
re-extracts candidates and rematches at GT as a diagnostic ONLY. Fixed-pair GT
evaluation projects the exact retained point into the SAME line, checks
finite segment and image bounds, and counts <=10px perpendicular residual.
Not a true correspondence precision; random planar points can satisfy it.
For comparisons use all retained points in denominator, including those
outside finite line/FOV at GT. Edge and all use same z-buffer. Counts are
not equal budgets and point-to-line groups are correlated.

## Preliminary A0 archive experiment (same core algorithm)

8 KITTI07 anchors, 145 E1 jump-dominant lines. Ground mask from archived
B4-c segment LiDAR plane. B4-c translation error 54.77cm. For each row:
matched line instances / retained points / fixed GT within10 fraction.

| initializer | arm | 20px | 40px | 70px |
|---|---|---|---|---|
| B4c | all | 113/1033/19.7% | 119/1085/19.5% | 120/1130/18.8% |
| B4c | edge | 69/451/23.9% | 81/578/21.6% | 92/691/19.4% |
| GT | all | 116/1090/91.1% | 118/1117/89.0% | 118/1147/86.7% |
| GT | edge | 69/485/81.2% | 85/612/67.2% | 93/690/59.9% |

These counts were obtained by running the core on uploaded A0 PLY and E1
CSV; the new repo CLI and W21 have not been run locally. A0 contains long
accumulation and possible ghosts; results are not evidence of calibration
performance. Widening gate does not resolve capture range. GT all baseline
has high geometric consistency by construction: closest arbitrary points
at GT may lie near line without being physical edges. Compare visual
physical identity, not just distance, before optimization. The depth-only
camera cues do not guarantee static identity.

Outputs: summary.json, metrics.csv, review.csv, frame overlays and compressed
pair arrays with original 3D coordinates, IDs, 2D lines and GT residuals.
Green image lines, orange candidate occlusion points, red retained points.
No review labels filled automatically.
