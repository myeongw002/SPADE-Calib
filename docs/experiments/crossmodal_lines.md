# M0 — manual finite LiDAR wall intersections and actual image lines

Three steps: prepare / annotate / fit. Use a graphical desktop for
annotation (Matplotlib interactive backend, e.g. TkAgg or QtAgg; no
OpenCV GUI required). Default prepare reads archived GLIM and LiDAR
paths, local centered 21-scan windows, 0.05m voxel and 40m range.
These are exploratory defaults, not selected optima. No deskew.
All prepared clouds are in the synchronous target LiDAR coordinates.
GT is never loaded by prepare/annotate, only camera intrinsics K.
Initial X defaults to archived HE, or a successful H summary arm.

Inspect prepared frame image and initial overlay. Select frames with
static building corners visible, not GT errors. For each physical
corner run annotate: select polygon around each adjacent wall in
LiDAR XY top view (left click vertices, middle click/Enter finish,
right click removes last). Choose thin separate wall strips with
vertical extent and points near the corner; z filter default -1 to
4.5m excludes much of road. The script fits independent local walls
and checks near-vertical intersection, supported overlap, minimum
length. No forced nearest edge. Click two endpoints on the RAW upper
image, not the initial projection below. Confirm same physical
static edge manually after review. If a corner has only one visible
wall or no supported intersection, skip it. No fallback invented line.
Save actual selected inliers and annotations. Use unique feature IDs
for different physical corners; reuse ID for same corner across frames.
Train/test frame and feature disjoint; not independent if scene
overlaps. Aim for 10–20 diverse lines, >=2 frames, different distances,
both sides of image; keep several separate test frames/features.
Only 3 lines is minimum execution gate, not sufficiency. 
Fit x-only, translation-only and full 6DoF separately; all start same
X with deterministic bounded restarts. Compare GT errors, train/test
line residual, positive depth, full rank and conditional X information.
Best restart selected only by train cost. GT-centered X sweep is an
oracle diagnostic, not initialization. Infinite line residual can
slide along edge; overlap is reported only. Correlated samples/lines
do not yield calibrated covariance. Vertical lines alone may not
constrain full 6DoF; do not equate solver success with identification.
No HE/Ground fusion, no automatic matching, no W1/W21 causal ablation yet.
Need physical paired supports and control experiments before M1/M2.


## Linked ROI editor (updated)

BEV is vehicle-oriented: display horizontal = -LiDAR Y (right),
vertical = LiDAR X (forward). The original XYZ, calibration and plane
fitting are unchanged. The editor displays BEV, initial projected
LiDAR on image, and raw image simultaneously. Click polygon vertices
in BEV or projected image; green highlights show the same original
points in all views. If both polygons are drawn the selection is their
intersection. Right click undoes last vertex in its panel, R clears
both, Enter accepts, Esc cancels. Turn off toolbar pan/zoom before
drawing. First wall inliers appear magenta for second wall. Image ROI
uses the initial X only for selection; it may include occluded points.
Use a BEV ROI to constrain depth or --depth-min/--depth-max (optical
depth, image selection only). The raw image endpoints are annotated
separately, never snapped to projected points. Prepared data and old
annotations remain compatible; new annotations record both ROI polygons.
No need to run prepare again.


## Optional existing target free-space filter

Add `--ghost-filter` to prepare and use a NEW output directory. It
reuses lidar/ghost_filter.py unchanged, after rigid accumulation in
the target LiDAR frame and BEFORE voxel sampling. Defaults: azimuth
0.2 deg, elevation 2 deg, range margin 0.5 m. The local target scan
ID is found from the actual frame list (not the KITTI frame number).
Historical and future non-target points in the centered window
are tested against target range bins. Target returns, unobserved
bins and points behind target returns are kept by this filter;
voxel sampling may subsequently reduce them. This is not full
dynamic-object removal or temporal consistency. Angular binning
and GLIM pose error can remove valid static points. No change to
annotation/fit. The manifest records filter configuration, pre/post
filter counts, removal fraction, target counts, per-scan counts
and post-voxel count. Compare to an unfiltered run with identical
frames/window/range/voxel/initialization; don't reuse annotations
across changed cloud manifests.
