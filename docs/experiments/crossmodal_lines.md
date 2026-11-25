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
