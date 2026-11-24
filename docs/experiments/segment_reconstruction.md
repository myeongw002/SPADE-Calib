# Segment reconstruction: initial A0/A1 experiment

Uses the merged scale segmentation API and existing KITTI rigid accumulation,
Joint/PGO scan pose refinement and finite plane extractor. No new LiDAR pose
solver, no GT-dependent fitting, no camera BA. The SfM cloud is selected from
the original global reconstruction using within-segment track count, global
reprojection error and a conservative parallax filter; it is not independent
local triangulation. The finite plane extractor is reused for both clouds.

## Inputs

- KITTI sequence directory containing times.txt (calib.txt optional evaluation).
- Explicit LiDAR .bin directory, e.g. your existing synthetic 16-channel data.
  This script does not simulate 16 channels, recover ring IDs or deskew.
- Same-sequence GLIM LiDAR TUM poses, one contiguous row per frame. For a
  partial export specify --glim-first-frame. Verify frame/time identity before
  running; the KITTI adapter uses row/frame association, not GLIM clocks.
- Full COLMAP sparse model (images.bin + points3D.bin, or corresponding txt).
  Use the same camera and reconstruction as the initial Hand-Eye.
- A handeye.json from the existing Hand-Eye command; used as initialization
  (no GT transform). Pair gating depends on this initialization.

## Run

```bash
python -m pip install -e '.[research,dev]'
python scripts/experiments/run_segment_reconstruction.py \
  --sequence /data/kitti/sequences/07 \
  --lidar-dir /data/kitti16/sequences/07/velodyne \
  --glim /data/glim07/odom_lidar.txt \
  --sfm /data/colmap07/sparse/0 \
  --handeye /data/handeye07.json \
  --out outputs/segment07_trial1 \
  --contains-frame 105 546 \
  --method joint \
  --gt-calib /data/kitti/sequences/07/calib.txt --camera-index 2
```

All selected scans are used by default, potentially expensive. For an initial
smoke test add --max-scans 40; it samples across the segment and retains the
anchor, but sparse spacing can reduce refinement overlap. Use --scan-stride 1
and no cap for the actual full-segment experiment if memory permits. Scan lists
are saved. Use --segment-id to select specific segments, or omit both selectors
for the full sequence. --partition fixed --segments 8 compares fixed
segmentation (same segment count need not imply identical fitting pairs);
--partition global tests a single segment. Change --min-scale/--max-scale and
--min-sfm-baseline if the SfM gauge differs.

A0 = GLIM rigid accumulation. A1 = same scan IDs, reference frame and input
cloud with existing joint solver (or PGO via --method pgo); D1 ROI disabled
and anchor fixed. Both use the same original multi-scale Hand-Eye solution.
Scales remain fixed after LiDAR correction to isolate the geometric effect
in this ablation; a fully coupled scale/pose refinement is not implemented.
A1 failures are recorded, not silently replaced by A0. Joint/PGO settings
were validated historically on short D1 windows, not on full KITTI segments.

## Outputs

segmented_handeye.json: all segments, scales, X, frame/image metadata.
config.json: run options. summary.json: per-segment report and any GT
evaluation; incremental reports survive a later failure.

segment_XX/sfm_camera: metric voxel PLY, original point IDs and XYZ NPZ,
plane PLY files and normals/offsets JSON. segment_XX/A0 and A1: LiDAR-frame
voxel PLY, individual finite plane PLY, HE-aligned camera-frame PLY and plane
JSON. A1 additionally saves per-scan corrections and correction matrices
in anchor-LiDAR coordinates. Each segment uses a different anchor; don't
combine these PLY files as if they were in one world frame. The report lists
provisional associations and bidirectional median point-plane distances.

Optionally add --refine-extrinsic to run fixed provisional plane-pair 6DoF
refinement per arm across common successfully reconstructed segments. It
uses mutual best angle/offset/finite support gating under initial X,
bidirectional point-to-plane residuals, equal segment/plane/direction
weights, soft-L1 and fixed scale. No motion prior. Rank-deficient cases
refuse fitting. Inspect rank, bound hits, per-plane associations and GT
errors; a low residual/full rank does not prove correctness. Pair sets may
differ across arms, so comparison is not a strict same-correspondence
ablation. Review the plane PLY and correspondences before interpreting
calibration. GT is optional and evaluation only; with it the report also
evaluates the *same* provisional matches at GT. No GT selection.

The A0/A1 comparison measures feasibility/geometry quality first, not a
proven calibration algorithm. Plane thickness alone can improve despite
biased position. Local BA, local re-triangulation, adaptive accumulation,
scale refit after scan corrections, and independent validation are future
experiments.
