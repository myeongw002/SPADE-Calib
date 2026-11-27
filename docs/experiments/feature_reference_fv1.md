# FV1 — independent reference review pilot
OPT1 did not improve extrinsics (best 54.67 cm vs 54.77 cm); do not tune
optimization until actual physical boundary validity is audited.

    python scripts/experiments/run_feature_reference.py \
      --lidar64-dir /actual/KITTI/data_odometry_velodyne/sequences/07/velodyne \
      --gt-calib /actual/KITTI/data_odometry_calib/sequences/07/calib.txt \
      --out outputs/feature_reference07_FV1

Requires MV8 review/full, EP0, original raw64 (not synthetic16), corresponding
synthetic16 input from MV8 sources. Verify raw scan >1.5x points and >=90%
synthetic16 within 1mm of raw64. Raw64 and synthetic16 are not independent
sensor measurements. No ring classification, accumulation or DA2 used for
reference evidence. GT camera projection evaluation-only. Original64 projected
nearest per pixel, depth up to 80m, sparse jump proposals only stratify tiles.
RGB boundary annotations are manual and not snapped to any detector.
No interpolation; sparse raw64 can miss boundaries; baseline/viewpoint and
motion affect projection. Camera and LiDAR reference tasks independent:
- Camera: random non-overlapping 224x124 tiles, two sampled from upper
  raw64-proposal-count half, two from lower half per anchor. Actual sampling
  fractions and strata private in manifest. RGB and raw64 depth overlay
  toggle, manual positive/uncertain polylines, complete/partial/unknown.
  Complete only if all clear edges reviewed, not exhaustive true depth GT.
  Do not score recall in partial/unknown. Evaluation limited to reviewed ROI.
- LiDAR: independent random draw of 3 IDs per arm per anchor (max120 draws);
  deduplicate IDs within anchor, privately record sampled_by and memberships.
  Eligible population is all source arm IDs whose GT projection lies in camera
  FOV and optical z .5..40m; no GT visibility or edge-proximity gate. Raw64
  local az/el +/-2deg and observed range profiles, no RGB/DA2, no arm names.
  Judge candidate as anchor-view foreground depth discontinuity, not source-view
  validity; positive/negative/unknown. Raw64 local range plots may be ambiguous;
  unknown must stay separate. Candidate magenta marker does not imply truth.
  For per-arm estimates use sampled_by, not every membership in pooled draw.
  Counts and uncertainty report separately, not a full recall reference.
Open review.html locally in browser, no server, no network. Export
fv1_labels.json and upload it with manifest.json and summary.json; preserve
review directory for FV2/FV3. Local storage best effort, explicit JSON export.
Camera coordinates in tile-local original pixels; manifest box converts global.
No automatic physical precision/recall or optimization in FV1. Later FV2/FV3
must evaluate independent labels, unknown masks, boundary tolerances, sampled
denominators and directional diversity. Reviewer not blinded to sensor evidence,
only to detector source. Small pilot, not publication-ready ground truth.
