# MV3 — source-ring jump vs LOAM-inspired smoothness

Preserve existing MV1/MV2 scripts; reuse their raw scan input, calibration XML,
same GLIM rigid accumulation, target free-space filter and EP0 camera data /
reference z-buffer. Use same input paths and ring parameters as MV1.

    python scripts/experiments/run_ring_smoothness_comparison.py \
        --ring-calib ~/PROJECT/hand-eye/configs/velodynecalib_S2_factory_flatness_intensity.xml \
        --out outputs/edgepoints07_MV3

Optional --frames 389 (fresh out directory). Defaults use all eight EP0 anchors.
Existing EP0 and M0 outputs required; no ZIP input, no new depth inference.
Use original synthetic16 scans, not voxel clouds or MV1 source_edges.npz.

Feature score is norm of sum of same-ring neighboring point differences divided
by 2*m times center range, following LOAM paper's normalized smoothness
idea (not full LOAM). Five returns each side, cyclic azimuth order. Every
consecutive angular gap <= .8deg and >0. Invalid/ambiguous rings excluded.
Smooth_raw is top-K scores after ghost filter. Smooth_clean additionally
rejects every window containing an adjacent radial jump >= max(.3m, .03*min
range). This gate is deliberately distinct from the existing MV1 threshold;
it is a heuristic, not proof of a crease. Score also responds to sampling
asymmetry, smooth curvature and grazing surfaces. No sector quota, NMS,
LOAM parallel-beam mask, or nonuniform-spacing correction.

K is the number of unique post-ghost Jump returns in that source scan
(raw/clean independently selected up to K; shortages reported). Comparison
is at equal preprojection budget where enough eligible returns exist, not
necessarily equal projected budget. Jump_plus_Smooth_clean uses union of IDs
and may have more candidates; count and overlap reported. No recurrence gate,
no 3D line, no matching or extrinsic fitting. All arms use same optical depth
projection and reference visibility rule. DA2 jump, crease and union evaluated
separately via 3/5/10px proximity, absolute counts and pooled fractions.
DA2 crease overlaps jump and is not a physical label. No GT used for selection.
Outputs source_features.npz with original scan/return IDs, feature masks,
score (NaN if not selected by either smoothness arm), selected-window gate,
ring rank, original target-frame XYZ; source_jump_edges.npz preserves far/back
source evidence. Per-arm candidate_indices index source_features.npz.
summary.json contains frame-level and aggregate metrics. Compare Jump_all
against earlier MV1_all under identical inputs. Unit tests cover seam,
missing ring, known range jump and deterministic budget. Real-data run
requires local raw scans; no claim of improved performance before execution.
