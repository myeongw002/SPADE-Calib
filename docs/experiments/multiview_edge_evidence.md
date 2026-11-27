# MV1/MV2 — source-ring multi-view edge evidence

MV0 = existing EP0 accumulated then projected sparse depth jumps.
MV1 = range jumps inside each inferred source ring, accumulate near returns.
MV2 = MV1 with distinct source-scan recurrence thresholds 2,3,5.
No calibration optimization, no cross-modal matches. DA2 only for reporting.

Run from repository root, using the same raw synthetic16 .bin files and GLIM
pose file recorded in segment07_trial1/config.json, existing M0 and EP0.
Supply the actual factory XML used by the original downsampling, not KITTI
camera calib.txt, not nominal VLP16 angles:

    python scripts/experiments/run_multiview_edge_evidence.py \
        --ring-calib /absolute/path/to/velodynecalib_S2_factory_flatness_intensity.xml \
        --out outputs/edgepoints07_MV

First use --frames 389 --out outputs/edgepoints07_MV_smoke to inspect
ring diagnostics. No forced 16-way classification: all 64 beams tested;
optional --ring-ranks takes 16 increasing-elevation rank IDs (0 lowest),
rejects all other best assignments. Do not assume original rank ordering or
selected channels. Check actual histograms before providing this option.
--ring-model angle compares elevation only, default offset uses approximate
single-distance beam geometry with XML cm->m. KITTI decoder two-point
correction/deskew unknown, so this is not exact inversion. Coordinates never
modified. Reject residual > .20deg or best/second margin < .04deg.
No original ring recovery code available at implementation. Ring validity
must be checked using per-scan rank histograms and residual quantiles.
Raw .bin lacks firing time and ring metadata.

Same-ring cyclic azimuth neighbors with gap <= .8deg each side.
Near/far observed range jump and opposite similar range back support use
same numeric thresholds as EP0, but radial range vs optical Z differ.
Rigid GLIM transforms; target free-space filter same as M0 (near candidate
only). Far/back retained as source-view evidence. No voxel sampling before
recurrence, no new inferred points. For each near candidate query nearest
edge per distinct source scan in .25m (including self). Record supporting
indices, distances, RMS, max deviation from own viewing ray. Spatial
recurrence favors stable geometric boundaries but can lose view-dependent
silhouettes and retain repeated clutter. No guarantee of identity.
MV1_target single scan, MV1_all all candidates, MV2_n2/n3/n5 thresholds.
Each arm independently projects with EP0 B4c and optional archived GT;
one candidate per pixel, reject >.5m behind known EP0 full-cloud depth.
Unknown pixels retained and reported. EP0 reference only same-pixel check.
Compare unique projected pixels, absolute count within 3/5/10px of DA2
jump pixels and 5px fraction, not physical precision. Different candidate
count and raw versus voxel sampling are confounders; not a controlled
equal-density ablation. No direction comparison yet (source-view
foreground/background sign need not be valid in target camera view).
Outputs source_edges.npz retains source scan, original .bin row IDs, near/
far/back XYZ, sensor origins and recurrence support table. Per-arm NPZ
candidate_indices refer to source_edges array and visible mask. Summary
contains per-scan 64-rank histograms, residuals, input hashes, count and
proximity results. Overlays orange LiDAR, green DA2. Test file synthetic;
real KITTI execution not yet verified.
