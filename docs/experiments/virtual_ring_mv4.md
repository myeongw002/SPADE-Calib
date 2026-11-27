# MV4 — virtual-ring reconstruction

Use same raw synthetic16 scans, GLIM, W21 M0 source-frame list and EP0 outputs
as MV1. Do not consume exported edge-only archives as input.

    python scripts/experiments/run_virtual_ring_comparison.py \
      --ring-calib /actual/path/to/velodynecalib_S2_factory_flatness_intensity.xml \
      --out outputs/edgepoints07_MV4

Optional --frames 389 for a pilot; use fresh output dir. --virtual-ranks
defaults to 3,7,...,63 ascending elevation (observed in earlier MV run).
Validate target coverage >=95% of assigned target returns; override if
different input. Source recovery remains the same full-64 assignment.
No camera or extrinsic in feature extraction. No interpolation, voxel
sampling or invented returns. Same target free-space mask as MV1.
Actual target points keep accepted source ring. Historical points classified
against nearest eligible virtual beam using same approximate offset model.
Virtual angular mismatch <=.20deg, best/second margin >=.04deg; adjustable.
Fixed .2deg azimuth bin (effective 360/ceil(360/.2)); nearest radial range
per cell. Target priority in Fill16 prevents replacing occupied target cells,
but within each target cell only the nearest accepted target return is used.
Historical candidates outside target-frame .5..M0 max_range removed.

Arms:
- MV1_target/MV1_all: source-view ring edges before accumulation, existing MV1.
- Target_grid16: target-only binning control.
- Nearest16: accumulated 16 virtual beam nearest-return grid.
- Fill16: preserve Target_grid16 bins, fill empty bins from accumulated data.
- Target_grid64: target-only 64-row control (same actual sensor).
- Nearest64: accumulated virtual 64 beam nearest-return grid.
- Fill16_new_vs_target / _MV1: near source IDs absent from respective
  source-view baselines. These are ID novelty, not new physical boundaries.

Same original ring_edges thresholds; actual azimuth separation <=.8deg, not
grid-center separation. Range jump and opposite surface support use actual
target-view XYZ. No vertical edge extraction or smoothness in this pilot.
DA2 jump proximity in B4c/GT, common EP0 same-pixel visibility filter, counts
and pooled fractions. Pixel-level proximity is not true correspondence
precision. Compare coverage vs precision and absolute inlier proxy count.
Grid stats include cyclic gaps, added/replaced bins, edge hypotheses using
added/replaced bins, mixed source scans. Nearest64 may trade sparsity for
vertical coverage; compare per-arm denominators. Empty angular bins can be
due to occlusion, range limit, sensor sampling or ring rejection; not all
are physically missing returns. No full surface occlusion or deskew.
All selected points retain raw scan/return identities. Grid NPZ index_map
refers to original accumulated array; scan_ids and source_return_index are
also stored. source_features.npz contains the subset used by any near/far/back
triple; per-arm edge_triples feature_indices and projected candidate_indices
index this subset. Occupancy PNG gray target, orange historical, black unknown.
Virtual_error is nearest-virtual residual for eligible points; actual target
ring is inherited from source classifier and may differ. No optimization.
