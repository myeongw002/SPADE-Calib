# MV5 — multi-view virtual 64-ring evidence

Use existing raw synthetic16 scans, GLIM, W21 M0 list and EP0. No ZIP input.
Same ring thresholds and angular nearest rule as MV4 Nearest64. No change to
existing MV1/MV4 code. Run:

    python scripts/experiments/run_multiview_virtual_ring.py \
      --ring-calib /actual/path/to/velodynecalib_S2_factory_flatness_intensity.xml \
      --out outputs/edgepoints07_MV5

Default --view-stride 5 gives center and offsets -10,-5,0,+5,+10 for W21.
--view-stride 1 evaluates all 21 poses; use fresh output and expect more work.
--frames 389 for smoke. No additional scans outside the existing W21 list.
All methods extract horizontal range jumps only. No LOAM, PCA or new depth.

Build raw unfiltered anchor-coordinate cloud once. For each selected real
GLIM pose, transform same cloud to that coordinate frame. Restore exact
raw own-scan points in non-anchor view, use float32 coordinates as in MV4.
Per-view target free-space filter against own scan, then 64 virtual ranks
and .2deg angular nearest-return grid. Same source-ring labels for own
scan, same virtual gates for other returns, same .5..max_range cutoff.
Near/far/back extracted in view coordinates. Do not discard other-view
points from the input by anchor free-space first. Final near returns must
pass anchor free-space for fair comparison to MV1. Camera visibility
uses the same EP0 reference depth at anchor only. All original IDs retained;
return to anchor by index lookup, not repeated transforms or averaging.

Controls: MV1 all/selected/target, target-only 64-grid at center and union
across selected views, MV4 center Nearest64. MV5 inner (center and nearest
selected pose each side) vs full union. Additional candidate IDs vs center
and MV1 separately; MV1+MV5 union tests complementarity. Counts can differ;
no equal-budget precision claim. Additional source IDs are not guaranteed
new physical edges. View count counts distinct virtual views, not independent
sensor observations. Near/far/back can mix surfaces and source scans.

Per-view edge NPZ retains original accumulated indices, scan/return IDs,
ring/range/view xyz. source_features.npz stores anchor xyz for near-return
union and distinct virtual-view counts. Per-arm source_ids and projected
candidate_indices reference source_features. Per-view occupancy plots and
anchor projection overlays, pooled 3/5/10px DA2 proximity in summary.json.
No GT used to extract, no matching or optimization. DA2 is a proxy, not
physical ground truth. No real-data verification until local execution.
