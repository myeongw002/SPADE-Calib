# MV8 — Native edge validation with MV7 visibility
Run the same KITTI07 anchors, W21, input EP0 reference, B4c/GT, original
MV1 and MV7 extraction and evaluation. No optimizer or camera-guided selection.
    python scripts/experiments/run_native_edge_validation.py \
      --ring-calib /actual/path/to/velodynecalib_S2_factory_flatness_intensity.xml \
      --out outputs/edgepoints07_MV8 --save-mode minimal

A: native MV1 near IDs except newly surface-occluded returns. Surface mask
exactly as MV7, preserve target and unknown. Do not mistake outside anchor
range/FOV or unknown surface for occlusion. A_splat control, A_supported and
A_unknown diagnostic subsets. "Supported" means finite ray depth, not ground
truth visibility; protected target conflicts can remain.

B: retain native near ID if any original near/far/back triple has no newly
surface-occluded return. B_ghost additionally checks original target free-space
mask for all three. Both are source-support ablations, not necessary anchor
edge validity: a hidden original far point can be consistent with an anchor edge.

C: start from A, use the filtered MV7 anchor grid as lookup only. Candidate
itself is not replaced by a grid winner. Require occupied same cell with
absolute range agreement <= .2m (heuristic; --center-tol), then nearest
occupied cells on opposite sides of its row or column, actual candidate
relative angular gaps >0 and <=.8deg, range jump and foreground back support
as MV7. H/V separated; C_HV union. No synthetic point, no interpolation,
no camera evidence. Strict C rejects unknown/unsupported, not proof of invalid
edge. C_raw_HV uses the same A candidates and unfiltered grid to isolate
surface filtering of anchor support. C_unsupported and C_no_jump partition
failures relative to A. H/V describes angular jump axes, not camera line
orientation. Exact same-cell candidate agreement is a heuristic, not a
physical surface identity or independent observation test.

A/B/C are subsets of MV1 source near IDs. Lost subsets separately projected
(same EP0 camera z reference, may overlap pixels). Raw and Splat and Surface
H/V/HV exact MV7 controls. MV7_V_new_vs_MV1 and two union arms are separate
complementarity diagnostics, not evidence of observability; same ID only.
Output modes: minimal JSON; review NPZ and JPEG comparison; full adds grids,
raw masks, triples, per-arm overlays. Source index arrays refer to
source_features.npz; native_validation.npz stores original and anchor support.
No GT used for filtering. DA2 proximity is proxy only. No claim of
calibration accuracy, direction diversity, physical identity or generalization.
