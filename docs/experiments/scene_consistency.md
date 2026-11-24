# G: segment sliding-window scene consistency (fixed poses)

For segment anchor a and camera i, A_i=L_a^-1 L_i, B_i=C_a^-1 C_i.
Raw return p_j is transformed q_ij=L_i^-1 L_j p_j. Compare A_i q_ij
against X^-1 B_i(s_k) X q_ij in anchor LiDAR coordinates.
Point-to-point uses same raw return IDs, point-to-plane projects that
same displacement onto frozen normals from archived A0 map PCA.
No ICP association, no RGB constraints, no pose refinement. The result
is a geometry-weighted anchored motion objective, not independent
cross-modal information; planar-motion degeneracy remains.

G0 archived lag-10 HE; G0_anchor_fixed/scale use SAME centers and anchor as
G1-G3 (componentwise archived HE loss, equal segment/center weights).
G1 fixed scales, G2 free segment scales (bounded log shifts), G3 same
as G2 with archived B4 ground offset (no normal). All share X. Point and
plane use identical support within each width, same camera centers across widths (same point budget only in legacy sampled mode). Legacy sampled mode uses capped returns per raw scan, A0
normal neighborhood gates, deterministic per-center balanced scan
sampling. Default voxel mode is described below. Windows are raw frame radius, clipped to segment; camera
centers are registered frames. Anchor excluded (zero residual).
Fixed support, compact exact per-center second moment factors, block
soft-L1 on normalized RMS spatial displacement, equal center/segment
weights. --linear is an unrobustified control. Ground weight and scene
sigma are exploratory, not calibrated covariances. Relative-motion
baseline uses its own normalization. No GT-based parameter tuning.
Reused returns are correlated; unique counts recorded. Source-cap and
points-per-center control computational budget, not a density claim.
A0 normals may be biased or include dynamic points. PLYs are sampled
maps with repeated returns, not full-resolution reconstructions.

```bash
KITTI=/media/myungw00/256GB_SSD/kitti
SEQ=07
python scripts/experiments/run_scene_consistency.py \
    --run "outputs/segment${SEQ}_trial1" \
    --ground-run "outputs/ground${SEQ}_B4c" \
    --out "outputs/scene${SEQ}_G" \
    --windows 1 3 5 11 \
    --gt-calib "$KITTI/data_odometry_calib/sequences/${SEQ}/calib.txt" \
    --image-dir "$KITTI/data_odometry_color/sequences/${SEQ}/image_2" \
    --support-mode voxel --local-voxel .18 --oracle
```

Omit --ground-run for G1/G2 only. --oracle uses GT X only as labeled
start with same HE-centered bounds and support; not a deployable arm.
Projection reuses existing A0/single-scan renderer, HE/B4c/GT and
successful G2/G3 at smallest/largest widths. Image is evaluation only.
Summary records baseline, rank, marginalized translation singular values,
axis GT errors, scale, bound hits and costs for each arm. Compare spatial
RMS on common support (point/plane costs not directly interchangeable),
GT and projection. Do not select best window/weight on GT as a deployed rule.
Pose/BA intentionally not implemented: scene consistency alone allows
arbitrary extrinsic to be absorbed by free poses.


## Voxel support (default; legacy sampled mode remains available)

Use all range-valid raw returns from the archived scan list. Transform
with original GLIM poses to each camera center's LiDAR coordinate frame,
then voxelize the merged window once (default archived voxel size, 0.18 m
in the original experiment). Keep the actual return nearest each voxel
centroid, not a synthetic point, to preserve exact return correspondence.
Apply the same fixed A0 normal gate AFTER voxel selection, to both point
and plane arms. Normals/validity can be cached before merging but no
return is removed by that gate before voxelization. No per-scan cap,
no per-center cap, no equal-count thinning in voxel mode. The same
centers are shared by all window sizes; equal center/segment objective
normalization is unchanged. This tests density/coverage without changing
relative weights by point count. It does not add independent camera
observations or estimate normals separately per local window.

Use `--support-mode sampled` to reproduce legacy G (512/128);
`--support-mode voxel --local-voxel .18` for the new arm.
The added G0_anchor_scale_ground is the anchored-motion + ground control.
Voxel mode stores compact factors and point counts by default, not
millions of per-center returns. Omit --export-maps for the first run;
it is optional and can consume substantial RAM/disk for full voxel support.
Summary reports raw accumulated counts, occupied voxels, retained counts,
unique original return IDs and per-center point counts.
