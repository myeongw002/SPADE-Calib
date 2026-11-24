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
plane use identical support within each width, same camera centers and
point budget across widths. Sample capped returns per raw scan, A0
normal neighborhood gates, deterministic per-center balanced scan
sampling. Windows are raw frame radius, clipped to segment; camera
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
    --export-maps --oracle
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
