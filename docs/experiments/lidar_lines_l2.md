# L2 scan-wise conditional stability

Use L0 raw 2m lines by default, existing fixed plane assignments and
M0 per-return retained scan IDs. No L1 prefilter, no rerun RANSAC,
no merge, no calibration. Optional L1 coverage join. Odd/even indices
in original chronological source frame list, first/second half of same
list, joint scan-block bootstrap 50 replicates. Both planes receive
same scan multiplicities. Weighted orthogonal PCA; minimum 15 unique
points, 2 distinct scan IDs, .10m minor in-plane spread, .30 thickness
ratio, 30deg plane intersection angle. Record insufficient support,
nonplanarity, degeneracy and insufficient overlap separately. Compare
split lines on common supported reference-axis interval >=30% of
original length, 21 stations. Bootstrap compares against full L0
reference on fixed central 30% and reports valid fraction and conditional
median/p90; failed draws not treated as zero or silently discarded.
No threshold calibrated; do not interpret as unbiased uncertainty.
Full-cloud assignment selection and voxel thinning couple splits; GLIM
systematic bias is shared.

```bash
python scripts/experiments/run_lidar_line_stability.py \
 --data outputs/lines07_M0_ghost --l0 outputs/lines07_L0 \
 --l1 outputs/lines07_L1 --out outputs/lines07_L2
```
L1 optional. Per-frame full metrics and bootstrap trial statuses,
summary.json, stability.csv, review.csv and original/projection images.
Random 12 FOV-projectable candidates per frame, plus reference best
candidates, review independently of stability. Label real_edge,
false_edge, or uncertain based on raw image/geometry as feasible;
projection alone cannot prove 3D identity. No automatic classification.
Reference labels are same-cloud; random sample includes uncertain cases
and duplicates from halo, not an independent sample of physical edges.
