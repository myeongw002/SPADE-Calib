# D0–D3 trajectory oracle (diagnostic only)
Freeze the archived segmentation, frame identities, selected motion pairs,
HE settings and initializer. D0 GLIM/SfM reproduces archived HE; D1 GT LiDAR/SfM
refits segment scales; D2 GLIM/GT Camera and D3 GT/GT fix metric scale to one.
GT poses are camera0-to-world; camera2 baseline and Tr applied by conjugation.
D3 is circular (known extrinsic constructs both GT streams), useful for
solver/conditioning checks only. Ground excluded. GT-R-fixed controls,
GT-fixed-X scale profiling, local translation Jacobian after marginalizing
rotation/scale, unweighted and frozen IRLS bias, per-segment contributions.
Camera motion+scale and LiDAR translation decompose exactly at common
GT X/scale; rotation residual kept combined. Not causal attribution or
covariance. D1/D2 error may cancel or interact; compare carefully.

```bash
KITTI=/media/myungw00/256GB_SSD/kitti
for SEQ in 07 09 10; do
    python scripts/experiments/run_trajectory_oracle.py \
        --run "outputs/segment${SEQ}_trial1" \
        --out "outputs/trajectory_oracle${SEQ}" \
        --gt-poses "$KITTI/data_odometry_poses/poses/${SEQ}.txt" \
        --gt-calib "$KITTI/data_odometry_calib/sequences/${SEQ}/calib.txt"
done
```
Paths from original config; override --glim --sfm --sequence if needed.
Outputs summary.json and compact motion_pairs.npz (original/GT selected
relative motions, frame IDs and scales) for offline analysis.
