# Within-segment motion length diagnostic (after D0–D3)

Uses the compact trajectory_oracleXX/motion_pairs.npz alone. No new reconstruction or optimization.
GT effective pair scale = ||tb_gt||/||tb||; signed metric length error = s_k||tb||-||tb_gt||.
Also reports direction, radial and tangential vector error, per-segment scale dispersion and
within-segment correlations with time, speed and rotation. Frame-lag autocorrelation separates
strictly disjoint intervals from overlapping pairs.

Compare segment-constant and segment-linear log scale via contiguous blocked five-fold CV,
purging training intervals intersecting each test block. Both GT-supervised oracle and GLIM
proxy (R_X ta - (Rb-I)t_X) are reported. GT is diagnostic only. The GLIM proxy and archived
scale are conditional on full-data HE: not independent validation. Boundary extrapolation is
reported separately. An effective pair scale is not a consistent pose trajectory or proof of
scale drift; do not use these predictions as refined poses. Do not infer causality from correlation.

```bash
KITTI=/media/myungw00/256GB_SSD/kitti
for SEQ in 07 09 10; do
    python scripts/experiments/analyze_motion_length.py \
        --run "outputs/trajectory_oracle${SEQ}" \
        --out "outputs/motion_length${SEQ}" \
        --times "$KITTI/data_odometry_calib/sequences/${SEQ}/times.txt"
done
```

Outputs summary.json, motion_errors.csv and motion_diagnostics.png.
`--times` optional; without it time is measured in raw frame units.
