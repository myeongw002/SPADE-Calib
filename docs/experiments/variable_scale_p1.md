# P1 — Variable segment scale, motion-only control

Use original adaptive partition, lag and frozen pair mask from segmentXX_trial1.
No BA, ground, line or additional pose variables. Camera orientations fixed.
Normalize actual timestamp per segment to [-1,1]; adjacent edge midpoint.
Integrate SfM world center increments with exp(a_k+b_k*tau_mid), express
in camera i for every original selected pair. b=0 exactly recovers original
model; archived robust cost must reproduce. No re-gating. Default slope
guard |b|<=.25 (end/start ratio at most exp(.5)); heuristic, report hits.
Same HE whitening and robust loss, same start per matched control.
Full fits warm-start at original archive; blocked central-half fits use
training-only closed-form HE and OLS; boundary-crossing pairs excluded.
Frozen original selection/partition was estimated on full data, so blocked
validation is conditional, not fully independent. Report failures; no
GT for fit or initialization. GT loaded after fits for calibration and
camera relative translation evaluation. Same-state raw/frozen IRLS
Jacobian diagnostics for extrinsic and slopes; no covariance claims.
Do not tune slope bounds on GT. If motion improves but calibration does
not consistently improve, stop additional scale-model development.

```bash
KITTI=/media/myungw00/256GB_SSD/kitti
for SEQ in 07 09 10; do
 python scripts/experiments/run_variable_scale.py \
 --run outputs/segment${SEQ}_trial1 \
 --gt-calib "$KITTI/data_odometry_calib/sequences/${SEQ}/calib.txt" \
 --gt-poses "$KITTI/data_odometry_poses/poses/${SEQ}.txt" \
 --out outputs/scale${SEQ}_P1
done
zip -r scale_P1.zip outputs/scale{07,09,10}_P1
```

Original full synchronized GLIM/COLMAP/times inputs required; lag-only
motion archives cannot reconstruct adjacent increments in general.
