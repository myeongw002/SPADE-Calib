# S0-R multi-scale synthetic controls

python scripts/experiments/run_scale_geometry_s0r.py --out outputs/S0_R --noisy-trials 5

Two motion/layout families (planar with vertical lines, general with diverse
lines), three known segments, constant scale vs piecewise scale 1.08/1.2/1.32.
Ground offset factor in all arms. Global vs free multi-scale vs fixed oracle
scale. Motion-only, anchor line, camera multi-view, LiDAR multi-view.
Matched camera/line observations, weights and initial X for all arms in a
trial. Two paired noise regimes perturb both trajectories (higher LiDAR
noise sensitivity). Exact anchor preserved; camera center perturbation
expressed in metric units then divided by true scale. Independent
anchor-relative noise, not a real odometry covariance, no BA, no scan
accumulation, known segment boundaries. Ground normals exact.
Noiseless once, five seeds per noisy condition by default. Seed varies
both measurement noise and initial X, not independent factorial starts.
Information at fitted state; global model under scale drift has no exact
truth solution. Oracle uses true scales only for diagnosis.
Report failures and bounds; do not select weights from GT or call
unwhitened/weighted GN information a statistical covariance.
