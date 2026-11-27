# S1-R scale oracle

Use same H archive and support as S1. Add --s1r --gt-calib and --gt-poses
instead of --s1. GT is deliberately used only in oracle arms and evaluation.
For each H1/H2: archive (objective check only), free warm-start refit,
fixed archived scale, fixed gauge-coordinate GT scale, fixed LS GT scale,
fixed gauge GT scale +/- --scale-bias-pct (default 1%).
Every refit starts from identical archived X, camera poses and landmarks;
fixed scales removed exactly from variable vector. Original free variable
bounds, gauge, weights and observations kept. Oracle values beyond original
scale bounds are permitted and explicitly flagged, not clipped.
Gauge ratio = GT anchor-local center component / original fixed baseline
component. LS ratio = sum(t_archive * t_GT) / sum(t_archive^2) in anchor
coordinates. Both multiplied by original scale reference. Neither is a
unique physical true scale when SfM is nonuniformly distorted. Check their
disagreement, trajectory RMSE, objective cost, active bounds, convergence,
and signed X translation error. Compare each fixed variant against
free_warm and fixed_archived, not just original H. GT scale is not deployable.
No new feature extraction or accumulation. No claim that information loss
proves actual calibration bias.
