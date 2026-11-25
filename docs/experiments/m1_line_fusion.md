# M1: line fusion with fixed or jointly optimized camera states

Run run_local_visual_ba.py with --line-data outputs/lines07_M0_ghost,
--reference-ba outputs/local_ba07_H, --window 5, same H inputs.
M1a is the exact archived M0 initializer (H2), M1b XZ-only post-fit.
M1c fixed original local SfM + free X/8 scales + HE/Ground.
M1d free local pose/landmark/scale/X + HE or Scene W5/Ground.
All M1c/d start from archived HE X, same H visual support, gauge,
settings and bounds. Compare each to its own w0 control. Positive line
weights default .01,.1,1, sigma 2px, not calibrated noise or optimized
hyperparameters. Each line uses 5 fixed samples, equal weight across
records. Line cost = lambda/(2*5*N) sum soft_l1((d/sigma)^2) plus
positive depth barrier. Line uses synchronous target LiDAR coordinates
and raw image label, only X enters, NOT camera BA pose or scale.
No changing matches or support in optimization. Original manual train/test
split enforced. Optional --line-holdout-frame for exploratory leave-one-out;
may share same feature IDs (reported); not independent scene evaluation.
GT evaluation only. Do not select lambda based on GT or holdout.
No adaptive accumulation, no LiDAR pose refinement. No extra shell file.
Full BA requires original COLMAP intrinsics and observations; archived H
NPZ alone doesn't contain the camera intrinsics/distortion model.
