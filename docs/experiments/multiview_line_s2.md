# S2 fixed-state multiview line pilot

Use existing S2 matching prepare/review/finalize. Explicit human review
required, no auto-accept. Only nonanchor views in exact same archived
BA segment support enter, original M0 manual 2D anchor used for every
track; source ID defines physical track. Test split excluded from fit.
At least one train track with nonanchor view required. Report skipped
views. H2 W5 original archive cost verified with existing S1 machinery.
Fixed H2 camera poses and landmarks, same original HE bounds and scale
reference, warm X/scales from H2. Same motion/ground objective as H2
with constant visual cost omitted. Baseline no line; anchor-only;
GLIM multiview and camera multiview same observations; camera with
scales fixed to H2. Equal weight per physical track, equal views within
track, 5 samples per observation. Signed pixel distance and depth barrier.
Default weights .01/.1/1; no GT-based selection. All arms evaluated on
both multiview residuals. Same-state scale-coupling information (raw and
robust) at archived initializer, no statistical covariance claim.
No tracking/automatic matching development or adaptive accumulation yet.
One or two tracks on one sequence can only demonstrate feasibility.

```bash
python scripts/experiments/run_line_matching.py finalize \
 --data outputs/lines07_S2_matching --out outputs/lines07_S2_reviewed.json
KITTI=/media/myungw00/256GB_SSD/kitti
python scripts/experiments/run_local_visual_ba.py \
 --run outputs/segment07_trial1 --ground-run outputs/ground07_B4c \
 --scene-factors outputs/scene07_H_factors --reference-ba outputs/local_ba07_H \
 --s2-m0 outputs/lines07_M0_ghost --s2-reviewed outputs/lines07_S2_reviewed.json \
 --window 5 --max-nfev 120 --line-weights .01 .1 1 \
 --gt-calib "$KITTI/data_odometry_calib/sequences/07/calib.txt" \
 --gt-poses "$KITTI/data_odometry_poses/poses/07.txt" --out outputs/lines07_S2_fit
```
Review CSV decisions accept/reject/skip. Ambiguous candidates require
chosen_index. Do not accept uncertain matches. No GT for review.
