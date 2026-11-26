# L1 frozen-line support ablation

Reuses M0 cloud point identities and L0 raw/merged JSON/NPZ. No rerun
of plane fitting, no new matching, no merge modifications.
Default cell 2m. Baselines raw and merged; evaluate same parent geometry
with symmetric nearest-point tests inspired by Yuan et al. (2021),
livox_camera_calib calcLine. Independent implementation, not a reproduction:
L0 finite endpoints (not voxel-clipped full intersection), already passed
L0 gates. Both endpoints included at <=5mm spacing; thresholds (m)
.03/.06, .05/.10, .10/.20, .15/.30. Report original-like >10 accepted
sample count separately; output strictly contiguous spans >=.30m, never
bridge rejected samples. This minimum is exploratory, not original.
Lengths, counts and coverage are not precision. Manual reference union
overlap computed only for candidate spans with <=10deg and <=.30m mean
reference cross-section distance; individual best requires >=30% overlap.
Paired merge diagnostic compares independently chosen best raw/merged
on their common finite interval, not tracked ancestry. Same-cloud references
are not GT. Keep density/threshold confounds separate from future W sweep.

```bash
python scripts/experiments/run_lidar_line_support.py \
    --data outputs/lines07_M0_ghost \
    --l0 outputs/lines07_L0 --out outputs/lines07_L1
```
Outputs summary.json, counts.csv, per-frame raw/merged comparison.png,
per-threshold metrics JSON and spans NPZ (with parent IDs). Optional
--save-ply. No new dependencies. L0 source arrays were not hashed in its
summary; hashes recorded at L1 consumption, not historical integrity proof.
