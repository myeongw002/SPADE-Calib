# L0: automatic local LiDAR plane intersections

Use the frozen M0 21-frame ghost-filtered .05m cloud, no new accumulation,
no image-guided extraction or calibration. All prepared anchors by default,
including unlabeled frames; --frames selects a subset. Compare 1m/2m grid
with 25% halo, repeat RANSAC up to 3 planes, SVD refit, finite intersection.
No vertical-only gate. Require 30 points per plane, 2D spread .10m, thickness
ratio <=.30, angle >=30deg (unoriented), within .7m of line, 15th percentile
near distance <=.25m, >=10 near points per plane from >=2 scan IDs in
shared 5–95% spans, length >=.75m and max along-line gap .5m.
The scan IDs do not imply independent samples. Greedy merge close
collinear fragments only if both planes match; refit pooled support and
revalidate. No guaranteed physical boundary test or global segmentation.
Ground retained by default so wall-ground and horizontal candidates are
not suppressed; horizontal-plane flag is descriptive. Optional
--remove-ground fits existing global ground extractor and removes all
points within .14m of its plane (can remove non-ground/slope points).
60m range cap and all thresholds provisional. No GT.

```bash
python scripts/experiments/run_lidar_line_extraction.py \
  --data outputs/lines07_M0_ghost --cells 1 2 \
  --out outputs/lines07_L0
```
Outputs per cell/frame: raw/merged JSON+NPZ (plane indices into original
M0 cloud), projection.png (original / auto cyan IDs / manual magenta),
cloud_lines.ply (gray PCD, cyan auto and magenta manual sampled points),
summary.json. Manual comparison uses same-cloud M0 line references,
angle <=10deg, mean symmetric infinite-line endpoint distance <=.30m,
reference overlap >=30%; not independent accuracy or precision. Inspect
unlabeled frames and false positives manually. PLY opens in CloudCompare
or Open3D; exact line endpoints in NPZ. No new dependencies.
