# C0 camera-guided local LiDAR line pilot

No semantics, learned masks, CLAHE, or new accumulation. Use M0
frozen cloud and initializer; default all eight anchors. ELSED original
grayscale, reuse previous arrays for available anchors (pixel identity
and segment count checked); otherwise call pyelsed. Keep >=40px lines.
Project cloud, vote for 2m cells in finite +/-30px corridors with
20px endpoint extension, >=5 returns per cell per line. Separate depth
cells, fit all local halo points in selected cells, not just corridor
points. L0 RANSAC/PCA/intersection settings, no merge. Independent cell
RNG, so not bitwise comparable to archived L0 global extraction.
Associate both methods against same eligible ELSED lines (20px,
10deg, overlap .3, ambiguity gap .1). Guided winner must be among
its cell's triggers. Many 3D candidates per 2D line retained and counted,
not certified correspondence. Review random proposals plus separately
marked manual 3D overlap candidates. Manual 3D overlap is not 2D
association correctness. No dynamic object rejection, temporal tracking,
visibility/occlusion checks, or BA yet. This tests search restriction,
not guaranteed false-edge rejection.

```bash
python scripts/experiments/run_camera_guided_lines.py \
 --data outputs/lines07_M0_ghost --l0 outputs/lines07_L0 \
 --edge-data outputs/edge07_canny_elsed --out outputs/lines07_C0
```
Output summary.json, review.csv, per-frame comparison.png,
guided_lines JSON/NPZ with archived cloud point IDs, both matches JSON,
ELSED arrays and review overlays. Review label correct/incorrect/uncertain
based on physical identity as feasible; 2D projection alone is not proof.
