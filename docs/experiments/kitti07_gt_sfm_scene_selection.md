# KITTI07: GT-pose SfM Scene Selection experiment

## Purpose

The existing SPADE-Calib select-scenes ranks trajectory-motion diversity without reading SfM geometry. This experiment fixes **KITTI GT camera poses**, triangulates camera features **independently within each candidate window**, and ranks the resulting sparse SfM geometry. Only the camera poses are ground truth: triangulated points and scene quality scores are estimates, not 3D ground truth or extrinsic calibration accuracy.

KITTI Odometry inputs:
- dataset/sequences/07/times.txt
- dataset/sequences/07/calib.txt containing rectified P0 and P2 matrices; **Tr is not required** for camera-only GT SfM (as in the KITTI data_odometry_color download)
- dataset/sequences/07/image_2/000000.png, etc.
- dataset/poses/07.txt (T_world_rectified_camera0)
- locally installed COLMAP executable

Rectified Camera2 pose conversion:

    Ck = -inv(Kk) @ Pk[:,3]
    T_world_C2 = T_world_C0 @ Translation(C2-C0)
    T_C2_world = inverse(T_world_C2)

Camera2 intrinsics are fx,fy,cx,cy from the rectified P2; distortion parameters must NOT be estimated for these already-rectified images.

## Example: previous Windows 005 and 012

Run from repository root:

    python -m pip install -e ".[research,dev]"

Step 1. Prepare candidates and fixed camera GT:

    python scripts/experiments/run_kitti_gt_sfm_scene_selection.py prepare \
      --sequence /path/to/dataset/sequences/07 \
      --poses /path/to/dataset/poses/07.txt \
      --camera 2 --windows 005 012 \
      --out results/kitti07_gt_sfm_scenes

005 = [37.5, 52.5] seconds; 012 = [90, 105] seconds.

Step 2. Run independent COLMAP SfM for each scene:

    python scripts/experiments/run_kitti_gt_sfm_scene_selection.py reconstruct \
      --out results/kitti07_gt_sfm_scenes \
      --colmap colmap --matcher sequential --sequential-overlap 15

Each scene uses its own database and only its own frames for:
feature_extractor, sequential_matcher, point_triangulator, model_converter.
Feature-extractor database IDs are read from SQLite and matched by image name; no sequential IDs are assumed. Poses and camera intrinsics are verified unchanged (thresholds 1e-5 deg and 1e-6 m) after triangulation. No free bundle adjustment or mapper is run. Existing database/model directories cause an error rather than silent reuse.

Step 3. Score and select:

    python scripts/experiments/run_kitti_gt_sfm_scene_selection.py evaluate \
      --out results/kitti07_gt_sfm_scenes \
      --min-track 3 --min-parallax-deg 1 \
      --max-reproj-px 2 --min-qualified-points 100 --max-scenes 2

A qualified 3D point has >=3 tracked observations, >=1 degree maximum view-ray parallax, and <=2 pixel median reprojection error calculated directly from **fixed GT poses** and KITTI P2.

Outputs:
- manifest.json: exact scene candidates and input paths
- <scene>/scene.json, image_list.txt, gt_T_world_camera.npy
- <scene>/database.db, model_gt/, sparse/, sparse_txt/ and COLMAP command logs
- scene_scores.csv: triangulated_points, qualified_points, qualified_points_per_image, median_track_length, median_parallax_deg, parallax_p10_deg, median_reproj_error_px, reproj_p95_px, mean_image_grid_coverage, geometry_score_heuristic
- scene_selection.json: eligible candidates, ranking, selected disjoint scenes, caveats

The heuristic score is:

    log(1 + qualified_points / num_images)
    * log(1 + median_parallax_deg)
    * mean_image_grid_coverage
    / (1 + median_reproj_error_px)

This is **not** a Fisher Information Matrix or guaranteed extrinsic utility. The score is affected by textures, image resolution, thresholds, moving objects, correspondence reliability and 3D depth uncertainty. Keep the number of input frames comparable when judging candidate windows.

## Sliding-window experiment

Optional sliding candidates with a capped equal image budget:

    python scripts/experiments/run_kitti_gt_sfm_scene_selection.py prepare \
      --sequence /path/to/dataset/sequences/07 \
      --poses /path/to/dataset/poses/07.txt \
      --sliding --start-s 0 --stop-s 110 \
      --window-s 15 --stride-s 7.5 --max-images 60 \
      --out results/kitti07_gt_sfm_sliding

Reconstruct/evaluate the sliding directory using the same commands as above, changing --out. Note that candidate windows are reconstructed separately, so a large number of overlapping windows is computationally expensive. Non-overlapping selection is enforced in evaluation.

## Experimental interpretation

1. GT-pose SfM removes the camera pose-estimation variable but does not create ground-truth points or guarantee surface accuracy.
2. Candidate scenes are reconstructed independently to avoid use of out-of-window feature tracks.
3. Raw SfM point counts are biased toward textured surfaces, not necessarily camera/LiDAR common surfaces.
4. Reprojection error alone cannot validate 3D accuracy or extrinsic calibration accuracy.
5. The experiment provides a **camera geometry-only** scene-selection baseline. Existing motion-only select-scenes, estimated-pose SfM, actual GT LiDAR-pose scene quality, and downstream KITTI camera–LiDAR extrinsic evaluation require independent follow-on comparisons.
6. Real KITTI data and COLMAP must be run on the user's machine. CI only tests synthetic invariants, not real-world reconstruction quality.

Tests:

    python -m pytest tests/test_kitti_gt_sfm_scene_selection.py -q

Official COLMAP known-pose reconstruction guidance:
https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses


## Global SfM vs Local SfM: full KITTI07 scene-selection comparison

Four NEW operations are available through the SAME experiment script. Keep the
same prepared candidate manifest for both evaluations. A new output directory
is recommended; never mix incompatible old experiments.

1. Prepare a sequence-wide sliding-window candidate set with equal local frame budgets:

       python scripts/experiments/run_kitti_gt_sfm_scene_selection.py prepare \
         --sequence /media/myungw00/256GB_SSD/kitti/data_odometry_color/sequences/07 \
         --poses /media/myungw00/256GB_SSD/kitti/data_odometry_poses/poses/07.txt \
         --camera 2 --sliding --window-s 15 --stride-s 7.5 \
         --max-images 60 --out results/kitti07_gt_sfm_compare

   The candidate windows use at most 60 sampled camera frames each. The global
   reconstruction deliberately uses **ALL KITTI07 camera frames**, not merely
   the union of those candidate samples. Therefore this comparison is
   information-imbalanced by design: it measures the effect of unrestricted
   external temporal observations.

2. Construct ONE fixed-pose global SfM reconstruction:

       python scripts/experiments/run_kitti_gt_sfm_scene_selection.py reconstruct-global \
         --out results/kitti07_gt_sfm_compare \
         --colmap colmap --matcher sequential --sequential-overlap 15

   The output is stored under global/ (image_list.txt, GT camera pose matrices,
   model_gt/, database.db, sparse/, sparse_txt/, four COLMAP logs). All global
   input images are checked against the KITTI timestamps and camera intrinsics.
   Poses remain fixed. A pre-existing global directory is rejected to prevent
   accidental reuse of stale databases or partial models.

3. Score candidate windows from global 3D tracks:

       python scripts/experiments/run_kitti_gt_sfm_scene_selection.py evaluate-global \
         --out results/kitti07_gt_sfm_compare \
         --min-track 3 --min-parallax-deg 1 --max-reproj-px 2 \
         --min-qualified-points 100 --max-scenes 2

   Uses **only the selected window images' observations** for track length,
   parallax angle, reprojection error and 2D grid coverage, but the global 3D
   point coordinates were triangulated from all-sequence observations.
   Qualified points with observations OUTSIDE the candidate are separately
   reported. This does **NOT** make it window-only triangulation.
   Outputs: global_scene_scores.csv, global_scene_selection.json.

4. Run local window reconstruction and evaluation on the SAME candidate windows:

       python scripts/experiments/run_kitti_gt_sfm_scene_selection.py reconstruct \
         --out results/kitti07_gt_sfm_compare \
         --colmap colmap --matcher sequential --sequential-overlap 15

       python scripts/experiments/run_kitti_gt_sfm_scene_selection.py evaluate \
         --out results/kitti07_gt_sfm_compare \
         --min-track 3 --min-parallax-deg 1 --max-reproj-px 2 \
         --min-qualified-points 100 --max-scenes 2

5. Compare rankings and top-K matches:

       python scripts/experiments/run_kitti_gt_sfm_scene_selection.py compare \
         --out results/kitti07_gt_sfm_compare --top-k 3

   Outputs:
   - scene_comparison.csv: candidate IDs, two score/rank columns, differences
     in qualified points, reprojection errors, parallax and grid coverage,
     plus global tracks with outside-window observations.
   - scene_comparison.json: chosen nonoverlapping windows per strategy,
     selected-set Jaccard, Top-K overlap, descriptive rank correlation and
     warnings about global information leakage.

The compare command refuses incompatible evaluation thresholds, image counts,
candidate names and temporal windows. This prevents silently comparing two
different experiments. If local reconstruction does not finish for all windows,
reconstruct subsets with --only scene_ID ...; however evaluate and compare need
a complete candidate set. Both reconstructions may be expensive, especially
full-sequence COLMAP features and matching.

### Interpretation and future controlled experiment

The Global reconstruction has information from frames outside each candidate.
A scene's good Global score therefore does **not** prove it independently
supports well-conditioned triangulation. This implementation provides (A) Local
SfM and (B) Global XYZ with in-window track filtering. It does NOT yet perform
(C) new triangulation of Global-matched tracks using only in-window observations;
that is a separate controlled baseline.

The two earlier Window005/012 presets can be used to debug execution, but with
max_scenes=2 both are selected if eligible. Use sliding candidates for
meaningful selection-agreement analysis. None of these steps measures
camera–LiDAR GT extrinsic calibration error.
