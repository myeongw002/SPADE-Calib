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
