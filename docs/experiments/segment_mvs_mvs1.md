# MVS1 — Per-scale-segment dense reconstruction

Use archived global COLMAP poses and segmented_handeye.json image_names/scale.
No local SfM, bundle adjustment, LiDAR/GT guidance, scale refit, or alignment.
Select exact segment membership, optional image stride retains anchor and
endpoints; default all registered images. Convert sparse model to TXT once,
prune tracks to selected views, keep >=2-view points, preserve original
camera poses, point XYZ and image observation indices. Legacy monocular
COLMAP model (not multi-camera rigs). Undistort using COLMAP, verify pose
and image membership. OpenMVS InterfaceCOLMAP + DensifyPointCloud, default
full resolution level0, max image size1600, 8 estimation views, 3 fusion
views; installed version default CPU/GPU unless --cuda-device specified.
Optional COLMAP backend uses PatchMatch geometric consistency and fusion;
same undistorted sparse model, not identical internal view selection or
fusion meaning. No mesh reconstruction. Keep raw depth maps and dense PLY.
Use existing Open3D for PLY reading/conversion; no need for pycolmap.

    python scripts/experiments/run_segment_mvs.py \
      --image-dir /actual/COLMAP/image/root \
      --run outputs/segment07_trial1 --segment-id 0 \
      --out outputs/mvs07_MVS1_seg00

Use --prepare-only to validate and export submodels without MVS or Open3D.
Omit --segment-id for all archived HE segments; --contains-frame 546
selects segment by frame interval, not a new anchor. --image-stride 3
optional smoke test, not the full experiment. --backend both comparison
or --backend colmap. --sfm overrides relocated model, must be identical
to archived run; no archived hash available. Supply --interface/--densify
full paths if not on PATH. All commands/logs saved, nonzero status on failed
or partial run, fresh output required. Verify installed command help if
version differs. MVS outputs assumed in original SfM world gauge (no
alignment/normalization); metric transform x=s(Rcw*x_world+tcw), normals
rotate only, same anchor as prior segment reconstruction. No separate
segment cloud fusion. Check dense-vs-sparse position and scale before
plane fitting. Point count is not plane accuracy; no GT comparison yet.
