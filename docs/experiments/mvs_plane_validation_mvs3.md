# MVS3 — Fixed-GT LiDAR plane audit and multi-segment diversity

Continuation of MVS1/MVS2. No ZIP input, ICP, extrinsic optimization, or scale fitting.
Use existing KITTI calib.txt, camera-0 GT poses, original raw64 .bin scans and MVS outputs.
GT is evaluation-only. Use the existing SPADE environment (NumPy/SciPy).

## Segment 00 geometry audit (EXP-058)

```bash
cd ~/PROJECT/SPADE-Calib
KITTI=/media/myungw00/256GB_SSD/kitti
OPENBLAS_NUM_THREADS=1 python scripts/experiments/run_mvs_plane_validation.py validate \
  --planes outputs/mvs07_MVS2_seg00 \
  --report outputs/mvs07_pycolmap_seg00/segment_00/report.json \
  --calib "$KITTI/data_odometry_calib/sequences/07/calib.txt" \
  --gt-poses "$KITTI/data_odometry_poses/poses/07.txt" \
  --lidar-dir "$KITTI/data_odometry_velodyne/sequences/07/velodyne" \
  --sequence 07 --stride 5 --out outputs/mvs07_MVS3_seg00
```

Adjust paths to match extracted KITTI folders. Use actual image_2 frame IDs; Tr maps LiDAR to rectified camera 0 and P2 determines camera-2 shift. Separate anchor raw64 from two disjoint alternating groups of GT-pose accumulated scans; rigid scans, no deskew. Reference search constrained by finite MVS tangent footprint, +/-1.5 m offset slab and 20 degree orientation prior; fit candidate planes by LiDAR-only RANSAC/TLS, retain up to three. Candidate association requires physical review. Conditional holdout residuals can be selection-biased; report unconditional ROI residuals and support as well. Overlay colors: MVS orange, train ROI cyan, holdout ROI green, anchor ROI purple. Optional --extra-z-npz for the previous MVS2-D candidate if its sample indices match.

## Multi-segment audit (EXP-059)

Use existing CUDA PyCOLMAP environment for MVS1 and existing SPADE environment for MVS2. Preserve original parameters; run remaining segment IDs 1..7 with run_segment_mvs.py and run_mvs_planes.py, then:

```bash
python scripts/experiments/run_mvs_plane_validation.py diversity \
  --planes outputs/mvs07_MVS2_seg{00,01,02,03,04,05,06,07} \
  --out outputs/mvs07_MVS3_diversity
```

Reports per-segment and pooled G1+P1 and nonground normal SVD in local camera axes. Equal plane weights, 5% relative rank threshold; assumes correct correspondence and fixed R/scale. Rank alone is not calibration observability. Use new output folders to avoid mixing old overlays.

Local checks: direct KITTI07 calib/GT pose parsing (1101 poses), anchor transform identity consistency, Python compile and real MVS2 seg00 diversity. Raw64 end-to-end not yet run. No numerical calibration improvement claimed.
