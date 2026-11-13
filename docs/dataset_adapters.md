# Dataset timestamp adapters (KITTI and custom recordings)

Hand–Eye, motion observability, and Geometry Calibration consume common
pose/point-cloud interfaces. They must not branch on dataset names.

## Canonical frames

- GLIM pose: `T_world_lidar`, LiDAR-to-its-odometry-world, metres.
- SfM pose: `T_sfm_world_camera` (camera-to-its-SfM-world), **unknown metric scale**.
- Public extrinsic: **`T_camera_lidar`**, transforming a LiDAR point into camera
  coordinates: `p_camera = R_camera_lidar @ p_lidar + t_camera_lidar`.
- Current Hand–Eye solver internally optimizes `T_lidar_camera` and also exports
  `T_camera_lidar`. Do not confuse its LiDAR-frame uncertainty coordinates with
  camera-frame observability; the latter requires a covariance coordinate change.
- SfM and GLIM world frames are independent. Do not apply one world transform
  directly to the other.

## Custom dataset: timestamped filenames

When images, scans and the GLIM pose stream have synchronized timestamp clocks,
use the original schema-v2 `recording.yaml`:

```yaml
schema_version: 2
dataset:
  name: custom_01
  root: /data/custom_01
inputs:
  images: images
  lidar_scans: scans
  glim: odom_lidar.txt
  sfm: colmap/sparse/0
  image_timestamps: image_timestamps.csv  # optional
  lidar_timestamps: lidar_timestamps.csv  # optional
matching:
  camera_to_lidar_offset_ns: 0
  max_pair_delta_ms: 100
pcd:
  time_field: time
  time_unit: s
  ring_field: ring
  range_m: [0.5, 100]
recursive: true
```

If filenames already use Unix nanoseconds, timestamp CSVs can be omitted.
Otherwise supply explicit timestamp CSVs using columns
`file,timestamp_ns` (or `filename,timestamp_ns`); paths are relative to
their respective image/scan roots, and the timestamps must be **integers in
the same clock domain** as GLIM after any declared clock offset.

```csv
file,timestamp_ns
frame_a.png,1791439637016960512
frame_b.png,1791439637116960512
```

The `spade-calib handeye --config ...` path now forwards the recording
indexer's image timestamp mapping into the common trajectory matcher. This
fixes prior failures when COLMAP image names were not numeric nanosecond stems.

If images/PCDs are not available and only poses exist, the same mapping works
in trajectory-only mode:

```bash
spade-calib handeye --glim odom_lidar.txt --sfm camera_poses.csv \
    --image-timestamps image_timestamps.csv --output handeye.json
```

## KITTI Odometry: indexed images, times.txt

**Preserve original KITTI filenames** (`000000.png`, `000001.bin`) and
`times.txt`. Each line of `times.txt` corresponds to the same **frame ID**
and records a timestamp in seconds relative to sequence start. The KITTI
adapter converts it to integer nanoseconds using decimal arithmetic.

For the historical KITTI 07 experiments, GLIM and camera TUM exports may
carry *synthetic* 0.1-second time increments, while KITTI `times.txt`
contains actual inter-frame durations. These are **not interchangeable clocks**.

Use the **frame-index adapter** instead of interpolating based on synthetic
timestamps:

```bash
spade-calib handeye \
    --glim /data/kitti/07/odom_lidar.txt \
    --sfm /data/kitti/07/colmap/sparse/0 \
    --kitti-times /data/kitti/07/times.txt \
    --output outputs/kitti07_handeye.json
```

The same flags work with `spade-calib select-scenes`. The adapter:
1. Reads COLMAP `images.bin`, `images.txt` or `camera_poses.csv`, extracting
   numeric frame IDs from COLMAP image names.
2. Assumes **GLIM row 0 corresponds to KITTI frame 0** and rows are
   consecutive; specify `--glim-first-frame N` if the export starts at N.
3. Matches images and LiDAR odometry by frame ID, even if COLMAP has missing
   images. Returns the shared `MatchedTrajectories` representation.
4. Uses KITTI `times.txt` for the resulting Motion Pair time intervals.
   GLIM file timestamps are validated as monotonic but intentionally not used
   for cross-sensor matching in this mode.

The legacy `camera_trajectory_tum.txt` is accepted **only** as a complete
full-frame export where row N corresponds to KITTI frame N. Without frame
IDs, an export that drops or reorders frames cannot be matched safely; use
native COLMAP `images.bin`/CSV with frame names instead.

A frame-index association does not estimate sensor synchronization or
correct motion distortion, and it must not be used for arbitrary asynchronous
custom recordings. For those use timestamps and interpolate GLIM poses.

## Dataset responsibilities

The dataset adapter only normalizes frame indexing, time metadata and sensor
pose conventions. The standard Hand–Eye solver, Jacobian observability,
covariance and geometry stages remain common.

KITTI GT poses and GT Extrinsics must **never** enter initial Hand–Eye
optimization; they are evaluation-only references. The reported uncertainty
under assumed motion noise and temporally overlapping motion pairs is not
a certified physical pose error.
