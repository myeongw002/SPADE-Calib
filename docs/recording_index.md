# Whole-recording dataset layer (schema_version: 2)

This is a **pre-scene-selection** index, not a scene-specific Calibration dataset. It is an independent extension of the existing `schema_version: 1` D1 artifact inspector; `spade-calib inspect` remains unchanged. No DA2/GLIM/SfM processes are invoked.

## Required source inputs

- `images`: directory of RGB files with **nanosecond timestamp stems** (17–20 decimal digits) or an explicit timestamp CSV.
- `lidar_scans`: directory of raw `.pcd` or KITTI Odometry `.bin` files with the same timestamp policy. Index-named KITTI files require an explicit timestamp CSV for this generic recording path.
- `glim`: `traj_lidar.txt`/`glim_trajectory.txt` trajectory file or directory with exactly one recognized trajectory file. Lines: `timestamp_seconds x y z qx qy qz qw`; frame convention: `T_world_lidar`.
- `sfm`: directory containing **one** COLMAP/pycolmap sparse reconstruction (registered `images.bin` / `images.txt`), or a `camera_poses.csv` exported with columns `image_id,image_name,camera_id,qw,qx,qy,qz,tx,ty,tz`. Pose convention: `T_camera_sfm_world`; translation is **not assumed metric**.
- `camera_calibration` optional for indexing, required later for projection.

Paths may be absolute or relative to `dataset.root`, which itself is relative to the YAML file or absolute. `sfm` directory discovery fails if multiple reconstruction directories are present instead of selecting one arbitrarily. A COLMAP image not registered (no pose) remains in the index.

## Execute

```bash
spade-calib index --config configs/examples/recording.yaml --output outputs/recording_index.json
```

The JSON manifest stores paths, sorted timestamp records, GLIM coverage, registered SfM camera pose **without inferred scale**, closest LiDAR per camera frame, pair time residual and summary counts. The fingerprint hashes metadata (path, size, mtime and matching settings), **not source-file content**; it cannot be used as a cryptographic file-integrity test.

### Timestamp association

All filename timestamps are interpreted as **integer nanoseconds**, not guessed based on digit count beyond validation. CSV mapping format:

```csv
file,timestamp_ns
frames/frame_01.jpg,1791439637016960512
frames/frame_02.jpg,1791439637116960512
```

Image files are associated with their nearest LiDAR frame using `t_lidar = t_camera + camera_to_lidar_offset_ns`, only within `max_pair_delta_ms`. Ties resolve to the earlier scan. LiDAR records can be reused by multiple images; no one-to-one pairing is implied. The camera–LiDAR clock offset **is never estimated**. Missing associations and SfM poses are reported; no synthetic timestamps or poses are generated. The same basename in multiple directories is permitted only when SfM names match exact relative paths. `glim_covered` tests only the **frame timestamp** bounds; it does not prove per-return deskew coverage.

### Lazy LiDAR PCD and KITTI .bin readers

`RecordingIndex.load_lidar_scan(i)` reads one PCD and returns existing `spade_calib.lidar.scan_io.LidarScan`, keeping original filtered return indices. Supports uncompressed `DATA binary` and `DATA ascii` with standard numeric scalar x/y/z/time fields and optional ring. The `time_field` is **scan-relative** and can be converted from `s/ms/us/ns`; PCDs with absolute per-return timestamps, `binary_compressed`, LAS/LAZ, ROS bag topics or vendor-specific layouts require a dedicated adapter. Missing ring is marked as `uint16=65535`, not ring 0. Missing per-return time in a **PCD** raises, rather than pretending
all returns were captured at the scan timestamp. The separate KITTI `.bin`
reader supports measured XYZ/reflectance and sets missing per-return time and
ring explicitly to `None`. The `LidarScan` and `AccumulatedCloud` models
retain optional reflectance; no automatic intensity normalization occurs.
A KITTI scan can be accumulated with `mode='rigid'` (scan-level pose only).
Attempting `mode='deskew'` when any scan has missing per-point time raises
explicitly. For indexed KITTI with synthetic GLIM pose timestamps, use
`spade-calib accumulate-kitti` instead of timestamp interpolation; see
[dataset adapters](dataset_adapters.md#kitti-raw-velodyne-bin-and-rigid-accumulation).

### Interface boundary

```python
from spade_calib.io.recording import load_recording_config, index_recording
recording = index_recording(load_recording_config("recording.yaml"))
all_frames = recording.associations
window = recording.lidar_window(start_ns, stop_ns)
scan = recording.load_lidar_scan(0)  # on-demand, not during indexing
```

The next module (trajectory observability/scene selection) should consume this index and choose windows. Only after a scene is chosen should heavy DA2, point-cloud deskew/refinement, plane extraction and Calibration execute. No assumption that the COLMAP and GLIM world frames have the same origin, orientation, or metric scale is made here.