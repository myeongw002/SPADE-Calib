# Provisional data interchange format (v0)

**Status: design draft.** Existing legacy files are not yet migrated. Shapes and field meanings below are contracts for future adapters, not a claim that the current ZIPs already conform.

## Principles

- Algorithm functions receive in-memory arrays/models rather than paths.
- File readers validate shape, unit, frame, timestamp domain, and finite values.
- Original point IDs and scan IDs are preserved when available; never synthesize `source_point_id` and call it original provenance.
- Store metadata next to each artifact; reject incompatible schema versions.
- Record a SHA-256 input checksum and configuration for experiments.

| Artifact | Proposed storage | Required data | Optional data |
| --- | --- | --- | --- |
| Camera | JSON | dimensions, K, distortion, frame_id, camera model | timestamps |
| LiDAR cloud | NPZ + JSON | xyz (N,3), frame_id, unit=m | source_point_id, scan_id, timestamp_ns, intensity |
| Metric depth | NPY + JSON | z (H,W), depth_definition=camera_z, unit=m | invalid mask |
| SfM anchors | NPZ + JSON | uv (M,2), depth_m (M,) | reprojection error, fold assignment |
| Camera plane | NPZ + JSON | normals (P,3), offsets_m (P,), masks (P,H,W) | centroids, support |
| Correspondences | NPZ + JSON | camera_plane_id, lidar_plane_id, matching policy | LiDAR support IDs, confidences |
| Extrinsic | JSON / NPY | T_camera_lidar (4,4), source/target frame | optimizer metrics |

Avoid creating enormous dense point-ID-by-plane matrices; store variable-length supports as keyed arrays or compact offsets.

## Example cloud metadata

```json
{
  "schema_version": "0",
  "frame_id": "lidar_reference",
  "units": "m",
  "source_point_id_available": false,
  "source_method": "PGO"
}
```

`source_method` describes how data were produced, not a mathematical guarantee of quality.

## Migration adapters needed

1. Legacy `Joint_xyz.npy` / `PGO_xyz.npy` -> PointCloud (without invented provenance).
2. Legacy `camera_planes.csv` + `camera_masks_320x180.npz` -> plane records (with mask resolution preserved).
3. Legacy Hand–Eye `T_handeye_only.txt` / YAML -> named RigidTransform.
4. `*_sfm_sparse_metric.npz` and `raw_da2.npy` -> SfM anchor and depth records.
5. Existing match result ZIPs -> candidate correspondence records.

**No migration adapter has been implemented in this bootstrap.**
