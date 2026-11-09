# D1 legacy data adapters — implemented interface

The initial GitHub package can now **read and inspect** D1 artifacts without
loading them from ZIP archives into the Git working tree.

## Command

    pip install -e ".[dev]"
    spade-calib inspect --config configs/examples/d1.yaml --output outputs/d1_manifest.json

Before running, edit the example config to point at an extracted private D1
inputs directory. The CLI is read-only except for an explicitly provided
manifest output destination.

## Supported inputs

| Config key | Source type | Interpretation |
| --- | --- | --- |
| joint_cloud | Joint_xyz.npy | N x 3 numeric XYZ, explicit lidar frame |
| pgo_cloud | PGO_xyz.npy | N x 3 numeric XYZ, explicit lidar frame |
| handeye | T_handeye_only.txt OR handeye_original.yaml | T_camera_lidar, with YAML SfM scale if present |
| raw_depth | raw_da2.npy | raw depth-model output, **not** metric Z |
| metric_depth | *_da2_metric_aligned.npy | metric camera-Z with separate valid mask |
| sfm_anchors | *_sfm_sparse_metric.npz | xy, depth_m, reprojection_error_px, image_shape, optional sfm_scale |
| camera_planes | camera_planes.csv | plane ID, normal, offset, RMSE in centimeters, pixel count |
| camera_masks | camera_masks_320x180.npz | plane_<id> masks at **180x320**, not full image resolution |

Camera plane offsets are stored with n·x+d=0 and an explicit camera frame.
Mask pixel counts are checked against the CSV. Normal signs and point order
remain unchanged.

## Important D1-specific findings

- The recovered aligned metric-depth map contains nonfinite pixels, so the
  loader returns the raw numeric map **plus** a finite-and-positive validity
  mask. Invalid values are not filled, clamped or converted to zero.
- The original SfM archive uses the field name 'xy', not 'uv'.
- The point clouds only supply XYZ; no raw point provenance is invented.
- Hand-Eye CSV/text matrices and the historical YAML both load to the same
  named rigid-transform model.
- The plane mask resolution (180x320) differs from the depth resolution
  (720x1280). This is intentional; interpolation/rescaling is not done in I/O.

The inspector records counts, shapes and SHA-256 hashes but **does not**
optimize Calibration or claim semantic consistency of the LiDAR reference frame
with the extrinsic. Frame correctness must be verified from capture metadata.

## Testing

Synthetic tests run in public CI. An optional private dataset-shape regression
test can run after setting:

    SPADE_D1_CONFIG=/absolute/path/to/local-d1.yaml pytest -q

That opt-in test checks the known original D1 artifact counts (215504 cloud
points, 24 planes, 6050 SfM anchors); it does **not** yet reproduce the
calibration matrix, correspondences or solver statistics. Those numerical
regression gates are part of the next migration step.
