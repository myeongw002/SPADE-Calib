# DA2 inference and SfM metric optical-Z alignment

This is a standalone Python venv workflow; Docker is not required. Model
source and checkpoints remain outside the repository.

## Provenance and key distinctions

- Inference uses the OFFICIAL Depth Anything V2 relative model from
  https://github.com/DepthAnything/Depth-Anything-V2.
- Output is the raw float32 HxW relative prediction, NOT the official
  demo's 0-255 visualization and NOT metric depth.
- Alignment adapts the uploaded D1 archive's affine inverse-depth model:
  1 / Z_camera_m = a * q_DA2_raw + b, with a > 0.
- It uses SfM anchors reconstructed from COLMAP images.bin/txt and
  points3D.bin/txt, with the Hand-Eye SfM metric scale fixed.
- Train/heldout anchors occupy disjoint spatial tiles. Heldout error is
  only internal SfM consistency, NOT independently measured depth GT.
- Optional KITTI calib.txt is never read in the inference/alignment stage.

## Install in SPADE-Calib .venv on Ubuntu / RTX 5070

    cd ~/PROJECT/SPADE-Calib
    source .venv/bin/activate

Install an RTX-5070-compatible CUDA PyTorch build first. An example with
CUDA 12.8 wheels (verify your installed NVIDIA driver supports it):

    python -m pip install torch torchvision \
      --index-url https://download.pytorch.org/whl/cu128
    python -m pip install -e '.[research,depth,dev]'

Clone and install the official model dependencies outside the project:

    git clone https://github.com/DepthAnything/Depth-Anything-V2.git \
      ~/PROJECT/Depth-Anything-V2
    python -m pip install -r ~/PROJECT/Depth-Anything-V2/requirements.txt

Verify torch.cuda.is_available() and torch.cuda.get_device_capability(0)
before running a GPU job. Download the correct checkpoint for the chosen
encoder from the official project's README and keep it outside git.
External Python code and checkpoints must be trusted.

## 1. Infer raw DA2 (NO image min-max visualization normalization)

    spade-calib infer-depth \
      --image /data/kitti07/image_2/000105.png \
      --encoder vitl --input-size 518 --device cuda \
      --da2-repo ~/PROJECT/Depth-Anything-V2 \
      --checkpoint /local/checkpoints/depth_anything_v2_vitl.pth \
      --output outputs/da2/000105.npy

Outputs: 000105.npy (relative, float32), 000105.json (checkpoint SHA256,
encoder, device, raw min/max, source image and shape).

## 2. COLMAP to metric Camera-Z

The COLMAP model must be the SAME sparse reconstruction whose motion
was used in the Hand-Eye estimate. Do not mix a different COLMAP run
even if the image file name is identical.

    spade-calib align-depth \
      --relative-depth outputs/da2/000105.npy \
      --handeye outputs/handeye07.json \
      --colmap-model /data/kitti07/colmap/sparse/0 \
      --image-name 000105.png \
      --camera-json /data/kitti07/camera2_intrinsics.json \
      --output outputs/metric_depth/000105.npy

Camera JSON: same intrinsics, distortion and resolution as the actual
Camera image and COLMAP reconstruction. For rectified KITTI Odometry
Camera2: K = P2 left 3x3, zero distortion only if image rectified.
The field depth_type=camera_z_m denotes the intended output, not raw DA2.

Outputs:
- 000105.npy: metric Camera optical Z, with NaN at invalid depths.
- 000105.json: fitted affine inverse-depth a,b; train/heldout depth
  error; valid fraction; frame/model/scale provenance.
- 000105_anchors.npz: xy, depth_m, reprojection_error_px, image_shape,
  sfm_scale. Anchor tracks come from COLMAP, not KITTI LiDAR.

Alternatively, supply the uploaded legacy-style anchor NPZ directly:

    spade-calib align-depth \
      --relative-depth outputs/da2/000105.npy \
      --handeye outputs/handeye07.json \
      --anchors /data/kitti07/000105_sfm_sparse_metric.npz \
      --output outputs/metric_depth/000105.npy

The NPZ should include sfm_scale to check it matches Hand-Eye. Its
depth_m is already Camera-Z in metres, NOT unscaled SfM coordinates.

## 3. Reuse existing Ground pipeline

    spade-calib prepare-ground \
      --handeye outputs/handeye07.json \
      --cloud outputs/accumulated_105.npz \
      --metric-depth outputs/metric_depth/000105.npy \
      --camera-json /data/kitti07/camera2_intrinsics.json \
      --kitti-times /data/kitti07/times.txt \
      --frame-index 105 --depth-frame-index 105 \
      --glim /data/kitti07/odom_lidar.txt \
      --sfm /data/kitti07/camera_trajectory_tum.txt \
      --calib /data/kitti07/calib.txt --camera-index 2 \
      --run-refinement --allow-provisional \
      --output-dir outputs/ground_105

Read docs/ground_extraction_pipeline.md for required LiDAR cloud format.

## Safeguards and research limitations

- Model output direction must correlate positively with inverse metric
  depth. Mismatched direction or insufficient spatial anchor diversity
  aborts rather than silently forcing a fit.
- Geometry must share pixel coordinates and image rectification; the
  implementation checks COLMAP reprojection against the supplied K.
- Spatial tile holdout is not independent physical depth GT: tracks and
  metric scaling still come from one global SfM reconstruction.
- The 30 percent heldout median relative error maximum is a heuristic
  safety gate, not a statistically validated noise model.
- No KITTI07 inference, real-trajectory alignment quality or downstream
  GT error improvement is asserted by synthetic CI alone.
- The DA2 metric-depth models are separate checkpoints; this path is
  intentionally for standard relative DA2 checkpoints + SfM fitting.
