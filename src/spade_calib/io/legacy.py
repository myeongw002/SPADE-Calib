"""Strict readers for the recovered D1 experiment files.

Each reader accepts an explicit path and coordinate frame. No scene ID, sensor
intrinsics, working directory, or private data path is hard-coded here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray

from spade_calib.io.models import Plane, PointCloud, RigidTransform


@dataclass(frozen=True)
class LegacyDepth:
    """Original depth and validity map. Invalid pixels are NOT inpainted."""

    values: NDArray[np.float64]
    valid: NDArray[np.bool_]

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def valid_fraction(self) -> float:
        return float(self.valid.mean())


@dataclass(frozen=True)
class LegacySfMAnchors:
    """SfM camera-Z anchors expressed in the original image pixel grid."""

    xy: NDArray[np.float64]
    depth_m: NDArray[np.float64]
    reprojection_error_px: NDArray[np.float64]
    image_shape: tuple[int, int]
    sfm_scale: float | None

    @property
    def count(self) -> int:
        return int(self.depth_m.size)

    def quality_mask(self, max_reprojection_error_px: float = 3.0) -> NDArray[np.bool_]:
        if max_reprojection_error_px <= 0:
            raise ValueError("max_reprojection_error_px must be positive")
        return self.reprojection_error_px < max_reprojection_error_px


@dataclass(frozen=True)
class LegacyCameraPlane:
    plane_id: int
    plane: Plane
    mask: NDArray[np.bool_]
    pixel_count: int
    fitting_rmse_m: float


@dataclass(frozen=True)
class LegacyHandEye:
    extrinsic: RigidTransform
    sfm_scale: float | None


def load_xyz_cloud(path: str | Path, *, frame_id: str) -> PointCloud:
    """Read Joint_xyz.npy/PGO_xyz.npy without inventing original point IDs."""
    array = np.load(Path(path), allow_pickle=False)
    if not isinstance(array, np.ndarray):
        raise ValueError("Expected a .npy point cloud, not a .npz archive")
    return PointCloud(xyz=array, frame_id=frame_id)


def load_depth(path: str | Path, *, valid_min_m: float = 0.0) -> LegacyDepth:
    """Read a 2D camera-Z map; preserve NaN/Inf and construct a validity mask.

    The same reader can inspect the raw DA2 array, but interpreting raw DA2 as
    metric camera-Z depth is NOT supported; the caller must label its meaning.
    """
    if not np.isfinite(valid_min_m) or valid_min_m < 0:
        raise ValueError("valid_min_m must be a finite nonnegative number")
    array = np.load(Path(path), allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("Depth input must be a 2D .npy array")
    if array.dtype.kind not in "fiu":
        raise ValueError("Depth array must have a numeric dtype")
    values = np.asarray(array, dtype=np.float64).copy()
    valid = np.isfinite(values) & (values > valid_min_m)
    return LegacyDepth(values=values, valid=valid)


def load_sfm_anchors(path: str | Path) -> LegacySfMAnchors:
    """Read the exact D1 *_sfm_sparse_metric.npz field names."""
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {"xy", "depth_m", "reprojection_error_px", "image_shape"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Missing SfM fields: {sorted(missing)}")
        xy = np.asarray(archive["xy"], dtype=np.float64).copy()
        depth = np.asarray(archive["depth_m"], dtype=np.float64).copy()
        error = np.asarray(archive["reprojection_error_px"], dtype=np.float64).copy()
        shape = np.asarray(archive["image_shape"])
        scale = float(archive["sfm_scale"]) if "sfm_scale" in archive.files else None

    if shape.shape != (2,) or not np.issubdtype(shape.dtype, np.integer):
        raise ValueError("image_shape must be [height, width]")
    height, width = int(shape[0]), int(shape[1])
    n = len(depth)
    if height <= 0 or width <= 0:
        raise ValueError("SfM image dimensions must be positive")
    if xy.shape != (n, 2) or depth.shape != (n,) or error.shape != (n,):
        raise ValueError("Incompatible SfM anchor array shapes")
    if (
        not np.isfinite(xy).all()
        or not np.isfinite(depth).all()
        or not np.isfinite(error).all()
        or (depth <= 0).any()
        or (error < 0).any()
    ):
        raise ValueError("SfM anchors contain invalid values")
    if ((xy[:, 0] < 0) | (xy[:, 0] >= width) | (xy[:, 1] < 0) | (xy[:, 1] >= height)).any():
        raise ValueError("SfM anchor pixel coordinates fall outside image_shape")
    if scale is not None and (not np.isfinite(scale) or scale <= 0):
        raise ValueError("SfM metric scale must be finite and positive")
    return LegacySfMAnchors(
        xy=xy, depth_m=depth, reprojection_error_px=error,
        image_shape=(height, width), sfm_scale=scale,
    )


def load_camera_planes(
    csv_path: str | Path,
    masks_path: str | Path,
    *,
    frame_id: str,
) -> tuple[LegacyCameraPlane, ...]:
    """Read the legacy camera_planes.csv and plane_<id> mask archive together.

    The CSV pixel count must match the number of nonzero mask pixels. The
    source-plane sign convention is preserved (only unit-normal normalization
    from the core Plane model is applied).
    """
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "id", "pixels", "plane_rmse_cm", "normal_x",
            "normal_y", "normal_z", "offset",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Camera plane CSV requires columns {sorted(required)}")
        rows = list(reader)

    if not rows:
        raise ValueError("Camera plane CSV is empty")
    results: list[LegacyCameraPlane] = []
    seen: set[int] = set()
    with np.load(Path(masks_path), allow_pickle=False) as archive:
        for row in rows:
            try:
                plane_id = int(row["id"])
                pixels = int(row["pixels"])
                rmse_m = float(row["plane_rmse_cm"]) / 100.0
                normal = np.array(
                    [float(row["normal_x"]), float(row["normal_y"]), float(row["normal_z"])],
                    dtype=np.float64,
                )
                offset = float(row["offset"])
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Malformed camera plane CSV row: {row!r}") from exc
            if plane_id < 0 or plane_id in seen:
                raise ValueError(f"Invalid/duplicate plane ID: {plane_id}")
            seen.add(plane_id)
            if pixels <= 0 or not np.isfinite(rmse_m) or rmse_m < 0:
                raise ValueError(f"Invalid pixel count or RMSE for plane {plane_id}")
            key = f"plane_{plane_id}"
            if key not in archive.files:
                raise ValueError(f"Missing camera plane mask: {key}")
            source_mask = np.asarray(archive[key])
            if source_mask.ndim != 2 or source_mask.dtype.kind not in "buif":
                raise ValueError(f"Camera mask {key} must be a numeric 2D array")
            if not np.isfinite(source_mask).all():
                raise ValueError(f"Camera mask {key} contains non-finite values")
            mask = source_mask != 0
            if int(mask.sum()) != pixels:
                raise ValueError(
                    f"Plane {plane_id}: CSV pixels={pixels} but mask pixels={int(mask.sum())}"
                )
            plane = Plane(normal=normal, offset_m=offset, frame_id=frame_id)
            results.append(
                LegacyCameraPlane(
                    plane_id=plane_id, plane=plane, mask=mask.copy(),
                    pixel_count=pixels, fitting_rmse_m=rmse_m,
                )
            )
        unexpected = set(archive.files) - {f"plane_{i}" for i in seen}
        if unexpected:
            raise ValueError(f"Unexpected masks without CSV rows: {sorted(unexpected)}")

    masks_shapes = {plane.mask.shape for plane in results}
    if len(masks_shapes) != 1:
        raise ValueError("Camera plane masks have inconsistent resolutions")
    return tuple(sorted(results, key=lambda plane: plane.plane_id))


def load_handeye(
    path: str | Path, *, source_frame: str = "lidar", target_frame: str = "camera"
) -> LegacyHandEye:
    """Read D1 4x4 CSV/text or historical handeye_original.yaml.

    Output always stores a named T_target_source, never an unlabelled matrix.
    """
    path = Path(path)
    scale: float | None = None
    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
        if not isinstance(document, dict) or "T_C_L" not in document:
            raise ValueError("Hand-Eye YAML must contain a T_C_L mapping")
        extrinsic = document["T_C_L"]
        rotation = np.asarray(extrinsic["rotation_matrix"], dtype=np.float64)
        translation = np.asarray(extrinsic["translation_m"], dtype=np.float64)
        matrix = RigidTransform.from_rt(
            rotation, translation, target_frame=target_frame, source_frame=source_frame
        )
        if "sfm_scale_to_metric" in document:
            scale = float(document["sfm_scale_to_metric"])
    else:
        with path.open("r", encoding="utf-8") as handle:
            first = next((line for line in handle if line.strip() and not line.lstrip().startswith("#")), "")
        delimiter = "," if "," in first else None
        raw = np.loadtxt(path, delimiter=delimiter, dtype=np.float64)
        matrix = RigidTransform(
            matrix=raw, target_frame=target_frame, source_frame=source_frame
        )
    if scale is not None and (not np.isfinite(scale) or scale <= 0):
        raise ValueError("Hand-Eye SfM scale must be finite and positive")
    return LegacyHandEye(extrinsic=matrix, sfm_scale=scale)
