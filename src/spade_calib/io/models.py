"""Minimal validated models shared by future SPADE-Calib pipeline stages.

Coordinate conventions are defined in docs/coordinate_systems.md.
These classes validate geometry; they do not perform calibration or file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}; got {array.shape}")
    return array.copy()


def _frame_id(frame: str, name: str) -> str:
    if not isinstance(frame, str) or not frame.strip():
        raise ValueError(f"{name} must be a nonempty frame identifier")
    return frame


@dataclass(frozen=True)
class CameraModel:
    """Pinhole/standard OpenCV distortion model.

    K maps normalized camera coordinates to pixel coordinates. The distortion
    coefficient order follows OpenCV for the selected camera model; callers
    must explicitly keep that model/ordering consistent.
    """

    width: int
    height: int
    K: NDArray[np.float64]
    distortion: NDArray[np.float64]
    frame_id: str = "camera"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width and height must be positive")
        k = _finite_array(self.K, (3, 3), "K")
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("K must have positive focal lengths and bottom row [0,0,1]")
        distortion = np.asarray(self.distortion, dtype=np.float64)
        if distortion.ndim != 1 or not np.isfinite(distortion).all():
            raise ValueError("distortion must be a finite 1D array")
        object.__setattr__(self, "K", k)
        object.__setattr__(self, "distortion", distortion.copy())
        _frame_id(self.frame_id, "camera frame_id")


@dataclass(frozen=True)
class RigidTransform:
    """Rigid transform T_target_source: p_target = T_target_source * p_source."""

    matrix: NDArray[np.float64]
    target_frame: str
    source_frame: str

    def __post_init__(self) -> None:
        matrix = _finite_array(self.matrix, (4, 4), "transform matrix")
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9):
            raise ValueError("Transform bottom row must equal [0,0,0,1]")
        r = matrix[:3, :3]
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(
            np.linalg.det(r), 1.0, atol=1e-6
        ):
            raise ValueError("Transform rotation must be in SO(3)")
        _frame_id(self.target_frame, "target_frame")
        _frame_id(self.source_frame, "source_frame")
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def from_rt(
        cls,
        R: NDArray[np.float64],
        t: NDArray[np.float64],
        target_frame: str,
        source_frame: str,
    ) -> "RigidTransform":
        r = _finite_array(R, (3, 3), "R")
        translation = _finite_array(t, (3,), "t")
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = r
        transform[:3, 3] = translation
        return cls(transform, target_frame=target_frame, source_frame=source_frame)

    def inverse(self) -> "RigidTransform":
        r = self.matrix[:3, :3]
        t = self.matrix[:3, 3]
        inverse = np.eye(4, dtype=np.float64)
        inverse[:3, :3] = r.T
        inverse[:3, 3] = -r.T @ t
        return RigidTransform(
            inverse, target_frame=self.source_frame, source_frame=self.target_frame
        )

    def apply(self, xyz: NDArray[np.float64]) -> NDArray[np.float64]:
        points = np.asarray(xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("xyz must have shape (N,3) with finite coordinates")
        return points @ self.matrix[:3, :3].T + self.matrix[:3, 3]


@dataclass(frozen=True)
class PointCloud:
    """Point cloud in a named Cartesian frame; source IDs are optional provenance."""

    xyz: NDArray[np.float64]
    frame_id: str
    source_point_id: Optional[NDArray[np.int64]] = None
    scan_id: Optional[NDArray[np.int64]] = None
    timestamp_ns: Optional[NDArray[np.int64]] = None

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
            raise ValueError("PointCloud.xyz must have shape (N,3) with finite coordinates")
        _frame_id(self.frame_id, "point cloud frame_id")
        object.__setattr__(self, "xyz", xyz.copy())
        for name in ("source_point_id", "scan_id", "timestamp_ns"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value)
            if array.shape != (len(xyz),) or array.dtype.kind not in "iu":
                raise ValueError(f"{name} must be a length-N integer array")
            object.__setattr__(self, name, array.astype(np.int64, copy=True))

    @property
    def point_count(self) -> int:
        return len(self.xyz)


@dataclass(frozen=True)
class Plane:
    """Plane in a named metric frame: normal.dot(x) + offset_m = 0."""

    normal: NDArray[np.float64]
    offset_m: float
    frame_id: str
    centroid: Optional[NDArray[np.float64]] = None

    def __post_init__(self) -> None:
        normal = _finite_array(self.normal, (3,), "plane normal")
        norm = float(np.linalg.norm(normal))
        if norm < 1e-12:
            raise ValueError("Plane normal must be nonzero")
        if not np.isfinite(self.offset_m):
            raise ValueError("Plane offset must be finite")
        _frame_id(self.frame_id, "plane frame_id")
        object.__setattr__(self, "normal", normal / norm)
        object.__setattr__(self, "offset_m", float(self.offset_m) / norm)
        if self.centroid is not None:
            object.__setattr__(self, "centroid", _finite_array(self.centroid, (3,), "centroid"))
