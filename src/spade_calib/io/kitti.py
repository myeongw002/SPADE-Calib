"""KITTI Odometry frame-index adapter for the shared trajectory representation.

KITTI 'times.txt' contains seconds relative to sequence start. Its rows and
image/scan filenames are indexed by frame ID. Do not assume the numeric times
stored in an independently exported GLIM TUM file have the same clock: legacy
exports may contain synthetic 0.1-second increments. This adapter explicitly
joins by frame index and uses times.txt only to time the matched motions.

This is deliberately a *trajectory* adapter, not an alternative Hand-Eye solver.
The full recording indexer can consume derived timestamp CSV metadata separately.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
import re

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from spade_calib.io.colmap import load_registered_images
from spade_calib.lidar.trajectory import GlimTrajectory
from spade_calib.selection.trajectory import MatchedTrajectories


_INT64_MAX = (1 << 63) - 1


def read_kitti_times_ns(path: str | Path) -> NDArray[np.int64]:
    """Read KITTI seconds exactly as decimal text, converting to nanoseconds."""
    src = Path(path)
    stamps: list[int] = []
    with src.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            token = line.strip()
            if not token:
                raise ValueError(f"Empty KITTI time at line {line_number}")
            try:
                seconds = Decimal(token)
            except InvalidOperation as exc:
                raise ValueError(f"Invalid KITTI time at line {line_number}: {token}") from exc
            if not seconds.is_finite() or seconds < 0:
                raise ValueError(f"KITTI time must be nonnegative and finite, line {line_number}")
            stamp = int((seconds * Decimal(1_000_000_000)).to_integral_value(
                rounding=ROUND_HALF_EVEN))
            if stamp > _INT64_MAX or (stamps and stamp <= stamps[-1]):
                raise ValueError(f"KITTI timestamps overflow or are not increasing at line {line_number}")
            stamps.append(stamp)
    if len(stamps) < 2:
        raise ValueError("KITTI times.txt must contain at least two timestamps")
    return np.asarray(stamps, dtype=np.int64)


def frame_id_from_name(name: str) -> int:
    """Read numeric basename without inferring its timestamp from the filename."""
    stem = Path(name).stem
    if not re.fullmatch(r"[0-9]+", stem):
        raise ValueError(f"KITTI image/scan filename must be a numeric frame ID: {name}")
    return int(stem)


def kitti_timestamp_map(times_path: str | Path, names: list[str] | tuple[str, ...]
                        ) -> dict[str, int]:
    """Map KITTI index-named files to their timestamp in sequence-relative ns."""
    times = read_kitti_times_ns(times_path)
    result: dict[str, int] = {}
    seen_frames: set[int] = set()
    for name in names:
        frame = frame_id_from_name(name)
        if frame >= len(times):
            raise ValueError(f"Frame {frame} outside KITTI times.txt")
        if name in result or frame in seen_frames:
            raise ValueError(f"Duplicate KITTI frame ID: {frame}")
        result[name] = int(times[frame])
        seen_frames.add(frame)
    return result


def _read_tum_camera_poses(path: Path) -> dict[int, tuple[str, NDArray, NDArray]]:
    """Legacy full-frame TUM export. Row n corresponds to KITTI frame n.

    A TUM pose export lacking frame IDs CANNOT represent dropped/unregistered
    frame identity safely; use COLMAP images.bin/camera_poses.csv instead.
    """
    raw = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if raw.ndim != 2 or raw.shape[1] != 8 or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError("Legacy KITTI Camera TUM must have finite N x 8 poses")
    if np.any(np.diff(raw[:, 0]) <= 0):
        raise ValueError("Legacy Camera TUM timestamps must increase")
    if np.any(np.linalg.norm(raw[:, 4:8], axis=1) < 1e-12):
        raise ValueError("Camera TUM has invalid quaternions")
    rotations = Rotation.from_quat(raw[:, 4:8]).as_matrix()
    return {i: (f"{i:06d}.png", rotations[i], raw[i, 1:4])
            for i in range(len(raw))}


def load_kitti_trajectory_data(
    glim_path: str | Path,
    sfm_path: str | Path,
    times_path: str | Path,
    *,
    glim_first_frame: int = 0,
) -> MatchedTrajectories:
    """Join GLIM and registered camera poses by KITTI frame ID.

    GLIM row 0 corresponds to `glim_first_frame`; the rows must be contiguous.
    COLMAP filenames provide explicit frame IDs. For legacy full-frame TUM
    Camera exports only, row n is assumed to be frame n, with no dropped poses.
    Unlike timestamp interpolation, this handles old synthetic-timestamp GLIM.
    """
    if type(glim_first_frame) is not int or glim_first_frame < 0:
        raise ValueError("glim_first_frame must be a nonnegative integer")
    times = read_kitti_times_ns(times_path)
    glim = GlimTrajectory.from_array(np.loadtxt(glim_path, dtype=np.float64, ndmin=2))
    if glim_first_frame + len(glim.timestamps) > len(times):
        raise ValueError("GLIM frame range exceeds KITTI times.txt")
    cam_path = Path(sfm_path)
    if cam_path.name == "camera_trajectory_tum.txt":
        camera = _read_tum_camera_poses(cam_path)
    else:
        records = load_registered_images(cam_path)
        camera = {}
        for pose in records.values():
            frame = frame_id_from_name(pose.name)
            if frame in camera:
                raise ValueError(f"Duplicate COLMAP KITTI frame {frame}")
            if frame >= len(times):
                raise ValueError(f"Camera frame {frame} outside KITTI times.txt")
            q = np.asarray(pose.qvec_wxyz, dtype=np.float64)
            Rcw = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            Rwc = Rcw.T
            twc = -Rwc @ np.asarray(pose.tvec, dtype=np.float64)
            camera[frame] = (pose.name, Rwc, twc)
    frame_ids = sorted(
        i for i in camera if glim_first_frame <= i < glim_first_frame + len(glim.timestamps))
    if len(frame_ids) < 5:
        raise ValueError("Fewer than five KITTI frame-index matched Camera/GLIM poses")
    indices = np.asarray([i - glim_first_frame for i in frame_ids], dtype=np.int64)
    stamps = times[np.asarray(frame_ids, dtype=np.int64)]
    lidar_R = Rotation.from_quat(glim.quaternions_xyzw[indices]).as_matrix()
    return MatchedTrajectories(
        timestamps_ns=stamps,
        image_names=tuple(camera[i][0] for i in frame_ids),
        lidar_R_world=lidar_R,
        lidar_t_world_m=glim.positions[indices].copy(),
        camera_R_world_sfm=np.stack([camera[i][1] for i in frame_ids]),
        camera_t_world_sfm=np.stack([camera[i][2] for i in frame_ids]),
        total_registered=len(camera),
        discarded_outside_glim=len(camera) - len(frame_ids),
    )
