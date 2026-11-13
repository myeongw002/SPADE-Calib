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
from typing import Sequence
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


def accumulate_kitti_rigid_window(
    lidar_dir: str | Path,
    glim_path: str | Path,
    times_path: str | Path,
    frame_ids: Sequence[int],
    *,
    target_frame_id: int,
    glim_first_frame: int = 0,
    min_range_m: float = 0.5,
    max_range_m: float = 100.0,
):
    """Accumulate indexed raw KITTI .bin scans using GLIM pose *rows*.

    Unlike timestamp interpolation, GLIM pose row k corresponds to KITTI frame
    glim_first_frame+k. This supports the historical synthetic 0.1-second
    GLIM timestamps; the times.txt entries are used only as real scan-time
    metadata. Every scan is treated as one rigid snapshot: there is NO deskew,
    no return-level timestamps or ring ID reconstruction.

    Output's scan_id is LOCAL (0..len(frame_ids)-1), matching frame_ids order.
    Its xyz lives in the LiDAR sensor frame at target_frame_id.
    """
    from spade_calib.io.kitti_bin import load_kitti_bin_scan
    from spade_calib.lidar.deskew import AccumulatedCloud

    if type(glim_first_frame) is not int or glim_first_frame < 0:
        raise ValueError('glim_first_frame must be a nonnegative integer')
    if type(target_frame_id) is not int or target_frame_id < 0:
        raise ValueError('target_frame_id must be a nonnegative integer')
    if (not frame_ids or len(frame_ids) > 65536 or
        any(type(f) is not int or f < 0 for f in frame_ids) or
        len(set(frame_ids)) != len(frame_ids)):
        raise ValueError('frame_ids must be nonempty, unique nonnegative integers <= 65536')
    if list(frame_ids) != sorted(frame_ids):
        raise ValueError('frame_ids must be in increasing order')
    times = read_kitti_times_ns(times_path)
    glim = GlimTrajectory.from_array(np.loadtxt(glim_path, dtype=np.float64, ndmin=2))
    last_glim_frame = glim_first_frame + len(glim.timestamps)
    if last_glim_frame > len(times):
        raise ValueError('GLIM pose rows exceed KITTI times.txt frame range')
    requested = list(frame_ids) + [target_frame_id]
    if any(f < glim_first_frame or f >= last_glim_frame for f in requested):
        raise ValueError('Selected KITTI frame is outside available GLIM pose rows')

    scan_dir = Path(lidar_dir)
    if not scan_dir.is_dir():
        raise ValueError('KITTI LiDAR scan directory not found')
    scan_paths: dict[int, Path] = {}
    for path in scan_dir.iterdir():
        if path.is_file() and path.suffix.lower() == '.bin':
            f = frame_id_from_name(path.name)
            if f in scan_paths:
                raise ValueError(f'Duplicate KITTI LiDAR frame {f}')
            scan_paths[f] = path
    if any(f not in scan_paths for f in frame_ids):
        missing = [f for f in frame_ids if f not in scan_paths]
        raise ValueError(f'Missing KITTI LiDAR scans for frame IDs: {missing[:10]}')

    reference = target_frame_id - glim_first_frame
    R_target = Rotation.from_quat(glim.quaternions_xyzw[reference]).as_matrix()
    p_target = glim.positions[reference]
    xyz_chunks, scan_chunks, index_chunks, intensity_chunks = [], [], [], []
    for local_id, frame in enumerate(frame_ids):
        src = load_kitti_bin_scan(
            scan_paths[frame],
            scan_id=local_id,
            timestamp_ns=int(times[frame]),
            min_range_m=min_range_m,
            max_range_m=max_range_m,
        )
        pose_idx = frame - glim_first_frame
        R_source = Rotation.from_quat(glim.quaternions_xyzw[pose_idx]).as_matrix()
        p_source = glim.positions[pose_idx]
        R_target_source = R_target.T @ R_source
        t_target_source = R_target.T @ (p_source - p_target)
        xyz = src.xyz @ R_target_source.T + t_target_source
        xyz_chunks.append(xyz.astype(np.float32))
        scan_chunks.append(np.full(len(xyz), local_id, dtype=np.uint16))
        index_chunks.append(src.source_return_index)
        intensity_chunks.append(src.intensity.astype(np.float32))

    return AccumulatedCloud(
        xyz=np.concatenate(xyz_chunks),
        scan_id=np.concatenate(scan_chunks),
        time_offset_s=None,
        ring=None,
        source_return_index=np.concatenate(index_chunks),
        reference_timestamp_ns=int(times[target_frame_id]),
        frame_id=f'kitti_lidar_frame_{target_frame_id}',
        intensity=np.concatenate(intensity_chunks),
        motion_model='rigid',
    )
