"""KITTI Odometry Velodyne .bin reader (XYZ + reflectance, no point timestamps).

The raw .bin format is little-endian float32 rows of x, y, z, reflectance.
It contains *no* per-return timestamps or ring IDs. Those remain None.
A scan timestamp must be supplied externally (times.txt or metadata).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from spade_calib.lidar.scan_io import LidarScan

_KITTI_COLUMNS = 4
_MAX_POINTS = 20_000_000


def load_kitti_bin_scan(
    path: str | Path,
    *,
    scan_id: int,
    timestamp_ns: int,
    min_range_m: float = 0.5,
    max_range_m: float = 100.0,
) -> LidarScan:
    """Load and range-filter a KITTI 4xfloat32 scan, preserving source indices.

    No point timing, ring, or deskew assumption is synthesized.
    Reflectance is carried as intensity with its native values.
    """
    if type(scan_id) is not int or not 0 <= scan_id <= 65535:
        raise ValueError("scan_id must be in [0,65535]")
    if type(timestamp_ns) is not int or timestamp_ns < 0:
        raise ValueError("timestamp_ns must be a nonnegative integer")
    if not (np.isfinite(min_range_m) and np.isfinite(max_range_m)
            and 0 <= min_range_m < max_range_m):
        raise ValueError("Invalid LiDAR range bounds")
    source = Path(path)
    if source.suffix.lower() != ".bin":
        raise ValueError("KITTI LiDAR reader only accepts .bin")
    size = source.stat().st_size
    if size == 0 or size % (_KITTI_COLUMNS * 4) != 0:
        raise ValueError("KITTI .bin size must be a nonempty multiple of 16 bytes")
    n = size // (_KITTI_COLUMNS * 4)
    if n > _MAX_POINTS:
        raise ValueError("KITTI .bin exceeds point count limit")
    raw = np.fromfile(source, dtype="<f4")
    if raw.size != n * _KITTI_COLUMNS:
        raise ValueError("KITTI .bin changed size while reading")
    points = raw.reshape(n, _KITTI_COLUMNS)
    xyz = points[:, :3]
    intensity = points[:, 3]
    distance_squared = np.einsum("ij,ij->i", xyz.astype(np.float64), xyz.astype(np.float64))
    mask = np.isfinite(points).all(axis=1)
    mask &= distance_squared > min_range_m ** 2
    mask &= distance_squared < max_range_m ** 2
    chosen = np.flatnonzero(mask)
    return LidarScan(
        scan_id=scan_id, timestamp_ns=timestamp_ns,
        xyz=xyz[chosen].astype(np.float64, copy=True),
        time_offset_s=None, ring=None,
        source_return_index=chosen.astype(np.int64),
        intensity=intensity[chosen].astype(np.float64, copy=True),
    )
