"""GLIM scan accumulation with explicit deskew or rigid-scan motion model.

KITTI Odometry .bin has no per-return timestamps, so use mode='rigid'.
Rigid mode transforms every return using its scan's associated GLIM pose;
it does NOT undo within-scan motion distortion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .trajectory import GlimTrajectory
from .scan_io import LidarScan


@dataclass(frozen=True)
class AccumulatedCloud:
    xyz: NDArray[np.float32]
    scan_id: NDArray[np.uint16]
    time_offset_s: NDArray[np.float32] | None
    ring: NDArray[np.uint16] | None
    source_return_index: NDArray[np.int64]
    reference_timestamp_ns: int
    frame_id: str = 'lidar_target'
    intensity: NDArray[np.float32] | None = None
    motion_model: Literal['deskew', 'rigid'] = 'deskew'

    def __post_init__(self):
        n = len(self.xyz)
        if self.xyz.shape != (n, 3) or any(
            np.asarray(a).shape != (n,) for a in (self.scan_id, self.source_return_index)
        ):
            raise ValueError('Accumulated point/provenance shape mismatch')
        for a in (self.time_offset_s, self.ring, self.intensity):
            if a is not None and np.asarray(a).shape != (n,):
                raise ValueError('Accumulated metadata shape mismatch')
        if self.motion_model not in ('deskew', 'rigid'):
            raise ValueError('Unknown motion model')
        if self.motion_model == 'deskew' and self.time_offset_s is None:
            raise ValueError('Deskewed cloud must retain per-return time offsets')
        if not np.isfinite(self.xyz).all() or (
            self.intensity is not None and not np.isfinite(self.intensity).all()
        ):
            raise ValueError('Non-finite accumulated cloud')
        if self.time_offset_s is not None and not np.isfinite(self.time_offset_s).all():
            raise ValueError('Non-finite accumulated time offsets')


def accumulate_glim(
    scans: Sequence[LidarScan],
    trajectory: GlimTrajectory,
    *,
    target_timestamp_ns: int,
    mode: Literal['deskew', 'rigid'] = 'deskew',
    frame_id: str = 'lidar_target',
) -> AccumulatedCloud:
    """Accumulate scans into the LiDAR coordinate system at target time.

    In deskew mode point positions use individual relative return timestamps.
    In rigid mode all points in one scan use the scan's *single* reference time.
    Point ordering and original per-scan source-return indices are preserved.
    Unknown timing/ring/intensity metadata remains None rather than invented.
    """
    if not scans or mode not in {'deskew', 'rigid'}:
        raise ValueError('Need nonempty scans and mode=deskew or rigid')
    if type(target_timestamp_ns) is not int or target_timestamp_ns < 0:
        raise ValueError('target_timestamp_ns must be nonnegative integer')
    if mode == 'deskew' and any(scan.time_offset_s is None for scan in scans):
        raise ValueError(
            'Cannot deskew LiDAR scans without per-return timestamps. '
            "Use mode='rigid' for KITTI Odometry .bin; this does not deskew motion distortion."
        )
    anchor_offset = target_timestamp_ns * 1e-9 - float(trajectory.timestamps[0])
    r_target, p_target = trajectory.at_offset(anchor_offset)
    target_inv = Rotation.from_matrix(r_target).inv()

    xyz_chunks, scan_chunks, dt_chunks, ring_chunks, return_chunks = [], [], [], [], []
    intensity_chunks = []
    all_timed = all(scan.time_offset_s is not None for scan in scans)
    all_ring = all(scan.ring is not None for scan in scans)
    all_intensity = all(scan.intensity is not None for scan in scans)
    for scan in scans:
        start_offset = (scan.timestamp_ns - target_timestamp_ns) * 1e-9 + anchor_offset
        if mode == 'deskew':
            offsets = start_offset + scan.time_offset_s
            rotation, position = trajectory.at_offset(offsets)
        else:
            # One pose interpolation per scan, not once per LiDAR return.
            rotation, position = trajectory.at_offset(start_offset)
        world = Rotation.from_matrix(rotation).apply(scan.xyz) + position
        local = target_inv.apply(world - p_target)
        xyz_chunks.append(local.astype(np.float32))
        scan_chunks.append(np.full(len(local), scan.scan_id, dtype=np.uint16))
        return_chunks.append(scan.source_return_index.astype(np.int64))
        if all_timed:
            dt_chunks.append(scan.time_offset_s.astype(np.float32))
        if all_ring:
            ring_chunks.append(scan.ring.astype(np.uint16))
        if all_intensity:
            intensity_chunks.append(scan.intensity.astype(np.float32))
    return AccumulatedCloud(
        np.concatenate(xyz_chunks), np.concatenate(scan_chunks),
        np.concatenate(dt_chunks) if all_timed else None,
        np.concatenate(ring_chunks) if all_ring else None,
        np.concatenate(return_chunks), target_timestamp_ns, frame_id,
        intensity=np.concatenate(intensity_chunks) if all_intensity else None,
        motion_model=mode,
    )
