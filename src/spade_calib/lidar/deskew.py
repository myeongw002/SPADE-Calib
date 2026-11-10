"""Per-return GLIM deskew to a target LiDAR frame (including provenance)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Sequence
import numpy as np
from numpy.typing import NDArray
from .trajectory import GlimTrajectory
from .scan_io import LidarScan


@dataclass(frozen=True)
class AccumulatedCloud:
    xyz: NDArray[np.float32]
    scan_id: NDArray[np.uint16]
    time_offset_s: NDArray[np.float32]
    ring: NDArray[np.uint16]
    source_return_index: NDArray[np.int64]
    reference_timestamp_ns: int
    frame_id: str = 'lidar_target'

    def __post_init__(self):
        n = len(self.xyz)
        if self.xyz.shape != (n, 3) or any(len(a) != n for a in
            (self.scan_id, self.time_offset_s, self.ring, self.source_return_index)):
            raise ValueError('Accumulated point/provenance shape mismatch')


def accumulate_glim(scans: Sequence[LidarScan], trajectory: GlimTrajectory,
                    *, target_timestamp_ns: int, mode: Literal['deskew', 'rigid'] = 'deskew',
                    frame_id: str = 'lidar_target') -> AccumulatedCloud:
    """T_target_source(t) = inv(T_world_target) @ T_world_lidar(t).

    Time offsets are scan-relative seconds. Source points are preserved in
    original manifest order. Output coordinates cast to float32 after transform
    to match the recovered D1 experiment's data representation.
    """
    if not scans or mode not in {'deskew', 'rigid'}:
        raise ValueError('Need nonempty scans and mode=deskew or rigid')
    anchor_offset = target_timestamp_ns * 1e-9 - float(trajectory.timestamps[0])
    r_target, p_target = trajectory.at_offset(anchor_offset)
    xyz_chunks, scan_chunks, dt_chunks, ring_chunks, return_chunks = [], [], [], [], []
    for scan in scans:
        start_offset = (scan.timestamp_ns - target_timestamp_ns) * 1e-9 + anchor_offset
        offsets = start_offset + scan.time_offset_s if mode == 'deskew' else np.full(len(scan.xyz), start_offset)
        # scipy Slerp is SO(3) interpolation; preserve legacy R.apply arithmetic.
        from scipy.spatial.transform import Rotation
        rotation, position = trajectory.at_offset(offsets)
        world = Rotation.from_matrix(rotation).apply(scan.xyz) + position
        local = Rotation.from_matrix(r_target).inv().apply(world - p_target)
        xyz_chunks.append(local.astype(np.float32))
        scan_chunks.append(np.full(len(local), scan.scan_id, dtype=np.uint16))
        dt_chunks.append(scan.time_offset_s.astype(np.float32))
        ring_chunks.append(scan.ring.astype(np.uint16))
        return_chunks.append(scan.source_return_index.astype(np.int64))
    return AccumulatedCloud(
        np.concatenate(xyz_chunks), np.concatenate(scan_chunks),
        np.concatenate(dt_chunks), np.concatenate(ring_chunks),
        np.concatenate(return_chunks), target_timestamp_ns, frame_id,
    )
