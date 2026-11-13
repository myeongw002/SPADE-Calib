"""Optional conservative target-scan free-space filter; NOT full ghost removal."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .deskew import AccumulatedCloud


@dataclass(frozen=True)
class FreeSpaceSettings:
    azimuth_resolution_deg: float = 0.2
    elevation_resolution_deg: float = 2.0
    margin_m: float = 0.5

    def __post_init__(self):
        if not 0 < self.azimuth_resolution_deg <= 360 or not 0 < self.elevation_resolution_deg <= 180:
            raise ValueError('Angular resolution must be positive')
        if not np.isfinite(self.margin_m) or self.margin_m < 0:
            raise ValueError('margin_m must be finite and >= 0')


def _range_bins(points: NDArray, params: FreeSpaceSettings) -> tuple[NDArray, NDArray, int]:
    x, y, z = np.asarray(points, dtype=np.float64).T
    ranges = np.sqrt(x*x + y*y + z*z)
    az = np.degrees(np.arctan2(y, x))
    el = np.degrees(np.arctan2(z, np.hypot(x, y)))
    n_az = int(np.ceil(360 / params.azimuth_resolution_deg))
    n_el = int(np.ceil(180 / params.elevation_resolution_deg))
    ai = np.clip(np.floor((az + 180)/params.azimuth_resolution_deg).astype(np.int64), 0, n_az-1)
    ei = np.clip(np.floor((el + 90)/params.elevation_resolution_deg).astype(np.int64), 0, n_el-1)
    return ei * n_az + ai, ranges, n_az*n_el


def target_free_space_mask(cloud: AccumulatedCloud, *, target_scan_id: int,
                           settings: FreeSpaceSettings = FreeSpaceSettings()) -> NDArray[np.bool_]:
    """Keep target points, unobserved bins, and points behind target surfaces.

    A historical point is removed only when it lies clearly IN FRONT of a
    target return in the same angular bin. The input cloud must be expressed
    in the target scan's LiDAR frame. This method is conservative but cannot
    identify ghosts in occluded or unobserved directions.
    """
    if not np.isfinite(cloud.xyz).all():
        raise ValueError('Point coordinates must be finite')
    target = cloud.scan_id == target_scan_id
    if not np.any(target):
        raise ValueError('Target scan has no returns')
    indices, ranges, bins = _range_bins(cloud.xyz, settings)
    known = np.full(bins, np.inf)
    np.minimum.at(known, indices[target], ranges[target])
    observed = np.isfinite(known[indices])
    conflicting = (~target) & observed & (ranges < known[indices] - settings.margin_m)
    return ~conflicting


def filter_cloud(cloud: AccumulatedCloud, keep: NDArray[np.bool_]) -> AccumulatedCloud:
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (len(cloud.xyz),):
        raise ValueError('Filtering mask has wrong shape')
    return AccumulatedCloud(
        cloud.xyz[keep], cloud.scan_id[keep],
        cloud.time_offset_s[keep] if cloud.time_offset_s is not None else None,
        cloud.ring[keep] if cloud.ring is not None else None,
        cloud.source_return_index[keep],
        cloud.reference_timestamp_ns, cloud.frame_id,
        intensity=cloud.intensity[keep] if cloud.intensity is not None else None,
        motion_model=cloud.motion_model,
    )
