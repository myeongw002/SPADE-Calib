"""Geometry-only LiDAR ground extraction with physically plausible up/height gates.

Adapted from archived ROI-free D1 repeated RANSAC + PCA refit. This module
does not use KITTI GT extrinsics or the Camera depth to fit a LiDAR plane.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spade_calib.calibration.ground import Plane
from spade_calib.camera.planes import _plane_fit


@dataclass(frozen=True)
class LidarGroundSettings:
    voxel_m: float = .08
    max_input_points: int = 300000
    max_sampled_points: int = 60000
    min_range_m: float = .8
    max_range_m: float = 55.0
    min_height_m: float = .35
    max_height_m: float = 3.5
    normal_max_tilt_deg: float = 28.
    inlier_threshold_m: float = .10
    refit_threshold_m: float = .14
    iterations: int = 420
    min_inliers: int = 180
    random_seed: int = 20261107


@dataclass(frozen=True)
class LidarGroundCandidate:
    plane: Plane
    xyz: NDArray[np.float64]
    source_indices: NDArray[np.int64]
    inlier_mask: NDArray[np.bool_]
    rmse_m: float
    lidar_up: NDArray[np.float64]

    @property
    def inlier_count(self) -> int:
        return int(self.inlier_mask.sum())


def extract_lidar_ground(points: NDArray,
                         settings: LidarGroundSettings = LidarGroundSettings(),
                         *, lidar_up: NDArray | None = None
                         ) -> tuple[LidarGroundCandidate, dict]:
    """RANSAC + plane PCA for the LiDAR ground in the target LiDAR frame.

    The positive-up normal is an explicit user assumption; a candidate
    must lie *below* the LiDAR origin at a plausible signed height.
    """
    cfg = settings
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('LiDAR cloud must be finite Nx3')
    if (cfg.voxel_m <= 0 or cfg.inlier_threshold_m <= 0 or cfg.refit_threshold_m <= 0
            or cfg.min_height_m <= 0 or cfg.max_height_m <= cfg.min_height_m
            or cfg.min_range_m < 0 or cfg.max_range_m <= cfg.min_range_m
            or not 0 < cfg.normal_max_tilt_deg < 80):
        raise ValueError('Invalid LiDAR ground geometry settings')
    if cfg.iterations < 1 or cfg.min_inliers < 3:
        raise ValueError('Invalid LiDAR RANSAC settings')
    up = np.array([0., 0., 1.]) if lidar_up is None else np.asarray(lidar_up, dtype=float)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-10:
        raise ValueError('Invalid lidar_up direction')
    up /= np.linalg.norm(up)
    radius = np.linalg.norm(p, axis=1)
    chosen = np.flatnonzero((radius >= cfg.min_range_m) & (radius <= cfg.max_range_m))
    if len(chosen) < cfg.min_inliers:
        raise ValueError('Insufficient LiDAR points after range filter')
    rng = np.random.default_rng(cfg.random_seed)
    if len(chosen) > cfg.max_input_points:
        chosen = np.sort(rng.choice(chosen, cfg.max_input_points, replace=False))
    cells = np.floor(p[chosen] / cfg.voxel_m).astype(np.int64)
    _, ix = np.unique(cells, axis=0, return_index=True)
    chosen = chosen[np.sort(ix)]
    if len(chosen) > cfg.max_sampled_points:
        chosen = np.sort(rng.choice(chosen, cfg.max_sampled_points, replace=False))
    xyz = p[chosen]
    if len(xyz) < cfg.min_inliers:
        raise ValueError('Insufficient LiDAR points after voxel filtering')

    cosine = float(np.cos(np.deg2rad(cfg.normal_max_tilt_deg)))
    best_count = 0
    best_inliers = None
    for _ in range(cfg.iterations):
        tri = xyz[rng.choice(len(xyz), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = np.linalg.norm(n)
        if length < 1e-8:
            continue
        n /= length
        if n @ up < 0:
            n = -n
        if n @ up < cosine:
            continue
        d = -float(n @ tri[0])
        # For positive-up normal, below-sensor ground has positive d.
        if not cfg.min_height_m <= d <= cfg.max_height_m:
            continue
        mask = np.abs(xyz @ n + d) < cfg.inlier_threshold_m
        count = int(mask.sum())
        if count > best_count:
            best_inliers, best_count = mask, count
    if best_inliers is None or best_count < cfg.min_inliers:
        raise ValueError('No physically plausible LiDAR ground plane found')
    plane, _ = _plane_fit(xyz[best_inliers])
    if plane.normal @ up < 0:
        plane = Plane(-plane.normal, -plane.offset_m)
    inliers = np.abs(xyz @ plane.normal + plane.offset_m) < cfg.refit_threshold_m
    if inliers.sum() < cfg.min_inliers:
        raise ValueError('Ground plane refinement left insufficient inliers')
    plane, rmse = _plane_fit(xyz[inliers])
    if plane.normal @ up < 0:
        plane = Plane(-plane.normal, -plane.offset_m)
    tilt = float(np.rad2deg(np.arccos(np.clip(plane.normal @ up, -1, 1))))
    if tilt > cfg.normal_max_tilt_deg or not (cfg.min_height_m <= plane.offset_m <= cfg.max_height_m):
        raise ValueError('Refitted LiDAR ground normal or height fails physical gate')
    inliers = np.abs(xyz @ plane.normal + plane.offset_m) < cfg.refit_threshold_m
    if inliers.sum() < cfg.min_inliers:
        raise ValueError('Insufficient LiDAR ground support after final refit')
    return LidarGroundCandidate(plane, xyz, chosen.astype(np.int64),
                                inliers, rmse, up), {
        'method': 'voxel_RANSAC_PCA_refit_ground_with_positive_up_height_prior',
        'input_points': len(p),
        'sampled_points': len(xyz),
        'ransac_inliers': best_count,
        'final_inliers': int(inliers.sum()),
        'rmse_m': rmse,
        'tilt_from_lidar_up_deg': tilt,
        'height_offset_m': plane.offset_m,
        'not_a_ground_truth_label': True,
    }
