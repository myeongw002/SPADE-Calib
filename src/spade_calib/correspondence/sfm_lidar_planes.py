"""Finite LiDAR plane patches and provisional SfM-point-to-patch associations.

All LiDAR coordinates refer to the accumulated cloud's *target sensor frame*.
SfM points are in the simultaneous camera axes, in unscaled SfM units.
This module never uses GT extrinsics, DA2, or KITTI ground-truth poses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class PatchSettings:
    voxel_size_m: float = 0.18
    min_range_m: float = 1.0
    max_range_m: float = 65.0
    max_sampled_points: int = 30000
    max_planes: int = 8
    ransac_iterations: int = 180
    distance_threshold_m: float = 0.14
    min_inliers: int = 80
    min_width_ratio: float = 0.08
    connectivity_radius_m: float = 1.0
    random_seed: int = 20261118


@dataclass(frozen=True)
class MatchSettings:
    max_normal_distance_m: float = 1.5
    max_support_distance_m: float = 2.5
    min_score_margin: float = 0.12
    min_matches_per_patch: int = 8
    max_matches_per_patch: int = 200


@dataclass(frozen=True)
class LidarPatch:
    patch_id: int
    normal: NDArray[np.float64]
    offset_m: float
    support_xyz_lidar: NDArray[np.float64]
    rmse_m: float
    scan_ids: NDArray[np.int64]


@dataclass(frozen=True)
class SfMPlaneMatches:
    camera_xyz_sfm: NDArray[np.float64]
    feature_xy: NDArray[np.float64]
    normal_lidar: NDArray[np.float64]
    offset_lidar_m: NDArray[np.float64]
    patch_id: NDArray[np.int32]
    feature_index: NDArray[np.int64]

    @property
    def count(self) -> int:
        return len(self.camera_xyz_sfm)

    def validate(self) -> None:
        n = self.count
        expected = ((self.camera_xyz_sfm, (n, 3)), (self.feature_xy, (n, 2)),
                    (self.normal_lidar, (n, 3)), (self.offset_lidar_m, (n,)),
                    (self.patch_id, (n,)), (self.feature_index, (n,)))
        if n == 0:
            raise ValueError('No point-plane matches')
        for arr, shape in expected:
            if np.asarray(arr).shape != shape or not np.isfinite(arr).all():
                raise ValueError(f'Invalid point-plane array: expected finite {shape}')
        if np.any(np.linalg.norm(self.normal_lidar, axis=1) < 0.999) or np.any(
                np.linalg.norm(self.normal_lidar, axis=1) > 1.001):
            raise ValueError('LiDAR plane normals must be unit length')
        if np.unique(self.feature_index).size != n:
            raise ValueError('One SfM 3D feature must match at most one patch')


def _fit_plane(x: NDArray) -> tuple[NDArray, float, float, float]:
    center = x.mean(axis=0)
    eigenvalues, vectors = np.linalg.eigh((x-center).T @ (x-center)/len(x))
    normal = vectors[:, 0]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    width_ratio = np.sqrt(max(eigenvalues[1], 0.) / max(eigenvalues[2], 1e-15))
    return normal, -float(normal @ center), float(np.sqrt(max(eigenvalues[0], 0.))), float(width_ratio)


def _largest_connected(xyz: NDArray, radius: float) -> NDArray[np.int64]:
    if len(xyz) < 3:
        return np.arange(len(xyz), dtype=np.int64)
    tree = cKDTree(xyz)
    ds, ids = tree.query(xyz, k=min(13, len(xyz)), workers=1)
    rows = np.broadcast_to(np.arange(len(xyz))[:, None], ids[:, 1:].shape)
    good = np.isfinite(ds[:, 1:]) & (ds[:, 1:] <= radius)
    if not good.any():
        return np.array([], dtype=np.int64)
    graph = coo_matrix((np.ones(good.sum(), dtype=np.uint8),
                        (rows[good], ids[:, 1:][good])), shape=(len(xyz), len(xyz)))
    _, labels = connected_components(graph, directed=False)
    return np.flatnonzero(labels == np.argmax(np.bincount(labels)))


def extract_lidar_patches(xyz_lidar: NDArray, scan_id: NDArray | None = None,
                          settings: PatchSettings = PatchSettings()
                          ) -> tuple[list[LidarPatch], dict]:
    """Repeated RANSAC, finite connected support and refitting; no semantic labels.

    This is a baseline extractor, not a calibrated noise/visibility model.
    """
    cfg = settings
    if (cfg.voxel_size_m <= 0 or cfg.distance_threshold_m <= 0 or
            cfg.connectivity_radius_m <= 0 or cfg.min_inliers < 3 or
            cfg.max_planes < 1 or cfg.ransac_iterations < 1 or
            cfg.max_sampled_points < cfg.min_inliers or
            not 0 < cfg.min_width_ratio < 1 or
            cfg.max_range_m <= cfg.min_range_m or cfg.min_range_m < 0):
        raise ValueError('Invalid patch extraction settings')
    xyz = np.asarray(xyz_lidar, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('LiDAR points must be finite Nx3')
    sid = np.zeros(len(xyz), dtype=np.int64) if scan_id is None else np.asarray(scan_id)
    if sid.shape != (len(xyz),) or not np.isfinite(sid).all():
        raise ValueError('LiDAR scan_id shape mismatch')
    lengths = np.linalg.norm(xyz, axis=1)
    selected = np.flatnonzero((lengths >= cfg.min_range_m) & (lengths <= cfg.max_range_m))
    if len(selected) < cfg.min_inliers:
        return [], {'reason': 'too_few_points', 'sampled_points': len(selected)}
    cells = np.floor(xyz[selected]/cfg.voxel_size_m).astype(np.int64)
    _, unique = np.unique(cells, axis=0, return_index=True)
    selected = selected[np.sort(unique)]
    rng = np.random.default_rng(cfg.random_seed)
    if len(selected) > cfg.max_sampled_points:
        selected = np.sort(rng.choice(selected, cfg.max_sampled_points, replace=False))
    remaining = selected.copy()
    patches: list[LidarPatch] = []
    for _ in range(cfg.max_planes * 5):
        if len(patches) >= cfg.max_planes or len(remaining) < cfg.min_inliers:
            break
        p = xyz[remaining]
        best = None
        best_count = 0
        for _ in range(cfg.ransac_iterations):
            triplet = p[rng.choice(len(p), 3, replace=False)]
            n = np.cross(triplet[1] - triplet[0], triplet[2] - triplet[0])
            mag = np.linalg.norm(n)
            if mag < 1e-9:
                continue
            n /= mag
            d = -float(n @ triplet[0])
            mask = np.abs(p @ n + d) <= cfg.distance_threshold_m
            count = int(mask.sum())
            if count > best_count:
                best, best_count = mask, count
        if best is None or best_count < cfg.min_inliers:
            break
        tentative = remaining[best]
        normal, offset, _, _ = _fit_plane(xyz[tentative])
        planar = remaining[np.abs(xyz[remaining] @ normal + offset) <=
                           cfg.distance_threshold_m]
        connected = _largest_connected(xyz[planar], cfg.connectivity_radius_m)
        if len(connected) < cfg.min_inliers:
            remaining = remaining[~best]
            continue
        chosen = planar[connected]
        normal, offset, _, _ = _fit_plane(xyz[chosen])
        chosen = chosen[np.abs(xyz[chosen] @ normal + offset) <=
                        cfg.distance_threshold_m]
        if len(chosen) < cfg.min_inliers:
            remaining = remaining[~best]
            continue
        normal, offset, rmse, width_ratio = _fit_plane(xyz[chosen])
        if width_ratio >= cfg.min_width_ratio and rmse <= cfg.distance_threshold_m:
            patches.append(LidarPatch(
                patch_id=len(patches), normal=normal, offset_m=offset,
                support_xyz_lidar=xyz[chosen].copy(), rmse_m=rmse,
                scan_ids=np.unique(sid[chosen]).astype(np.int64)))
        # Remove ONLY the actual finite patch; disconnected coplanar surfaces remain.
        remaining = remaining[~np.isin(remaining, chosen)]
    return patches, {'input_points': len(xyz), 'sampled_points': len(selected),
                     'patch_count': len(patches), 'method': 'voxel_ransac_connected_refit',
                     'accumulation_is_not_deskew': True}


def match_sfm_to_patches(camera_xyz_sfm: NDArray, feature_xy: NDArray,
                         patches: Sequence[LidarPatch], T_camera_lidar: NDArray,
                         sfm_scale_to_metric: float,
                         settings: MatchSettings = MatchSettings()
                         ) -> tuple[SfMPlaneMatches | None, dict]:
    """Conservative, unique, finite-support *provisional* correspondence search.

    Matches are determined ONLY with the initial Hand-Eye, never GT or optimized T.
    Does not treat infinite coplanar wall extensions as observed support.
    """
    cfg = settings
    if (cfg.max_normal_distance_m <= 0 or cfg.max_support_distance_m <= 0 or
            cfg.min_score_margin < 0 or cfg.min_matches_per_patch < 3 or
            cfg.max_matches_per_patch < cfg.min_matches_per_patch):
        raise ValueError('Invalid correspondence settings')
    xyz = np.asarray(camera_xyz_sfm, dtype=np.float64)
    xy = np.asarray(feature_xy, dtype=np.float64)
    T = np.asarray(T_camera_lidar, dtype=np.float64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or xy.shape != (len(xyz), 2) or
            not np.isfinite(xyz).all() or not np.isfinite(xy).all() or
            T.shape != (4, 4) or not np.isfinite(T).all() or
            not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8) or
            not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-4) or
            abs(np.linalg.det(T[:3, :3])-1) > 1e-4 or
            not np.isfinite(sfm_scale_to_metric) or sfm_scale_to_metric <= 0):
        raise ValueError('Invalid SfM points / initial transform / scale')
    if not len(patches) or not len(xyz):
        return None, {'accepted': 0, 'reason': 'empty_candidates'}
    p_lidar = (sfm_scale_to_metric * xyz - T[:3, 3]) @ T[:3, :3]
    scores = np.full((len(xyz), len(patches)), np.inf)
    for i, patch in enumerate(patches):
        distance = np.abs(p_lidar @ patch.normal + patch.offset_m)
        # Nearest finite patch support, not distance to an infinite plane.
        near = cKDTree(patch.support_xyz_lidar).query(p_lidar, workers=1)[0]
        valid = ((distance <= cfg.max_normal_distance_m) &
                 (near <= cfg.max_support_distance_m))
        scores[valid, i] = (distance[valid] / cfg.max_normal_distance_m +
                            near[valid] / cfg.max_support_distance_m)
    best = np.argmin(scores, axis=1)
    best_score = scores[np.arange(len(xyz)), best]
    second = (np.partition(scores, 1, axis=1)[:, 1] if len(patches) > 1
              else np.full(len(xyz), np.inf))
    # Avoid inf-inf when a feature has no eligible patch.
    margin = np.full(len(xyz), np.inf)
    both = np.isfinite(second) & np.isfinite(best_score)
    margin[both] = second[both] - best_score[both]
    accepted = np.isfinite(best_score) & (margin >= cfg.min_score_margin)
    candidates = np.flatnonzero(accepted)
    kept: list[int] = []
    for i in range(len(patches)):
        ids = candidates[best[candidates] == i]
        if len(ids) < cfg.min_matches_per_patch:
            continue
        # Deterministically bound a single large ground plane's contribution.
        ids = ids[np.argsort(best_score[ids], kind='stable')]
        if len(ids) > cfg.max_matches_per_patch:
            # Prefer image coverage rather than selecting only one dense texture cluster.
            region = xy[ids]
            low = region.min(axis=0)
            span = np.maximum(region.max(axis=0) - low, 1.0)
            tile = np.minimum(((region-low) / span * [8, 6]).astype(int), [7, 5])
            selected, deferred, seen = [], [], set()
            for idx, cell in zip(ids, tile):
                key = tuple(cell.tolist())
                if key not in seen:
                    selected.append(int(idx))
                    seen.add(key)
                else:
                    deferred.append(int(idx))
            ids = np.asarray((selected + deferred)[:cfg.max_matches_per_patch], dtype=int)
        kept.extend(ids.tolist())
    kept = np.asarray(sorted(kept), dtype=np.int64)
    if len(kept) == 0:
        return None, {'accepted': 0, 'potential': int(len(candidates)),
                      'reason': 'insufficient_patch_support'}
    chosen = best[kept]
    matches = SfMPlaneMatches(
        camera_xyz_sfm=xyz[kept].copy(), feature_xy=xy[kept].copy(),
        normal_lidar=np.asarray([patches[j].normal for j in chosen]),
        offset_lidar_m=np.asarray([patches[j].offset_m for j in chosen]),
        patch_id=np.asarray([patches[j].patch_id for j in chosen], dtype=np.int32),
        feature_index=kept.copy())
    matches.validate()
    unique_normals = np.stack([p.normal for p in patches if p.patch_id in matches.patch_id])
    normal_sv = np.linalg.svd(unique_normals, compute_uv=False)
    normal_rank = int(np.count_nonzero(normal_sv > normal_sv[0] * 0.05))
    report = {'accepted': matches.count, 'potential': int(len(candidates)),
              'matched_normal_rank_proxy': normal_rank,
              'matched_normal_singular_values': normal_sv.tolist(),
              'matched_patch_count': len(np.unique(matches.patch_id)),
              'per_patch': {str(p.patch_id): int(np.count_nonzero(matches.patch_id == p.patch_id))
                            for p in patches},
              'status': 'provisional_not_independently_verified',
              'initial_transform_only': True,
              'caveat': 'Nearest finite support does not prove physical surface identity'}
    return matches, report