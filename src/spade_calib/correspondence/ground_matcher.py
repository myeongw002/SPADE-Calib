"""Camera-guided LiDAR ground correspondence from metric depth + rigid PCD.

Adapted from archived 2026-11-07 run_camera_guided.py: image mask dilation,
projection, metric depth gate, normal-constrained RANSAC, 3D connected support,
and geometric acceptance gates. Automatic output remains PROVISIONAL: it is
not physical correspondence ground truth and must be independently reviewed.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from spade_calib.calibration.ground import GroundMatch, Plane, transform_plane_to_camera
from spade_calib.camera.planes import CameraGeometry, CameraPlaneCandidate, _plane_fit
from spade_calib.lidar.ground_planes import LidarGroundCandidate


@dataclass(frozen=True)
class GroundMatchingSettings:
    mask_dilation_grid_px: int = 10
    max_match_points: int = 4500
    min_guided_points: int = 35
    guided_iterations: int = 300
    guided_distance_m: float = .11
    guided_max_normal_angle_deg: float = 35.
    max_camera_normal_angle_deg: float = 25.
    max_lidar_normal_difference_deg: float = 18.
    max_lidar_offset_difference_m: float = .35
    max_camera_offset_difference_m: float = 1.2
    max_depth_error_m: float = 1.8
    relative_depth_gate: float = .23
    min_inside_mask_fraction: float = .18
    connected_radius_m: float = .75
    min_candidate_pixels: int = 90
    max_camera_plane_rmse_m: float = .12
    random_seed: int = 20261107


def _plane_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.rad2deg(np.arccos(np.clip(abs(float(a @ b)), 0., 1.))))


def _connected_region(xyz: np.ndarray, radius: float) -> np.ndarray:
    """Largest spatially connected 3D patch, not distant coplanar fragments."""
    if len(xyz) < 2:
        return np.arange(len(xyz), dtype=int)
    tree = cKDTree(xyz)
    edges = tree.query_pairs(radius, output_type='ndarray')
    if len(edges) == 0:
        return np.arange(0, 0, dtype=int)
    edges = np.vstack([edges, edges[:, ::-1]])
    graph = coo_matrix(
        (np.ones(len(edges), dtype=np.int8), (edges[:, 0], edges[:, 1])),
        shape=(len(xyz), len(xyz)),
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels)
    return np.flatnonzero(labels == int(np.argmax(counts)))


def _guided_ransac(xyz: np.ndarray, expected_normal: np.ndarray, *,
                   settings: GroundMatchingSettings, seed: int
                   ) -> tuple[Plane, np.ndarray, float] | None:
    cfg = settings
    if len(xyz) < cfg.min_guided_points:
        return None
    rng = np.random.default_rng(cfg.random_seed + seed)
    cos_gate = np.cos(np.deg2rad(cfg.guided_max_normal_angle_deg))
    best_count, best_mask = 0, None
    for _ in range(cfg.guided_iterations):
        tri = xyz[rng.choice(len(xyz), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        nn = np.linalg.norm(n)
        if nn < 1e-8:
            continue
        n /= nn
        if abs(n @ expected_normal) < cos_gate:
            continue
        d = -float(n @ tri[0])
        mask = np.abs(xyz @ n + d) < cfg.guided_distance_m
        count = int(mask.sum())
        if count > best_count:
            best_count, best_mask = count, mask
    if best_mask is None or best_count < cfg.min_guided_points:
        return None
    plane, _ = _plane_fit(xyz[best_mask])
    if _plane_angle_deg(plane.normal, expected_normal) > cfg.guided_max_normal_angle_deg:
        return None
    refined_ids = np.flatnonzero(
        np.abs(xyz @ plane.normal + plane.offset_m) < cfg.guided_distance_m)
    if len(refined_ids) < cfg.min_guided_points:
        return None
    region = _connected_region(xyz[refined_ids], cfg.connected_radius_m)
    if len(region) < cfg.min_guided_points:
        return None
    kept = refined_ids[region]
    plane, rmse = _plane_fit(xyz[kept])
    kept = kept[np.abs(xyz[kept] @ plane.normal + plane.offset_m) <
                1.2 * cfg.guided_distance_m]
    if len(kept) < cfg.min_guided_points:
        return None
    plane, rmse = _plane_fit(xyz[kept])
    if _plane_angle_deg(plane.normal, expected_normal) > cfg.guided_max_normal_angle_deg:
        return None
    return plane, kept, rmse


def match_ground(
    camera_planes: list[CameraPlaneCandidate], lidar_ground: LidarGroundCandidate,
    depth_m: np.ndarray, camera: CameraGeometry, T_camera_lidar: np.ndarray,
    settings: GroundMatchingSettings = GroundMatchingSettings(),
) -> tuple[GroundMatch | None, dict]:
    """Return at most one unique physical-ground hypothesis and diagnostics.

    This is an automatic *hypothesis*, not independently verified measurement.
    The initial extrinsic guides search only; no GT or optimized result is used.
    """
    cfg = settings
    if cfg.mask_dilation_grid_px < 0 or cfg.max_match_points < cfg.min_guided_points:
        raise ValueError('Invalid guided matching window/sample size')
    depth = np.asarray(depth_m)
    if depth.shape != (camera.height, camera.width):
        raise ValueError('Camera depth and camera geometry dimensions differ')
    T = np.asarray(T_camera_lidar, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('Initial extrinsic must be a finite 4x4')
    R, t = T[:3, :3], T[:3, 3]
    points = lidar_ground.xyz
    pc = points @ R.T + t
    allowed = np.isfinite(pc).all(axis=1) & (pc[:, 2] > .6) & (
        pc[:, 2] < max(55., float(np.nanmax(depth[np.isfinite(depth)]))
                       if np.isfinite(depth).any() else 55.))
    lidar_ids = np.flatnonzero(allowed)
    if not len(lidar_ids):
        raise ValueError('No LiDAR ground-candidate points visible in Camera front direction')
    projected, _ = cv2.projectPoints(
        pc[lidar_ids].reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
        camera.K, camera.distortion,
    )
    uv = projected[:, 0, :]
    within = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (
        uv[:, 0] < camera.width) & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height)
    lidar_ids, uv = lidar_ids[within], uv[within]
    if not len(lidar_ids):
        raise ValueError('No LiDAR points project into camera image')
    rays_xy = cv2.undistortPoints(
        uv.reshape(-1, 1, 2).astype(np.float64),
        camera.K, camera.distortion)[:, 0, :]
    rays = np.column_stack([rays_xy, np.ones(len(rays_xy))])
    results = []
    for index, candidate in enumerate(camera_planes):
        outcome = {
            'camera_plane': candidate.identifier,
            'camera_pixels': candidate.pixel_count,
            'camera_rmse_m': candidate.rmse_m,
            'predepth_points': 0, 'depth_gated_points': 0, 'lidar_inliers': 0,
            'normal_angle_deg': None, 'plane_offset_difference_m': None,
            'within_mask_fraction': None, 'median_depth_error_m': None,
            'guided_rmse_m': None, 'accepted': False, 'reason': None,
        }
        if (candidate.pixel_count < cfg.min_candidate_pixels or
                candidate.rmse_m > cfg.max_camera_plane_rmse_m):
            outcome['reason'] = 'camera_patch_quality'; results.append(outcome); continue
        n_expected = R @ lidar_ground.plane.normal
        if _plane_angle_deg(n_expected, candidate.plane.normal) > cfg.max_camera_normal_angle_deg:
            outcome['reason'] = 'camera_normal_prior'; results.append(outcome); continue
        h, w = candidate.image_shape
        px = np.floor(uv[:, 0] * w / camera.width).astype(int)
        py = np.floor(uv[:, 1] * h / camera.height).astype(int)
        px, py = np.clip(px, 0, w-1), np.clip(py, 0, h-1)
        kernel = np.ones((2*cfg.mask_dilation_grid_px+1,)*2, np.uint8)
        broad = cv2.dilate(candidate.mask.astype(np.uint8), kernel) > 0
        selected = np.flatnonzero(broad[py, px])
        outcome['predepth_points'] = int(len(selected))
        if len(selected) < cfg.min_guided_points:
            outcome['reason'] = 'no_projected_mask_support'; results.append(outcome); continue
        den = rays[selected] @ candidate.plane.normal
        predicted_z = np.full(len(selected), np.nan)
        ok = np.abs(den) > 1e-6
        predicted_z[ok] = -candidate.plane.offset_m / den[ok]
        observed_z = pc[lidar_ids[selected], 2]
        depth_error = np.abs(observed_z - predicted_z)
        depth_allowed = (
            np.isfinite(predicted_z) & (predicted_z > .6) &
            (depth_error < np.maximum(cfg.max_depth_error_m,
                                      cfg.relative_depth_gate * predicted_z))
        )
        selected = selected[depth_allowed]
        outcome['depth_gated_points'] = int(len(selected))
        if len(selected) < cfg.min_guided_points:
            outcome['reason'] = 'depth_gate'; results.append(outcome); continue
        # Only ground-near LiDAR samples: prevent a wall matching a broad image patch.
        near_lidar_ground = (
            np.abs(points[lidar_ids[selected]] @ lidar_ground.plane.normal +
                   lidar_ground.plane.offset_m) < cfg.max_lidar_offset_difference_m
        )
        selected = selected[near_lidar_ground]
        if len(selected) < cfg.min_guided_points:
            outcome['reason'] = 'lidar_ground_prior'; results.append(outcome); continue
        if len(selected) > cfg.max_match_points:
            rng = np.random.default_rng(cfg.random_seed + index)
            selected = np.sort(rng.choice(selected, cfg.max_match_points, replace=False))
        ids = lidar_ids[selected]
        expected_lidar_normal = R.T @ candidate.plane.normal
        fit = _guided_ransac(points[ids], expected_lidar_normal, settings=cfg, seed=index)
        if fit is None:
            outcome['reason'] = 'guided_ransac_or_spatial_support'
            results.append(outcome); continue
        plane, support, rmse = fit
        if plane.normal @ lidar_ground.plane.normal < 0:
            plane = Plane(-plane.normal, -plane.offset_m)
        lidar_angle = _plane_angle_deg(plane.normal, lidar_ground.plane.normal)
        lidar_offset = abs(plane.offset_m - lidar_ground.plane.offset_m)
        inside = float(np.mean(candidate.mask[py[selected[support]], px[selected[support]]]))
        matched_camera = transform_plane_to_camera(plane, R, t)
        sign = 1. if matched_camera.normal @ candidate.plane.normal >= 0 else -1.
        normal_angle = _plane_angle_deg(matched_camera.normal, candidate.plane.normal)
        camera_offset = abs(matched_camera.offset_m - sign * candidate.plane.offset_m)
        signed_depth = np.abs(
            pc[ids[support], 2] -
            (-candidate.plane.offset_m /
             np.where(np.abs(rays[selected[support]] @ candidate.plane.normal) > 1e-6,
                      rays[selected[support]] @ candidate.plane.normal, np.nan))
        )
        finite_depth = signed_depth[np.isfinite(signed_depth)]
        median_depth = float(np.median(finite_depth)) if len(finite_depth) else float('inf')
        passed = (len(support) >= cfg.min_guided_points and
                  normal_angle <= cfg.max_camera_normal_angle_deg and
                  lidar_angle <= cfg.max_lidar_normal_difference_deg and
                  lidar_offset <= cfg.max_lidar_offset_difference_m and
                  camera_offset <= cfg.max_camera_offset_difference_m and
                  inside >= cfg.min_inside_mask_fraction and
                  median_depth <= max(cfg.max_depth_error_m, .23 * float(np.median(
                      pc[ids[support], 2]))))
        outcome.update({
            'lidar_inliers': len(support), 'guided_rmse_m': rmse,
            'normal_angle_deg': normal_angle, 'plane_offset_difference_m': camera_offset,
            'within_mask_fraction': inside, 'median_depth_error_m': median_depth,
            'lidar_plane_difference_m': lidar_offset,
            'lidar_plane_normal_difference_deg': lidar_angle,
            'accepted': bool(passed),
            'reason': 'passed_geometric_gates' if passed else 'failed_geometric_gates',
        })
        # Score explicit and independent of KITTI GT.
        outcome['_rank'] = (len(support) * inside /
                            (1. + camera_offset + normal_angle / 10.))
        if passed:
            outcome['_plane'] = plane
        results.append(outcome)
    viable = [v for v in results if v['accepted']]
    best = max(viable, key=lambda x: x['_rank']) if viable else None
    match = None
    if best is not None:
        camera_plane = next(c for c in camera_planes if c.identifier == best['camera_plane'])
        match = GroundMatch(
            match_id='auto_ground_' + camera_plane.identifier,
            lidar=best['_plane'], camera=camera_plane.plane,
            sigma_normal_deg=3., sigma_offset_m=.15)
    for v in results:
        v.pop('_plane', None)
        v.pop('_rank', None)
    return match, {
        'method': 'camera_guided_projection_mask_metric_depth_normal_ransac_connected',
        'projected_points_in_fov': len(lidar_ids),
        'camera_plane_attempts': len(camera_planes),
        'accepted_candidates_before_unique_selection': len(viable),
        'selected_camera_plane': best['camera_plane'] if best is not None else None,
        'candidates': results,
        'correspondence_status': 'provisional_requires_independent_review',
        'uses_gt_extrinsic': False,
        'no_manual_roi': True,
    }
