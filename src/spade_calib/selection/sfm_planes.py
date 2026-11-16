"""Window-local plane-normal screening for GT-fixed sparse SfM.

The plane proposal geometry is adapted from sfm_plane_window_screen.py:
local-neighbor RANSAC, refit, connected support, width-ratio rejection,
multi-view image support and normal-information log-det ranking.

Unlike the original whole-model proposal -> visibility-only window assignment,
a candidate plane here is fitted AGAIN using ONLY 3D points with sufficient
IN-WINDOW tracks. Global-model XYZ still depends on outside-window views; this
is explicitly reported and is NOT local retriangulation.

Information H = sum(w * n_world n_world.T) is a heuristic translation-normal
diversity proxy, NOT a Camera-LiDAR extrinsic 6DoF Fisher information matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError, cKDTree


@dataclass(frozen=True)
class PlaneObservation:
    """One image feature corresponding to a triangulated SfM point."""
    image_id: int
    x: float
    y: float


@dataclass(frozen=True)
class PlanePoint:
    xyz: NDArray[np.float64]  # metric world XYZ because Camera Pose is KITTI GT
    observations: tuple[PlaneObservation, ...]  # strictly inside candidate window
    reprojection_px: float
    parallax_deg: float
    has_outside_observations: bool = False


@dataclass(frozen=True)
class ImageGeometry:
    width: int
    height: int
    R_cw: NDArray[np.float64]


@dataclass(frozen=True)
class PlaneScreenSettings:
    threshold_m: float = 0.15
    min_inliers: int = 70
    max_planes: int = 8
    max_points: int = 12000
    ransac_iters: int = 400
    neighbors: int = 24
    connectivity_factor: float = 3.0
    min_width_ratio: float = 0.08
    min_visible_frames: int = 2
    min_image_observations: int = 8
    min_hull_fraction: float = 0.003
    min_grid_cells: int = 3
    image_grid_x: int = 8
    image_grid_y: int = 6
    normal_information_weight: float = 0.15
    regularizer: float = 0.01
    min_normal_rank: int = 3
    normal_rank_ratio: float = 0.02
    min_normal_eigen_fraction: float = 0.02
    min_planes: int = 3
    exclude_ground: bool = False
    ground_angle_deg: float = 25.0
    # The source's translation-motion prior is optional, since its scale and
    # calibration objective are not independently justified.
    weak_motion_guidance: bool = False
    motion_gap_frames: int = 20
    motion_stride_frames: int = 10
    seed: int = 20261113

    def __post_init__(self) -> None:
        for value in (self.threshold_m, self.connectivity_factor,
                      self.min_width_ratio, self.normal_information_weight,
                      self.regularizer, self.normal_rank_ratio,
                      self.min_normal_eigen_fraction, self.ground_angle_deg):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Plane thresholds/weights must be finite and positive')
        for value in (self.min_inliers, self.max_planes, self.max_points,
                      self.ransac_iters, self.neighbors, self.min_visible_frames,
                      self.min_image_observations, self.min_grid_cells,
                      self.image_grid_x, self.image_grid_y, self.min_normal_rank,
                      self.min_planes, self.motion_gap_frames, self.motion_stride_frames):
            if value < 1:
                raise ValueError('Plane counts must be positive')
        if (self.min_inliers < 3 or self.max_points < self.min_inliers or
                self.neighbors < 3 or self.min_normal_rank > 3 or
                self.min_normal_eigen_fraction >= 1 / 3 or
                not 0 <= self.min_hull_fraction < 1 or
                not 0 < self.ground_angle_deg < 90):
            raise ValueError('Inconsistent plane screening configuration')


def _fit_plane(x: NDArray) -> tuple[NDArray, float, float, float]:
    center = np.mean(x, axis=0)
    cov = (x - center).T @ (x - center) / len(x)
    eigen, vecs = np.linalg.eigh(cov)
    normal = vecs[:, 0]
    offset = -float(normal @ center)
    rms = math.sqrt(max(float(eigen[0]), 0.0))
    width_ratio = math.sqrt(max(float(eigen[1]), 0.) /
                            max(float(eigen[2]), 1e-15))
    return normal, offset, rms, width_ratio


def _connected_component(x: NDArray, threshold_m: float,
                         factor: float) -> NDArray[np.int64]:
    """Bounded-k 3D connectivity avoids quadratic query_pairs memory blowup."""
    if len(x) < 4:
        return np.arange(len(x), dtype=np.int64)
    tree = cKDTree(x)
    d, idx = tree.query(x, k=min(13, len(x)), workers=1)
    nearest = d[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 1e-9)]
    radius = max(float(np.percentile(nearest, 75)) * factor
                 if len(nearest) else 0., threshold_m * 4)
    mask = (d[:, 1:] <= radius) & np.isfinite(d[:, 1:])
    row = np.broadcast_to(np.arange(len(x))[:, None], idx[:, 1:].shape)[mask]
    col = idx[:, 1:][mask]
    if not len(row):
        return np.array([0], dtype=np.int64)
    edges = coo_matrix((np.ones(len(row), dtype=np.int8), (row, col)),
                       shape=(len(x), len(x))).tocsr()
    _, labels = connected_components(edges, directed=False)
    return np.flatnonzero(labels == np.argmax(np.bincount(labels)))


def _ground_normal(images: Mapping[int, ImageGeometry]) -> NDArray:
    # KITTI rectified camera +Y convention is approximately down.
    up = np.mean([image.R_cw.T @ [0., 1., 0.] for image in images.values()], axis=0)
    length = float(np.linalg.norm(up))
    if length < 1e-8:
        raise ValueError('Cannot infer a camera-up ground orientation prior')
    return up / length


def _motion_prior(images: Mapping[int, ImageGeometry], settings: PlaneScreenSettings) -> NDArray:
    """World-frame translation-motion surrogate; optional guided proposal only.

    The original script sums J.T@J in unrelated instantaneous camera frames.
    Express every such block in a COMMON world frame to avoid axis mixing.
    """
    ordered = list(images.values())
    matrices = []
    for i in range(0, len(ordered) - settings.motion_gap_frames,
                   settings.motion_stride_frames):
        ri = ordered[i].R_cw
        rj = ordered[i + settings.motion_gap_frames].R_cw
        rel = ri @ rj.T
        jac = rel - np.eye(3)
        matrices.append(ri.T @ (jac.T @ jac) @ ri)
    return np.mean(matrices, axis=0) if matrices else np.zeros((3, 3))


def _gain(h: NDArray, n: NDArray, weight: float, regularizer: float) -> float:
    a = np.linalg.slogdet(np.eye(3) + h / regularizer)[1]
    b = np.linalg.slogdet(np.eye(3) + (h + weight * np.outer(n, n)) / regularizer)[1]
    return max(0., float(b - a))


def _visible_support(indices: NDArray, points: Sequence[PlanePoint],
                     images: Mapping[int, ImageGeometry],
                     settings: PlaneScreenSettings) -> tuple[list[int], set[int], int]:
    """Find real per-window image support; unique 3D IDs matter, not raw detections."""
    per_image: dict[int, list[tuple[int, float, float]]] = {}
    for index in indices:
        for obs in points[int(index)].observations:
            if obs.image_id in images:
                per_image.setdefault(obs.image_id, []).append(
                    (int(index), obs.x, obs.y))
    supported_images: list[int] = []
    supported_ids: set[int] = set()
    for image_id, obs_rows in per_image.items():
        # Deduplicate (point ID, image ID) and reject low-support observations.
        dedup = {index: (x, y) for index, x, y in obs_rows}
        if len(dedup) < settings.min_image_observations:
            continue
        image = images[image_id]
        xy = np.asarray(list(dedup.values()), dtype=float)
        try:
            hull = float(ConvexHull(xy).volume) / (image.width * image.height)
        except (QhullError, ValueError):
            hull = 0.
        if hull < settings.min_hull_fraction:
            continue
        bx = np.clip((xy[:, 0] / image.width * settings.image_grid_x).astype(int),
                     0, settings.image_grid_x - 1)
        by = np.clip((xy[:, 1] / image.height * settings.image_grid_y).astype(int),
                     0, settings.image_grid_y - 1)
        if len(set(zip(bx, by))) < settings.min_grid_cells:
            continue
        supported_images.append(image_id)
        supported_ids.update(dedup)
    return supported_images, supported_ids, sum(
        len(per_image[image]) for image in supported_images)


def screen_window_planes(
    points: Sequence[PlanePoint],
    images: Mapping[int, ImageGeometry],
    settings: PlaneScreenSettings,
) -> tuple[dict, list[dict]]:
    """Run RANSAC on per-window-qualified points and return proxy + plane records.

    All XYZ come from the passed sparse model. In global mode they retain the
    influence of outside-window tracks despite in-window observation filtering.
    """
    rng = np.random.default_rng(settings.seed)
    count = len(points)
    zero_info = np.zeros((3, 3), dtype=float)
    if count < settings.min_inliers:
        return _summarize([], zero_info, count, settings), []
    xyz_all = np.asarray([p.xyz for p in points], dtype=float)
    if xyz_all.shape != (count, 3) or not np.isfinite(xyz_all).all():
        raise ValueError('Window has invalid 3D SfM points')
    ids = np.arange(count, dtype=np.int64)
    if len(ids) > settings.max_points:
        ids = np.sort(rng.choice(ids, settings.max_points, replace=False))
    working = ids.copy()
    found: list[dict] = []
    h = np.zeros((3, 3), dtype=float)
    prior = _motion_prior(images, settings) if settings.weak_motion_guidance else zero_info
    ground = _ground_normal(images) if settings.exclude_ground else None
    min_neighbors = min(settings.neighbors, len(working))
    attempts = 0
    # A rejected huge plane must not produce an infinite rediscovery loop.
    while (len(found) < settings.max_planes and len(working) >= settings.min_inliers
           and attempts < settings.max_planes * 4):
        attempts += 1
        x = xyz_all[working]
        tree = cKDTree(x)
        _, nbr = tree.query(x, k=min(min_neighbors, len(x)), workers=1)
        if nbr.ndim != 2 or nbr.shape[1] < 3:
            break
        best_mask = None
        best_value = float('-inf')
        for _ in range(settings.ransac_iters):
            i = int(rng.integers(len(x)))
            j, k = rng.choice(nbr[i, 1:], size=2, replace=False)
            n = np.cross(x[int(j)] - x[i], x[int(k)] - x[i])
            magnitude = np.linalg.norm(n)
            if magnitude < 1e-10:
                continue
            n /= magnitude
            if ground is not None and abs(n @ ground) >= math.cos(
                    math.radians(settings.ground_angle_deg)):
                continue
            offset = -float(n @ x[i])
            inliers = np.abs(x @ n + offset) < settings.threshold_m
            total = int(np.sum(inliers))
            if total < settings.min_inliers:
                continue
            # Same local-neighbor RANSAC strategy as the uploaded source.
            # Weak-motion gain is a proposal preference, never a true FIM.
            weight = (_gain(h + prior, n, settings.normal_information_weight,
                            settings.regularizer)
                      if settings.weak_motion_guidance else 1.)
            reward = math.sqrt(total) * math.sqrt(max(weight, 1e-12))
            if reward > best_value:
                best_value, best_mask = reward, inliers
        if best_mask is None:
            break
        candidate = working[best_mask]
        # Refit and restrict to actual connected coplanar surface support.
        n, offset, _, _ = _fit_plane(xyz_all[candidate])
        candidate = working[np.abs(xyz_all[working] @ n + offset) <
                            settings.threshold_m]
        component = _connected_component(xyz_all[candidate], settings.threshold_m,
                                         settings.connectivity_factor)
        candidate = candidate[component]
        if len(candidate) < settings.min_inliers:
            working = working[~best_mask]
            continue
        n, offset, _, width_ratio = _fit_plane(xyz_all[candidate])
        candidate = candidate[np.abs(xyz_all[candidate] @ n + offset) <
                              settings.threshold_m]
        if len(candidate) < settings.min_inliers:
            working = working[~best_mask]
            continue
        visible, supported_ids, observations = _visible_support(
            candidate, points, images, settings)
        supported = np.asarray(sorted(supported_ids), dtype=np.int64)
        if (len(visible) < settings.min_visible_frames or
                len(supported) < settings.min_inliers):
            working = working[~best_mask]
            continue
        # Fit again using ONLY window-observed + image-supported points.
        n, offset, rms, width_ratio = _fit_plane(xyz_all[supported])
        if width_ratio < settings.min_width_ratio or rms > settings.threshold_m:
            working = working[~best_mask]
            continue
        # Quality weight is explicit heuristic, not estimated noise precision.
        weight = (settings.normal_information_weight *
                  min(1., len(supported) / (2 * settings.min_inliers)) *
                  (1. / (1. + (rms / settings.threshold_m) ** 2)))
        # Externally-triangulated global SfM points must remain traceable.
        outside_count = sum(points[int(idx)].has_outside_observations
                            for idx in supported)
        plane = {
            'plane_id': len(found),
            'normal_world_xyz': [float(v) for v in n],
            'offset_m': float(offset),
            'normal_information_weight': float(weight),
            'inlier_count': int(len(supported)),
            'connected_candidate_inliers': int(len(candidate)),
            'visible_image_count': len(visible),
            'valid_image_observations': int(observations),
            'plane_fit_rmse_m': float(rms),
            'in_plane_width_ratio': float(width_ratio),
            'candidate_points_with_outside_observations': int(outside_count),
            'window_only_triangulation': False,
        }
        found.append(plane)
        h += weight * np.outer(n, n)
        # Remove all support of this physical patch; do not double-count.
        keep = ~np.isin(working, candidate)
        working = working[keep]
    return _summarize(found, h, count, settings), found


def _summarize(planes: list[dict], h: NDArray,
               input_count: int, settings: PlaneScreenSettings) -> dict:
    values = np.linalg.eigvalsh(h)
    total = float(np.trace(h))
    rank = int(np.count_nonzero(
        values > max(total * settings.normal_rank_ratio, 1e-12)))
    min_fraction = float(max(values[0], 0.) / total) if total > 1e-12 else 0.
    proxy_sufficient = (
        len(planes) >= settings.min_planes
        and rank >= settings.min_normal_rank
        and (settings.min_normal_rank < 3
             or min_fraction >= settings.min_normal_eigen_fraction)
    )
    score = float(np.linalg.slogdet(np.eye(3) + h / settings.regularizer)[1])
    return {
        'plane_input_qualified_points': int(input_count),
        'plane_count': len(planes),
        'plane_normal_rank': rank,
        'plane_normal_min_eigen_fraction': min_fraction,
        'plane_normal_eigenvalues': [float(v) for v in values],
        'plane_normal_information_world': h.tolist(),
        'plane_score_heuristic': score,
        'plane_translation_normal_proxy_sufficient': bool(proxy_sufficient),
        'plane_external_xyz_point_count': int(
            sum(p['candidate_points_with_outside_observations'] for p in planes)),
        'plane_criterion': 'world_normal_diversity_translation_proxy_not_6d_calibration',
    }
