"""Metric camera plane extraction from dense z-depth (ROI-free).

Port of the 2026-11-07 archived ROI-free experiment's local PCA, seeded
region growing, connected-component screening and overlap NMS. Dataset
intrinsics and depth are explicit inputs; no fixed D1 scene IDs or GT poses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from spade_calib.calibration.ground import Plane


@dataclass(frozen=True)
class CameraGeometry:
    K: NDArray
    distortion: NDArray
    width: int
    height: int

    def __post_init__(self) -> None:
        k = np.asarray(self.K, dtype=np.float64)
        d = np.asarray(self.distortion, dtype=np.float64).ravel()
        if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError('Camera K must be finite 3x3 with positive focal length')
        if d.size not in (0, 4, 5, 8, 12, 14) or not np.isfinite(d).all():
            raise ValueError('Distortion must be 0/4/5/8/12/14 finite OpenCV coefficients')
        if self.width < 16 or self.height < 16:
            raise ValueError('Camera width and height must each be >=16')
        object.__setattr__(self, 'K', k)
        object.__setattr__(self, 'distortion', d)

    @classmethod
    def from_json(cls, path: str | Path) -> 'CameraGeometry':
        doc = json.loads(Path(path).read_text(encoding='utf-8'))
        if doc.get('depth_type') != 'camera_z_m':
            raise ValueError('Camera geometry requires depth_type=camera_z_m (metric optical Z)')
        return cls(doc['K'], doc.get('distortion', []), int(doc['width']), int(doc['height']))


@dataclass(frozen=True)
class CameraPlaneCandidate:
    identifier: str
    plane: Plane
    mask: NDArray[np.bool_]
    rmse_m: float
    pixel_count: int
    image_shape: tuple[int, int]


@dataclass(frozen=True)
class CameraPlaneSettings:
    processing_width: int = 320
    processing_height: int = 180
    min_depth_m: float = 1.0
    max_depth_m: float = 55.0
    local_window: int = 5
    max_local_planarity_m: float = .13
    max_depth_discontinuity_m: float = 1.0
    seed_step: int = 8
    seed_radius: int = 4
    seed_max_rmse_m: float = .07
    plane_distance_m: float = .11
    normal_angle_deg: float = 24.0
    max_region_rmse_m: float = .11
    min_region_pixels: int = 90
    max_planes: int = 24
    nms_overlap: float = .62


def _plane_fit(points: NDArray) -> tuple[Plane, float]:
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 3 or p.shape[1:] != (3,):
        raise ValueError('Plane fit requires >=3 3D points')
    center = p.mean(axis=0)
    cov = (p - center).T @ (p - center) / len(p)
    eig, directions = np.linalg.eigh(cov)
    normal = directions[:, 0]
    return Plane(normal, -float(normal @ center)), float(np.sqrt(max(eig[0], 0.)))


def dense_metric_xyz(depth_m: NDArray, geometry: CameraGeometry,
                     *, width: int, height: int,
                     min_depth_m: float = 1., max_depth_m: float = 55.,
                     ) -> tuple[NDArray, NDArray, NDArray]:
    """Resize z-depth and backproject using undistorted rays in Camera axes.

    The incoming depth is optical-axis Z (not Euclidean ray range and not
    unscaled DA2 inverse depth). Calibration K refers to its full resolution.
    """
    d = np.asarray(depth_m, dtype=np.float64)
    if d.shape != (geometry.height, geometry.width):
        raise ValueError('Depth shape does not match CameraGeometry width/height')
    if width < 16 or height < 16:
        raise ValueError('Processing resolution too small')
    if not np.isfinite(d[np.isfinite(d)]).all():
        raise ValueError('Depth contains non-finite finite entries')
    sampled = cv2.resize(d.astype(np.float32), (width, height),
                         interpolation=cv2.INTER_NEAREST).astype(np.float64)
    good = np.isfinite(sampled) & (sampled > min_depth_m) & (sampled < max_depth_m)
    # Grid centers map to full-resolution camera pixel centers.
    uu, vv = np.meshgrid((np.arange(width) + .5) * geometry.width / width - .5,
                         (np.arange(height) + .5) * geometry.height / height - .5)
    uv = np.stack([uu, vv], axis=-1).reshape(-1, 1, 2).astype(np.float64)
    raw_rays = cv2.undistortPoints(uv, geometry.K, geometry.distortion)[:, 0, :]
    rays = np.column_stack([raw_rays, np.ones(len(raw_rays))]).reshape(height, width, 3)
    xyz = rays * np.where(good, sampled, 0.)[:, :, None]
    return xyz, good, sampled


def extract_camera_planes(depth_m: NDArray, geometry: CameraGeometry,
                          settings: CameraPlaneSettings = CameraPlaneSettings()
                          ) -> tuple[list[CameraPlaneCandidate], dict]:
    """Detect image-supported metric planes independently of LiDAR matches."""
    cfg = settings
    if cfg.local_window < 3 or cfg.local_window % 2 != 1:
        raise ValueError('local_window must be odd and >=3')
    if cfg.seed_step < 1 or cfg.seed_radius < 1 or cfg.min_region_pixels < 3:
        raise ValueError('Invalid Camera Plane settings')
    height = min(cfg.processing_height, geometry.height)
    width = min(cfg.processing_width, geometry.width)
    xyz, good, depth = dense_metric_xyz(depth_m, geometry, width=width, height=height,
                                        min_depth_m=cfg.min_depth_m,
                                        max_depth_m=cfg.max_depth_m)
    box = lambda a: cv2.boxFilter(np.asarray(a, dtype=np.float64), -1,
                                   (cfg.local_window, cfg.local_window),
                                   normalize=False, borderType=cv2.BORDER_REFLECT101)
    count = box(good.astype(float))
    denom = np.maximum(count, 1.)
    means = np.stack([box(xyz[..., j]) / denom for j in range(3)], axis=-1)
    covariance = np.empty((height, width, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            covariance[..., i, j] = box(xyz[..., i] * xyz[..., j]) / denom - (
                means[..., i] * means[..., j])
    covariance = .5 * (covariance + np.swapaxes(covariance, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[..., 0]
    grad = cv2.morphologyEx(depth.astype('float32'), cv2.MORPH_GRADIENT,
                            np.ones((3, 3), np.uint8))
    locally_planar = (
        good & (count >= cfg.local_window**2 * .95) &
        (np.sqrt(np.maximum(eigenvalues[..., 0], 0)) < cfg.max_local_planarity_m) &
        (grad < cfg.max_depth_discontinuity_m)
    )

    cos_gate = np.cos(np.deg2rad(cfg.normal_angle_deg))
    raw_regions: list[tuple[int, NDArray, Plane, float]] = []
    margin = cfg.seed_radius + 1
    for y in range(margin, height-margin, cfg.seed_step):
        for x in range(margin, width-margin, cfg.seed_step):
            if not locally_planar[y, x]:
                continue
            seed_mask = locally_planar[
                y-cfg.seed_radius:y+cfg.seed_radius+1,
                x-cfg.seed_radius:x+cfg.seed_radius+1,
            ]
            if np.count_nonzero(seed_mask) < 3 * (2*cfg.seed_radius+1)**2 // 5:
                continue
            seed_pts = xyz[y-cfg.seed_radius:y+cfg.seed_radius+1,
                           x-cfg.seed_radius:x+cfg.seed_radius+1][seed_mask]
            seed_plane, seed_rmse = _plane_fit(seed_pts)
            if seed_rmse > cfg.seed_max_rmse_m:
                continue
            dist = np.abs(xyz @ seed_plane.normal + seed_plane.offset_m)
            compatible = (locally_planar & (dist < cfg.plane_distance_m) &
                          (np.abs(normals @ seed_plane.normal) > cos_gate))
            count_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                compatible.astype(np.uint8), connectivity=8)
            label_at_seed = int(labels[y, x])
            if label_at_seed == 0 or label_at_seed >= count_labels:
                continue
            if stats[label_at_seed, cv2.CC_STAT_AREA] < cfg.min_region_pixels:
                continue
            region = labels == label_at_seed
            plane, rmse = _plane_fit(xyz[region])
            if rmse > cfg.max_region_rmse_m:
                continue
            region &= np.abs(xyz @ plane.normal + plane.offset_m) < cfg.plane_distance_m * 1.2
            if np.count_nonzero(region) < cfg.min_region_pixels:
                continue
            plane, rmse = _plane_fit(xyz[region])
            if rmse > cfg.max_region_rmse_m:
                continue
            raw_regions.append((int(np.count_nonzero(region)), region, plane, rmse))
    raw_regions.sort(key=lambda item: -item[0])
    selected: list[CameraPlaneCandidate] = []
    for area, mask, plane, rmse in raw_regions:
        if any(np.count_nonzero(mask & kept.mask) /
               max(1, min(area, kept.pixel_count)) > cfg.nms_overlap
               for kept in selected):
            continue
        selected.append(CameraPlaneCandidate(
            identifier=f'camera_plane_{len(selected)}', plane=plane,
            mask=mask, rmse_m=rmse, pixel_count=area,
            image_shape=(height, width),
        ))
        if len(selected) >= cfg.max_planes:
            break
    return selected, {
        'input_depth_shape': [geometry.height, geometry.width],
        'processing_shape': [height, width],
        'metric_depth_valid_pixels': int(good.sum()),
        'locally_planar_pixels': int(locally_planar.sum()),
        'seeded_regions': len(raw_regions),
        'camera_plane_count': len(selected),
        'method': 'archived_local_pca_seed_region_grow_connected_components_nms',
        'not_a_ground_truth_label': True,
    }
