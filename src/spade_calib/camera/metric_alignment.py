"""DA2 relative inverse-depth proxy -> metric Camera optical Z via SfM anchors.

The archived D1 model was 1/Z = a*q + b, q=raw relative DA2, with a>0.
This implementation uses a condition-improved reparameterization and spatial
holdout tiles. Fit uses TRAIN anchors only; heldout depth is diagnostic and
optionally gates emission. This is not a physical GT depth estimator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from spade_calib.io.legacy import LegacySfMAnchors


@dataclass(frozen=True)
class MetricAlignmentSettings:
    min_anchor_depth_m: float = 1.0
    max_anchor_depth_m: float = 80.0
    max_reprojection_error_px: float = 3.0
    spatial_tiles_x: int = 8
    spatial_tiles_y: int = 6
    holdout_fraction: float = 0.25
    min_train_anchors: int = 40
    min_holdout_anchors: int = 10
    max_holdout_median_relative_error: float = 0.30
    min_output_depth_m: float = 0.5
    max_output_depth_m: float = 80.0
    min_output_valid_fraction: float = 0.05

    def __post_init__(self) -> None:
        scalar_pos = ('min_anchor_depth_m', 'max_anchor_depth_m',
                      'max_reprojection_error_px', 'max_holdout_median_relative_error',
                      'min_output_depth_m', 'max_output_depth_m',
                      'min_output_valid_fraction')
        for k in scalar_pos:
            v = getattr(self, k)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f'{k} must be positive and finite')
        if self.max_anchor_depth_m <= self.min_anchor_depth_m:
            raise ValueError('Invalid SfM anchor metric depth bounds')
        if self.max_output_depth_m <= self.min_output_depth_m:
            raise ValueError('Invalid output metric depth bounds')
        if not 0 < self.holdout_fraction < 0.5:
            raise ValueError('holdout_fraction must be in (0,0.5)')
        if not 0 < self.min_output_valid_fraction < 1:
            raise ValueError('min_output_valid_fraction must be in (0,1)')
        if self.spatial_tiles_x < 2 or self.spatial_tiles_y < 2:
            raise ValueError('Spatial tile grid must be at least 2x2')
        if self.min_train_anchors < 10 or self.min_holdout_anchors < 3:
            raise ValueError('Minimum anchor count too small')


def _split_spatial_tiles(xy: NDArray, width: int, height: int,
                         cfg: MetricAlignmentSettings) -> NDArray[np.bool_]:
    """Tile-level deterministic train/test split; never split one tile by point."""
    tx = np.minimum((xy[:, 0] / width * cfg.spatial_tiles_x).astype(int),
                    cfg.spatial_tiles_x - 1)
    ty = np.minimum((xy[:, 1] / height * cfg.spatial_tiles_y).astype(int),
                    cfg.spatial_tiles_y - 1)
    tile = ty * cfg.spatial_tiles_x + tx
    groups, counts = np.unique(tile, return_counts=True)
    if len(groups) < 3:
        raise ValueError('Anchors occupy <3 spatial tiles; cannot validate local depth fit')
    # Fixed permutation, independent of depth or model errors.
    rng = np.random.default_rng(20261117)
    perm = rng.permutation(len(groups))
    chosen = []
    target = max(1, round(cfg.holdout_fraction * len(xy)))
    total = 0
    # Reserve at least one tile for training.
    for index in perm[:max(1, len(groups) - 1)]:
        chosen.append(int(groups[index]))
        total += int(counts[index])
        if total >= target:
            break
    return np.isin(tile, chosen)


def _metrics(q: NDArray, z: NDArray, a: float, b: float) -> dict:
    inverse = a*q + b
    valid = np.isfinite(inverse) & (inverse > 0)
    predicted = np.full_like(z, np.nan, dtype=np.float64)
    predicted[valid] = 1. / inverse[valid]
    err = np.abs(predicted - z)
    rel = err / z
    good = np.isfinite(err)
    if not good.any():
        return {'valid': 0, 'median_absolute_error_m': None,
                'p90_absolute_error_m': None, 'median_relative_error': None}
    return {
        'valid': int(good.sum()),
        'median_absolute_error_m': float(np.median(err[good])),
        'p90_absolute_error_m': float(np.percentile(err[good], 90)),
        'median_relative_error': float(np.median(rel[good])),
    }


def align_relative_da2_to_sfm(raw_depth: NDArray,
                              anchors: LegacySfMAnchors,
                              settings: MetricAlignmentSettings = MetricAlignmentSettings()
                              ) -> tuple[NDArray[np.float32], dict]:
    """Fit inverse metric depth on spatial training anchors and validate heldout.

    Raw DA2 must be the UNNORMALIZED relative model array for the same image.
    Source anchors must be metric Camera Z, with independent COLMAP projection
    and the Hand-Eye SfM scale (not derived by projected KITTI LiDAR or KITTI GT).
    """
    raw = np.asarray(raw_depth, dtype=np.float64)
    if raw.ndim != 2 or raw.shape != anchors.image_shape:
        raise ValueError('Raw DA2 shape must match SfM anchor image_shape')
    if not np.isfinite(raw).all():
        raise ValueError('Raw DA2 must be finite; infer_image output expected')
    if float(np.ptp(raw)) <= 1e-10:
        raise ValueError('Raw DA2 prediction is constant')
    xy = np.asarray(anchors.xy, dtype=np.float64)
    z = np.asarray(anchors.depth_m, dtype=np.float64)
    error = np.asarray(anchors.reprojection_error_px, dtype=np.float64)
    # Bilinear sampling is the original D1 choice. Do not reorder XY or normalize q.
    q = map_coordinates(raw, [xy[:, 1], xy[:, 0]], order=1, mode='nearest')
    selected = (np.isfinite(q) & np.isfinite(z) & (z >= settings.min_anchor_depth_m)
                & (z <= settings.max_anchor_depth_m)
                & (error <= settings.max_reprojection_error_px))
    xy, q, z = xy[selected], q[selected], z[selected]
    if len(q) < settings.min_train_anchors + settings.min_holdout_anchors:
        raise ValueError('Too few valid SfM anchors for train + heldout fitting')
    val = _split_spatial_tiles(xy, raw.shape[1], raw.shape[0], settings)
    train = ~val
    if int(train.sum()) < settings.min_train_anchors or int(val.sum()) < settings.min_holdout_anchors:
        raise ValueError('Spatial heldout/train set insufficient; use another frame')
    q_train, z_train = q[train], z[train]
    center = float(np.median(q_train))
    scale_q = float(np.std(q_train))
    if scale_q <= 1e-8:
        raise ValueError('Train DA2 values insufficiently varied; inverse-depth scale not identifiable')
    normq = (q_train - center) / scale_q
    inv_z = 1. / z_train
    # Initial linear solution; reject reversed relative-depth orientation.
    guess = np.linalg.lstsq(np.column_stack([normq, np.ones(len(normq))]),
                            inv_z, rcond=None)[0]
    if not np.isfinite(guess).all() or guess[0] <= 1e-12:
        raise ValueError('DA2 prediction is not positively correlated with inverse depth')
    inv_sigma = float(max(0.002, 1.4826 * np.median(np.abs(
        inv_z - (normq * guess[0] + guess[1])))))
    # Log positive slope, unrestricted intercept. No ad-hoc clipping of depth values.
    def residual(x: NDArray) -> NDArray:
        return (np.exp(x[0]) * normq + x[1] - inv_z) / inv_sigma

    opt = least_squares(residual, [np.log(guess[0]), guess[1]],
                        loss='soft_l1', f_scale=1.0, max_nfev=400)
    if not opt.success:
        raise ValueError(f'SfM metric alignment failed: {opt.message}')
    a = float(np.exp(opt.x[0]) / scale_q)
    b = float(opt.x[1] - a * center)
    training = _metrics(q_train, z_train, a, b)
    heldout = _metrics(q[val], z[val], a, b)
    if heldout['valid'] < settings.min_holdout_anchors:
        raise ValueError('Insufficient valid positive metric depths on heldout anchors')
    if heldout['median_relative_error'] > settings.max_holdout_median_relative_error:
        raise ValueError(
            f'Heldout SfM anchor median relative error {heldout["median_relative_error"]:.3f} '
            f'exceeds threshold {settings.max_holdout_median_relative_error:.3f}'
        )
    inv_map = a * raw + b
    valid = (np.isfinite(inv_map) & (inv_map >= 1. / settings.max_output_depth_m)
             & (inv_map <= 1. / settings.min_output_depth_m))
    metric = np.full(raw.shape, np.nan, dtype=np.float32)
    metric[valid] = (1. / inv_map[valid]).astype(np.float32)
    fraction = float(valid.mean())
    if fraction < settings.min_output_valid_fraction:
        raise ValueError(f'Metric prediction coverage {fraction:.3f} below minimum')
    report = {
        'schema_version': 1,
        'kind': 'sfm_anchor_aligned_da2_metric_camera_z_not_physical_gt',
        'depth_type': 'camera_z_m',
        'inverse_depth_model': '1 / z_camera_m = a * raw_DA2_relative + b',
        'a_positive': float(a),
        'b_signed_inverse_m': float(b),
        'train_anchor_count': int(train.sum()),
        'heldout_anchor_count': int(val.sum()),
        'anchor_reprojection_filter_px': settings.max_reprojection_error_px,
        'train': training, 'heldout': heldout,
        'source_anchor_count': int(anchors.count),
        'accepted_anchor_count': int(len(q)),
        'output_shape': list(raw.shape),
        'metric_valid_pixels': int(valid.sum()),
        'metric_valid_fraction': fraction,
        'sfm_scale_to_metric': anchors.sfm_scale,
        'gt_used_for_fitting': False,
        'holdout_strategy': 'spatial_tile_group_split_seed_20261117',
        'fit_objective': 'soft_l1 normalized inverse depth errors on training anchors',
        'caveats': [
            'SfM anchor depths are not independent physical depth GT.',
            'Hand-Eye global SfM scale inherits trajectory-estimation uncertainty.',
            'Held-out tiles share the same underlying SfM reconstruction.',
            'DA2 raw output is a relative inverse-depth proxy; invalid extrapolations are NaN.',
        ],
    }
    return metric, report
