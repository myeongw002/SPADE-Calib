"""Joint Hand-Eye motion + Ground-plane refinement with fixed 7th SfM scale.

Both arms use identical motion pairs and plane correspondences:
  fixed_rotation: translation (3 DOF), Ground offset only;
  optimize_rotation: left Camera-frame rotation and translation (6 DOF),
                     Ground normal and signed offset.
GT is never used by the optimizer. Sigmas/weights are assumptions, not
calibrated physical uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from spade_calib.handeye.residuals import residuals as handeye_motion_residuals
from spade_calib.selection.trajectory import MotionPairs
from .ground import GroundMatch, ground_metrics, ground_residual_vector, prepare_matches


@dataclass(frozen=True)
class GroundRefinementSettings:
    sigma_relative_rotation_deg: float = 0.5
    sigma_relative_translation_m: float = 0.05
    handeye_weight: float = 1.0
    ground_weight: float = 1.0
    max_translation_delta_m: float = 2.0
    max_rotation_delta_deg: float = 10.0
    max_initial_normal_angle_deg: float = 45.0
    robust_loss: str = 'soft_l1'
    max_nfev: int = 250

    def __post_init__(self) -> None:
        for k in ('sigma_relative_rotation_deg', 'sigma_relative_translation_m',
                  'handeye_weight', 'ground_weight', 'max_translation_delta_m',
                  'max_rotation_delta_deg', 'max_initial_normal_angle_deg'):
            val = getattr(self, k)
            if not np.isfinite(val) or val <= 0:
                raise ValueError(f'{k} must be finite and positive')
        if self.max_rotation_delta_deg >= 90 or self.max_initial_normal_angle_deg >= 90:
            raise ValueError('Rotation bounds/normal gates must be < 90 degrees')
        if self.robust_loss not in {'linear', 'soft_l1', 'huber', 'cauchy'}:
            raise ValueError('Unsupported scipy least_squares robust loss')
        if type(self.max_nfev) is not int or self.max_nfev < 1:
            raise ValueError('max_nfev must be a positive integer')


def _validate_transform(transform: NDArray) -> NDArray[np.float64]:
    T = np.asarray(transform, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('Initial T_camera_lidar must be finite 4x4')
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=1e-9):
        raise ValueError('Invalid homogeneous bottom row')
    if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError('Extrinsic rotation is not orthonormal')
    if abs(np.linalg.det(T[:3, :3]) - 1.0) > 1e-5:
        raise ValueError('Extrinsic rotation must be proper')
    return T


def _transform(T0: NDArray, delta: NDArray, mode: str) -> NDArray:
    T = T0.copy()
    if mode == 'fixed_rotation':
        T[:3, 3] = T0[:3, 3] + delta
    else:
        T[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix() @ T0[:3, :3]
        T[:3, 3] = T0[:3, 3] + delta[3:]
    return T


def _motion_residual(pairs: MotionPairs, R: NDArray, t: NDArray,
                     scale: float, cfg: GroundRefinementSettings) -> NDArray:
    r = handeye_motion_residuals(pairs, R, t, scale)
    r[:, :3] /= np.deg2rad(cfg.sigma_relative_rotation_deg)
    r[:, 3:] /= cfg.sigma_relative_translation_m
    # Overlapping motion pairs are correlated; averaging avoids multiplying
    # their relative weight merely by adding more overlapping pairs.
    return np.sqrt(cfg.handeye_weight / pairs.count) * r.reshape(-1)


def _svd_report(fun: Any, zero: NDArray, steps: NDArray) -> dict[str, Any]:
    """Finite-difference local Jacobian, normalized by reference DOF units."""
    m = len(fun(zero))
    J = np.empty((m, len(zero)), dtype=float)
    for i, h in enumerate(steps):
        delta = np.zeros_like(zero)
        delta[i] = h
        J[:, i] = (fun(zero + delta) - fun(zero - delta)) / (2 * h)
    units = np.array([np.deg2rad(1)] * 3 + [0.1] * 3 if len(zero) == 6 else [0.1] * 3)
    sv = np.linalg.svd(J * units[None, :], compute_uv=False)
    rank = int(np.count_nonzero(sv > max(float(sv[0]) if len(sv) else 0., 1e-15) * 1e-8))
    return {
        'normalized_singular_values': sv.tolist(),
        'rank': rank,
        'parameter_count': len(zero),
        'full_column_rank': bool(rank == len(zero)),
        'condition_number': (float(sv[0] / sv[-1]) if len(sv) >= len(zero)
                             and sv[-1] > 1e-12 * sv[0] else None),
        'parameter_units': (['camera_rot_1deg'] * 3 + ['camera_translation_0.1m'] * 3
                            if len(zero) == 6 else ['camera_translation_0.1m'] * 3),
        'note': 'Local weighted geometry sensitivity, not calibrated Fisher information',
    }


def refine_with_ground(initial_T_camera_lidar: NDArray, sfm_scale: float,
                       pairs: MotionPairs, ground_matches: Sequence[GroundMatch],
                       *, mode: Literal['fixed_rotation', 'optimize_rotation'],
                       settings: GroundRefinementSettings = GroundRefinementSettings()
                       ) -> dict[str, Any]:
    """Refine extrinsic with Hand-Eye motion equation and matched Ground planes.

    The SfM scale is held fixed. All planes must be independently measured
    in metric Camera/LiDAR target frames, not derived from KITTI calibration GT.
    """
    if mode not in ('fixed_rotation', 'optimize_rotation'):
        raise ValueError('mode must be fixed_rotation or optimize_rotation')
    initial = _validate_transform(initial_T_camera_lidar)
    if not np.isfinite(sfm_scale) or sfm_scale <= 0:
        raise ValueError('SfM scale must be positive and finite')
    if pairs.count < 3:
        raise ValueError('At least 3 Hand-Eye motion pairs required for joint refinement')
    matches = list(ground_matches)
    signs, validation = prepare_matches(
        matches, initial[:3, :3],
        max_initial_normal_angle_deg=settings.max_initial_normal_angle_deg,
    )
    zero = np.zeros(3 if mode == 'fixed_rotation' else 6, dtype=float)
    bounds_m = settings.max_translation_delta_m
    if mode == 'fixed_rotation':
        lower = np.full(3, -bounds_m)
        upper = np.full(3, bounds_m)
        steps = np.full(3, 1e-5)
    else:
        ang = np.deg2rad(settings.max_rotation_delta_deg)
        lower = np.array([-ang] * 3 + [-bounds_m] * 3)
        upper = -lower
        steps = np.array([1e-6] * 3 + [1e-5] * 3)

    def motion(delta: NDArray) -> NDArray:
        T = _transform(initial, delta, mode)
        return _motion_residual(pairs, T[:3, :3], T[:3, 3], sfm_scale, settings)

    def ground(delta: NDArray) -> NDArray:
        T = _transform(initial, delta, mode)
        return np.sqrt(settings.ground_weight) * ground_residual_vector(
            matches, signs, T[:3, :3], T[:3, 3],
            normal_enabled=(mode == 'optimize_rotation'),
        )

    def combined(delta: NDArray) -> NDArray:
        return np.r_[motion(delta), ground(delta)]

    sol = least_squares(combined, zero, bounds=(lower, upper),
                        loss=settings.robust_loss, max_nfev=settings.max_nfev,
                        x_scale='jac')
    estimated = _transform(initial, sol.x, mode)
    bound_hit = (np.abs(sol.x - lower) <= 1e-5 * np.maximum(1, np.abs(lower))) | (
                 np.abs(sol.x - upper) <= 1e-5 * np.maximum(1, np.abs(upper)))
    return {
        'mode': mode,
        'convention': 'T_camera_lidar',
        'T_camera_lidar': estimated.tolist(),
        'sfm_scale_to_metric_fixed': float(sfm_scale),
        'optimizer': {
            'success': bool(sol.success), 'status': int(sol.status),
            'message': str(sol.message), 'nfev': int(sol.nfev),
            'cost_robust_weighted': float(sol.cost),
            'delta_rotvec_camera_deg': (np.rad2deg(sol.x[:3]).tolist()
                                        if mode == 'optimize_rotation' else [0., 0., 0.]),
            'delta_translation_camera_m': (sol.x if mode == 'fixed_rotation'
                                           else sol.x[3:]).tolist(),
            'bound_hit': bound_hit.tolist(),
            'any_bound_hit': bool(np.any(bound_hit)),
            'handeye_motion_residual_norm_initial': float(np.linalg.norm(motion(zero))),
            'handeye_motion_residual_norm_final': float(np.linalg.norm(motion(sol.x))),
            'ground_residual_norm_initial': float(np.linalg.norm(ground(zero))),
            'ground_residual_norm_final': float(np.linalg.norm(ground(sol.x))),
            'max_initial_ground_normal_angle_deg': float(max(v['initial_normal_angle_deg']
                                                            for v in validation)),
            'plane_count': len(matches),
            'motion_pair_count': int(pairs.count),
        },
        'ground_initial': ground_metrics(matches, signs, initial[:3, :3], initial[:3, 3]),
        'ground_final': ground_metrics(matches, signs, estimated[:3, :3], estimated[:3, 3]),
        'correspondence_validation': validation,
        'jacobian_ground_only': _svd_report(ground, sol.x, steps),
        'jacobian_joint': _svd_report(combined, sol.x, steps),
        'notes': [
            'Hand-Eye motion residual is retained, not a parameter prior.',
            'Both modes use the SAME motion pairs and ground correspondences.',
            'Ground normals affect optimized rotation only in optimize_rotation mode.',
            'Monocular SfM scale is held fixed in both modes.',
            'Relative weights and sigmas are heuristics, not sensor-calibrated noise.',
            'Ground normals/signs are oriented with initial extrinsic; no GT used.',
            'An optimizer success or plane-fit improvement is not GT extrinsic improvement.',
        ],
    }


def compare_modes(initial_T_camera_lidar: NDArray, sfm_scale: float,
                  pairs: MotionPairs, ground_matches: Sequence[GroundMatch],
                  *, settings: GroundRefinementSettings = GroundRefinementSettings()
                  ) -> dict[str, Any]:
    """Run both ablation arms with identical inputs and loss weights."""
    return {
        'schema_version': 1,
        'kind': 'joint_handeye_ground_refinement',
        'baseline': {'T_camera_lidar': _validate_transform(initial_T_camera_lidar).tolist(),
                     'sfm_scale_to_metric': float(sfm_scale)},
        'fixed_rotation': refine_with_ground(initial_T_camera_lidar, sfm_scale, pairs,
                        ground_matches, mode='fixed_rotation', settings=settings),
        'optimize_rotation': refine_with_ground(initial_T_camera_lidar, sfm_scale, pairs,
                        ground_matches, mode='optimize_rotation', settings=settings),
        'settings': {
            name: getattr(settings, name) for name in settings.__dataclass_fields__
        },
    }
