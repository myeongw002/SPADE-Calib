"""Joint 7-DoF Hand-Eye and SfM-point-to-LiDAR-plane refinement.

Point residual: n_L.T @ R_CL.T @ (s * p_C_sfm - t_CL) + d_L.
The SfM camera coordinates are UNSCALED; LiDAR planes are METRIC in the
same target LiDAR sensor frame as the camera observation. No GT is consulted.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from spade_calib.correspondence.sfm_lidar_planes import SfMPlaneMatches
from spade_calib.handeye.residuals import residuals as handeye_residuals
from spade_calib.selection.trajectory import MotionPairs


@dataclass(frozen=True)
class SfMPlaneRefinementSettings:
    sigma_motion_rotation_deg: float = 0.5
    sigma_motion_translation_m: float = 0.05
    sigma_point_plane_m: float = 0.20
    motion_weight: float = 1.0
    plane_weight: float = 1.0
    max_rotation_delta_deg: float = 10.0
    max_translation_delta_m: float = 2.0
    max_log_scale_delta: float = 0.35
    robust_loss: str = 'soft_l1'
    max_nfev: int = 300

    def __post_init__(self) -> None:
        positive = ('sigma_motion_rotation_deg', 'sigma_motion_translation_m',
                    'sigma_point_plane_m', 'motion_weight', 'plane_weight',
                    'max_rotation_delta_deg', 'max_translation_delta_m',
                    'max_log_scale_delta')
        if any(not math.isfinite(getattr(self, k)) or getattr(self, k) <= 0
               for k in positive):
            raise ValueError('Noise scales, weights and bounds must be positive and finite')
        if self.max_rotation_delta_deg >= 90 or self.robust_loss not in (
                'linear', 'soft_l1', 'huber', 'cauchy') or self.max_nfev < 1:
            raise ValueError('Invalid solver settings')


def point_plane_residual(matches: SfMPlaneMatches, R_CL: np.ndarray,
                         t_CL: np.ndarray, scale: float) -> np.ndarray:
    """Signed metric distance to associated LiDAR plane, one per unique SfM point."""
    p_l = (scale * matches.camera_xyz_sfm - t_CL) @ R_CL
    return np.sum(matches.normal_lidar * p_l, axis=1) + matches.offset_lidar_m


def refine_joint_7dof(initial_T_camera_lidar: np.ndarray, initial_sfm_scale: float,
                      motion_pairs: MotionPairs, matches: SfMPlaneMatches,
                      settings: SfMPlaneRefinementSettings = SfMPlaneRefinementSettings()
                      ) -> dict:
    """Optimize a fixed, explicitly accepted correspondence set.

    Does not reassign correspondences inside least-squares; generated matches
    are provisional and should be reviewed on real data.
    """
    matches.validate()
    if motion_pairs.count < 3:
        raise ValueError('At least three Hand-Eye motion pairs are required')
    T0 = np.asarray(initial_T_camera_lidar, dtype=np.float64)
    if (T0.shape != (4, 4) or not np.isfinite(T0).all() or
            not np.allclose(T0[3], [0, 0, 0, 1], atol=1e-8) or
            not np.allclose(T0[:3, :3].T @ T0[:3, :3], np.eye(3), atol=1e-5) or
            abs(np.linalg.det(T0[:3, :3])-1) > 1e-5 or
            not np.isfinite(initial_sfm_scale) or initial_sfm_scale <= 0):
        raise ValueError('Invalid initial Hand-Eye transform/scale')
    cfg = settings
    group, counts = np.unique(matches.patch_id, return_counts=True)
    if len(group) < 1:
        raise ValueError('No LiDAR plane groups')
    per_group = dict(zip(group.tolist(), counts.tolist()))
    # Keep a dense ground patch from drowning out the motion constraints.
    weights = np.array([np.sqrt(cfg.plane_weight /
                                (len(group) * per_group[int(i)]))
                        for i in matches.patch_id], dtype=float)
    weights /= cfg.sigma_point_plane_m
    motion_factor = np.sqrt(cfg.motion_weight / motion_pairs.count)

    def state(delta: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        R = Rotation.from_rotvec(delta[:3]).as_matrix() @ T0[:3, :3]
        t = T0[:3, 3] + delta[3:6]
        s = initial_sfm_scale * np.exp(delta[6])
        return R, t, s

    def motion(delta: np.ndarray) -> np.ndarray:
        R, t, s = state(delta)
        err = handeye_residuals(motion_pairs, R, t, s)
        err[:, :3] /= np.deg2rad(cfg.sigma_motion_rotation_deg)
        err[:, 3:] /= cfg.sigma_motion_translation_m
        return motion_factor * err.ravel()

    def planes(delta: np.ndarray) -> np.ndarray:
        R, t, s = state(delta)
        return weights * point_plane_residual(matches, R, t, s)

    def combined(delta: np.ndarray) -> np.ndarray:
        return np.r_[motion(delta), planes(delta)]

    rotation_bound = np.deg2rad(cfg.max_rotation_delta_deg)
    bound = np.array([rotation_bound]*3 + [cfg.max_translation_delta_m]*3 +
                     [cfg.max_log_scale_delta], dtype=float)
    zero = np.zeros(7, dtype=float)
    initial_motion = motion(zero)
    initial_planes = planes(zero)
    solution = least_squares(combined, zero, bounds=(-bound, bound),
                             loss=cfg.robust_loss, x_scale='jac',
                             max_nfev=cfg.max_nfev,
                             ftol=1e-10, xtol=1e-10, gtol=1e-10)
    R, t, s = state(solution.x)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    final_motion = motion(solution.x)
    final_planes = planes(solution.x)
    # Robust-loss-adjusted solver Jacobian is a local sensitivity proxy.
    scale_units = np.array([np.deg2rad(1.)]*3 + [0.1]*3 + [0.1])
    singular = np.linalg.svd(solution.jac * scale_units[None, :], compute_uv=False)
    rank = int(np.count_nonzero(singular > max(singular[0], 1e-15) * 1e-8))
    at_bound = np.abs(solution.x) >= bound - 1e-5 * np.maximum(1, bound)
    initial_raw = point_plane_residual(matches, T0[:3, :3], T0[:3, 3],
                                       initial_sfm_scale)
    final_raw = point_plane_residual(matches, R, t, s)
    return {
        'schema_version': 1, 'kind': 'joint_handeye_sfm_point_lidar_plane_provisional',
        'convention': 'T_camera_lidar', 'T_camera_lidar': T.tolist(),
        'sfm_scale_to_metric': float(s),
        'optimizer': {
            'success': bool(solution.success), 'message': str(solution.message),
            'nfev': int(solution.nfev), 'status': int(solution.status),
            'cost_robust_normalized': float(solution.cost),
            'rotation_delta_camera_deg': np.rad2deg(solution.x[:3]).tolist(),
            'translation_delta_camera_m': solution.x[3:6].tolist(),
            'log_sfm_scale_delta': float(solution.x[6]),
            'bound_hit': at_bound.tolist(), 'any_bound_hit': bool(at_bound.any()),
            'motion_pair_count': int(motion_pairs.count),
            'matched_point_count': int(matches.count),
            'distinct_lidar_patch_count': int(len(group)),
            'motion_residual_norm_initial': float(np.linalg.norm(initial_motion)),
            'motion_residual_norm_final': float(np.linalg.norm(final_motion)),
            'plane_residual_norm_initial': float(np.linalg.norm(initial_planes)),
            'plane_residual_norm_final': float(np.linalg.norm(final_planes)),
            'plane_median_abs_m_initial': float(np.median(np.abs(initial_raw))),
            'plane_median_abs_m_final': float(np.median(np.abs(final_raw))),
        },
        'local_observability': {
            'normalized_singular_values': singular.tolist(), 'rank': rank,
            'full_rank': bool(rank == 7),
            'condition_number': (float(singular[0]/singular[-1]) if
                                 singular[-1] > singular[0]*1e-10 else None),
            'coordinate_scales': scale_units.tolist(),
            'warning': 'Assumed noise, correlated SfM tracks/GLIM scans, and provisional '
                       'correspondences prevent interpreting this as calibrated covariance',
        },
        'warnings': [
            'Automatic point-to-plane associations are hypotheses, not GT.',
            'The solver does not estimate SfM 3D-point or LiDAR-plane uncertainty.',
            'Matching and refinement use the same initial geometry; assess on independent data.',
        ],
    }