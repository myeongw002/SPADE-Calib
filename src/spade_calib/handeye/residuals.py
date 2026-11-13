"""SE(3) Hand-Eye relative-motion residuals for T_camera_lidar.

Conventions:
  A_i = inv(T_world_lidar_i) @ T_world_lidar_j  (LiDAR relative motion)
  B_i = inv(T_sfm_world_camera_i) @ T_sfm_world_camera_j
  X = T_camera_lidar, with B(s) X = X A.

The residual coordinates are [left rotation error in camera radians,
translation error in camera metres]. Camera and GLIM world frames are
independent. The monocular scale s is metric metres per SfM unit.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from spade_calib.selection.trajectory import MotionPairs


def residuals(pairs: MotionPairs, rotation_camera_lidar: NDArray[np.float64],
              translation_camera_lidar_m: NDArray[np.float64],
              sfm_scale: float) -> NDArray[np.float64]:
    """Return (N, 6) errors for B(s)X=XA, with no probabilistic weighting."""
    if not np.isfinite(sfm_scale) or sfm_scale <= 0:
        raise ValueError("SfM scale must be positive and finite")
    if pairs.count == 0:
        return np.empty((0, 6), dtype=np.float64)
    rx = np.asarray(rotation_camera_lidar, dtype=np.float64)
    tx = np.asarray(translation_camera_lidar_m, dtype=np.float64)
    if rx.shape != (3, 3) or tx.shape != (3,) or not (np.isfinite(rx).all() and np.isfinite(tx).all()):
        raise ValueError("Rotation must be (3,3) and translation (3,), both finite")
    left = pairs.Rb @ rx
    right = rx @ pairs.Ra
    rotational_error = Rotation.from_matrix(np.swapaxes(left, 1, 2) @ right).as_rotvec()
    translational_error = (
        np.einsum("nij,j->ni", pairs.Rb - np.eye(3), tx)
        + sfm_scale * pairs.tb
        - np.einsum("ij,nj->ni", rx, pairs.ta)
    )
    return np.column_stack((rotational_error, translational_error))


def tangent_residuals(delta: NDArray[np.float64], pairs: MotionPairs,
                      base_rotation: NDArray[np.float64], base_translation_m: NDArray[np.float64],
                      base_scale: float, sigma_rotation_rad: float,
                      sigma_translation_m: float) -> NDArray[np.float64]:
    """Whitened residual with Camera-frame left SO(3) and additive Camera translation.

    delta = [left_camera_rotvec(3), camera_translation_xyz_m(3), log_scale_shift(1)].
    Sigmas are ASSUMED relative-motion residual scales, not pose covariances.
    """
    delta = np.asarray(delta, dtype=np.float64)
    if delta.shape != (7,) or not np.isfinite(delta).all():
        raise ValueError("delta must be a finite 7-vector")
    rot = Rotation.from_rotvec(delta[:3]).as_matrix() @ base_rotation
    translation = base_translation_m + delta[3:6]
    scale = float(base_scale * np.exp(delta[6]))
    r = residuals(pairs, rot, translation, scale)
    r[:, :3] /= sigma_rotation_rad
    r[:, 3:] /= sigma_translation_m
    return r.ravel()
