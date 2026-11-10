"""SE(3) relative-motion residuals for metric GLIM and scale-ambiguous SfM.

Conventions: A_i = inv(T_world_lidar_i) @ T_world_lidar_j;
B_i = inv(T_sfm_world_camera_i) @ T_sfm_world_camera_j.
X = T_lidar_camera, with A X = X B(s).

The six residual coordinates are [rotation-vector in radians, translation in m].
The translation residual is in the LiDAR frame of the first observation.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from spade_calib.selection.trajectory import MotionPairs


def residuals(pairs: MotionPairs, rotation_lidar_camera: NDArray[np.float64],
              translation_lidar_camera_m: NDArray[np.float64], sfm_scale: float) -> NDArray[np.float64]:
    """Return (N, 6) errors for A X = X B(s); no probabilistic weighting."""
    if not np.isfinite(sfm_scale) or sfm_scale <= 0:
        raise ValueError("SfM scale must be positive and finite")
    if pairs.count == 0:
        return np.empty((0, 6), dtype=np.float64)
    rx = np.asarray(rotation_lidar_camera, dtype=np.float64)
    tx = np.asarray(translation_lidar_camera_m, dtype=np.float64)
    if rx.shape != (3, 3) or tx.shape != (3,) or not (np.isfinite(rx).all() and np.isfinite(tx).all()):
        raise ValueError("Rotation must be (3,3) and translation (3,), both finite")
    left = pairs.Ra @ rx
    right = rx @ pairs.Rb
    rotational_error = Rotation.from_matrix(np.swapaxes(left, 1, 2) @ right).as_rotvec()
    translational_error = (
        np.einsum("nij,j->ni", pairs.Ra - np.eye(3), tx)
        + pairs.ta - sfm_scale * np.einsum("ij,nj->ni", rx, pairs.tb)
    )
    return np.column_stack((rotational_error, translational_error))


def tangent_residuals(delta: NDArray[np.float64], pairs: MotionPairs,
                      base_rotation: NDArray[np.float64], base_translation_m: NDArray[np.float64],
                      base_scale: float, sigma_rotation_rad: float,
                      sigma_translation_m: float) -> NDArray[np.float64]:
    """Whitened residual for local left SO(3) perturbation and additive translation.

    delta = [left_tangent_rotvec(3), additive_xyz_m(3), log_scale_shift(1)].
    sigmas are *assumed relative-motion residual scales*, NOT known pose covariances.
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