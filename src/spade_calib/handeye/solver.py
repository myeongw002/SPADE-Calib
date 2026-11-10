"""Whole-trajectory 7-DoF Hand-Eye fitting with SfM scale as a nuisance variable."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import math
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from spade_calib.selection.trajectory import (
    MatchedTrajectories, MotionPairs, SelectionSettings, build_motion_pairs,
    estimate_rotation_and_sfm_scale,
)
from .residuals import residuals, tangent_residuals


@dataclass(frozen=True)
class HandEyeSettings:
    pair_baseline_s: float = 2.0
    pair_stride_s: float = 1.0
    pair_tolerance_s: float = 0.12
    sigma_relative_rotation_deg: float = 0.5
    sigma_relative_translation_m: float = 0.05
    robust_loss: str = "soft_l1"
    max_nfev: int = 200
    # Normalize columns only for observability comparisons, not the estimator.
    reference_rotation_deg: float = 1.0
    reference_translation_m: float = 0.1
    reference_log_scale: float = 0.1
    weak_sv_ratio: float = 1e-3
    rank_sv_ratio: float = 1e-8

    def __post_init__(self) -> None:
        for field in ("pair_baseline_s", "pair_stride_s", "pair_tolerance_s",
                      "sigma_relative_rotation_deg", "sigma_relative_translation_m",
                      "reference_rotation_deg", "reference_translation_m",
                      "reference_log_scale", "weak_sv_ratio", "rank_sv_ratio"):
            v = getattr(self, field)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{field} must be positive and finite")
        if not (self.weak_sv_ratio > self.rank_sv_ratio):
            raise ValueError("weak_sv_ratio must exceed rank_sv_ratio")
        if self.robust_loss not in ("linear", "soft_l1", "huber", "cauchy"):
            raise ValueError("Unsupported scipy least-squares robust loss")
        if type(self.max_nfev) is not int or self.max_nfev < 10:
            raise ValueError("max_nfev must be an integer >= 10")


@dataclass(frozen=True)
class HandEyeResult:
    rotation_lidar_camera: NDArray[np.float64]
    translation_lidar_camera_m: NDArray[np.float64]
    sfm_scale_to_metric: float
    pairs: MotionPairs
    settings: HandEyeSettings
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = self.rotation_lidar_camera
        extrinsic[:3, 3] = self.translation_lidar_camera_m
        inverse = np.eye(4, dtype=np.float64)
        inverse[:3, :3] = self.rotation_lidar_camera.T
        inverse[:3, 3] = -self.rotation_lidar_camera.T @ self.translation_lidar_camera_m
        return {
            "convention": "T_lidar_camera (multiply camera point to lidar frame)",
            "translation_units": "m", "sfm_scale_units": "metric_m_per_sfm_unit",
            "T_lidar_camera": extrinsic.tolist(),
            "T_camera_lidar": inverse.tolist(),
            "sfm_scale_to_metric": self.sfm_scale_to_metric,
            "number_of_motion_pairs": self.pairs.count,
            "settings": asdict(self.settings),
            "optimization": self.diagnostics,
            "warning": "This extrinsic was inferred from relative trajectories, not independently validated ground truth",
        }


def fit_motion_pairs(pairs: MotionPairs, settings: HandEyeSettings,
                     *, initial_rotation: NDArray[np.float64] | None = None,
                     initial_translation_m: NDArray[np.float64] | None = None,
                     initial_scale: float | None = None) -> HandEyeResult:
    """Optimize all seven unknowns, with a fixed initial tangent coordinate chart."""
    if pairs.count < 7:
        raise ValueError("At least seven relative-motion pairs required")
    if initial_rotation is None or initial_translation_m is None or initial_scale is None:
        ri, si, diag0 = estimate_rotation_and_sfm_scale(pairs)
        if initial_rotation is None:
            initial_rotation = ri
        if initial_scale is None:
            initial_scale = si
        if initial_translation_m is None:
            initial_translation_m = np.array(diag0["provisional_t_lidar_camera_m"])
    R0 = np.asarray(initial_rotation, dtype=np.float64)
    t0 = np.asarray(initial_translation_m, dtype=np.float64)
    s0 = float(initial_scale)
    if R0.shape != (3, 3) or t0.shape != (3,) or not np.isfinite(s0) or s0 <= 0:
        raise ValueError("Invalid initial extrinsic / SfM scale")
    if not np.isfinite(R0).all() or not np.isfinite(t0).all():
        raise ValueError("Invalid initial extrinsic / SfM scale")
    sig_rot = math.radians(settings.sigma_relative_rotation_deg)
    sig_t = settings.sigma_relative_translation_m
    initial_unweighted = residuals(pairs, R0, t0, s0)
    solve = least_squares(
        tangent_residuals, np.zeros(7, dtype=np.float64),
        args=(pairs, R0, t0, s0, sig_rot, sig_t),
        method="trf", loss=settings.robust_loss, max_nfev=settings.max_nfev,
        x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10,
    )
    if not solve.success and solve.status <= 0:
        raise RuntimeError(f"Hand-Eye least_squares failed: {solve.message}")
    R = Rotation.from_rotvec(solve.x[:3]).as_matrix() @ R0
    t = t0 + solve.x[3:6]
    s = float(s0 * np.exp(solve.x[6]))
    err = residuals(pairs, R, t, s)
    rot_norm = np.linalg.norm(err[:, :3], axis=1)
    t_norm = np.linalg.norm(err[:, 3:], axis=1)
    rot_init = np.linalg.norm(initial_unweighted[:, :3], axis=1)
    diag = {
        "converged": bool(solve.success), "status": int(solve.status),
        "message": str(solve.message), "nfev": int(solve.nfev),
        "cost_whitened_robust": float(solve.cost),
        "initial_rotation_residual_median_deg": float(np.rad2deg(np.median(rot_init))),
        "rotation_residual_median_deg": float(np.rad2deg(np.median(rot_norm))),
        "rotation_residual_p95_deg": float(np.rad2deg(np.percentile(rot_norm, 95))),
        "translation_residual_median_m": float(np.median(t_norm)),
        "translation_residual_p95_m": float(np.percentile(t_norm, 95)),
        "relative_motion_pairs_may_be_correlated": True,
        "residual_noise_scales_are_assumed": True,
    }
    return HandEyeResult(R, t, s, pairs, settings, diag)


def fit_full_trajectory(samples: MatchedTrajectories,
                        settings: HandEyeSettings = HandEyeSettings()) -> HandEyeResult:
    """Use full synchronized GLIM/COLMAP motion, not selected scene windows."""
    cfg = SelectionSettings(
        pair_baseline_s=settings.pair_baseline_s,
        pair_stride_s=settings.pair_stride_s,
        pair_tolerance_s=settings.pair_tolerance_s,
    )
    pairs = build_motion_pairs(samples, int(samples.timestamps_ns[0]),
                               int(samples.timestamps_ns[-1]), cfg)
    return fit_motion_pairs(pairs, settings)