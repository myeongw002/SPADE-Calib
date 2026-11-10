"""Full-coupling Hand-Eye local observability, model-based covariance and block bootstrap.

Model-based covariance assumes independent residuals with explicitly supplied
rotation and translation noise scales. Because pair observations overlap in time,
this is a CONDITIONAL proxy, not an empirically calibrated physical covariance.
"""
from __future__ import annotations

from typing import Any
import math
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from spade_calib.selection.trajectory import MotionPairs
from .residuals import tangent_residuals
from .solver import HandEyeResult, fit_motion_pairs

_AXES = ("rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z", "log_sfm_scale")


def _frame_noise_jacobian(result: HandEyeResult) -> NDArray[np.float64]:
    settings = result.settings
    def fn(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return tangent_residuals(x, result.pairs, result.rotation_lidar_camera,
                                 result.translation_lidar_camera_m, result.sfm_scale_to_metric,
                                 math.radians(settings.sigma_relative_rotation_deg),
                                 settings.sigma_relative_translation_m)
    # Centered difference in physical coordinates. 0.5*step is held in both signs,
    # with parameter-specific steps to avoid cancellation / numerical instability.
    steps = np.array([1e-6] * 3 + [1e-5] * 3 + [1e-6], dtype=np.float64)
    cols = []
    for k, step in enumerate(steps):
        d = np.zeros(7)
        d[k] = step
        cols.append((fn(d) - fn(-d)) / (2 * step))
    return np.column_stack(cols)


def _coordinate_scales(result: HandEyeResult) -> NDArray[np.float64]:
    setting = result.settings
    return np.array([math.radians(setting.reference_rotation_deg)] * 3
                    + [setting.reference_translation_m] * 3
                    + [setting.reference_log_scale], dtype=np.float64)


def _eigendirections(axes: tuple[str, ...], vector: NDArray[np.float64]) -> list[dict[str, Any]]:
    return [{"coordinate": axes[i], "weight": float(vector[i])}
            for i in np.argsort(-np.abs(vector))[:min(4, len(vector))]]


def _std_details(covariance: NDArray[np.float64]) -> dict[str, Any]:
    d = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    diagonal = np.outer(d, d)
    cor = np.divide(covariance, diagonal, out=np.zeros_like(covariance), where=diagonal > 0)
    return {
        "standard_deviations": {
            "rotation_lidar_axis_deg": np.rad2deg(d[:3]).tolist(),
            "translation_lidar_axis_m": d[3:6].tolist(),
            "log_sfm_scale": float(d[6]) if len(d) > 6 else None,
        },
        "correlation_matrix": cor.tolist(),
    }


def analyze_local_uncertainty(result: HandEyeResult) -> dict[str, Any]:
    """Rank and weak 7-DoF modes; conditional covariance if FULL rank only.

    Uses the *joint* Jacobian, including rotation/translation/log-scale coupling.
    Marginal X covariance is the first 6x6 block of inverse joint information,
    not inverse of the 6x6 information subblock.
    """
    J = _frame_noise_jacobian(result)
    q = _coordinate_scales(result)
    Jn = J * q[None, :]
    _, sing, vt = np.linalg.svd(Jn, full_matrices=False)
    threshold_rank = result.settings.rank_sv_ratio * float(sing[0])
    rank = int(np.count_nonzero(sing > threshold_rank))
    weak = [i for i in range(7) if sing[i] < sing[0] * result.settings.weak_sv_ratio]
    info = J.T @ J
    out: dict[str, Any] = {
        "interpretation": "Assumed-noise local Jacobian; NOT validated sensor uncertainty",
        "parameters": list(_AXES),
        "coordinate_reference_units": q.tolist(),
        "jacobian_shape": list(J.shape),
        "normalized_singular_values": sing.tolist(),
        "normalized_condition_number": float(sing[0] / sing[-1]) if sing[-1] > 0 else None,
        "rank": rank,
        "full_rank": bool(rank == 7),
        "rank_relative_threshold": result.settings.rank_sv_ratio,
        "weak_relative_threshold": result.settings.weak_sv_ratio,
        "weak_modes": [{
            "relative_strength": float(sing[k] / sing[0]),
            "normalised_parameter_direction": dict(zip(_AXES, vt[k].tolist())),
            "largest_components": _eigendirections(_AXES, vt[k]),
        } for k in weak],
        "information_matrix_7x7": info.tolist(),
        "conditional_assumptions": [
            "Relative-motion rotation/translation sigma supplied in HandEyeSettings are assumptions",
            "Pairs sharing poses are correlated, ignored by the local covariance formula",
            "Jacobian treats input GLIM/COLMAP trajectories and the pair construction as fixed",
            "Robust soft_l1/huber influence is not represented by a statistically derived sandwich covariance",
        ],
    }
    if rank != 7:
        out["covariance_available"] = False
        out["covariance_reason"] = "Rank deficient; no finite 7-DoF joint covariance without extra constraints"
        return out
    # inv(J.T J) may be extremely ill conditioned: solve via SVD in normalized units.
    cov_normalized = (vt.T / (sing * sing)) @ vt
    cov_joint = (q[:, None] * cov_normalized) * q[None, :]
    cov_joint = (cov_joint + cov_joint.T) * 0.5
    cov_X = cov_joint[:6, :6]  # correctly marginalized over uncertain SfM scale
    cov_full_std = _std_details(cov_joint)
    ex_std = _std_details(cov_X)
    # Directions of largest uncertainty in dimensionless scaled units:
    # cov_X_normalized = inv(D_6) cov_X inv(D_6).
    cov_x_norm = cov_X / q[:6, None] / q[None, :6]
    eigen, evectors = np.linalg.eigh(cov_x_norm)
    k = int(np.argmax(eigen))
    out.update({
        "covariance_available": True,
        "model_based_joint_covariance_7x7": cov_joint.tolist(),
        "model_based_extrinsic_marginal_covariance_6x6": cov_X.tolist(),
        "model_based_extrinsic_standard_deviations": ex_std["standard_deviations"],
        "model_based_joint_standard_deviations": cov_full_std["standard_deviations"],
        "model_based_extrinsic_correlation_matrix_6x6": ex_std["correlation_matrix"],
        "largest_extrinsic_uncertainty_eigenvalue_normalized": float(eigen[k]),
        "largest_extrinsic_uncertainty_direction_normalized": dict(zip(_AXES[:6], evectors[:, k].tolist())),
        "largest_extrinsic_uncertainty_components": _eigendirections(_AXES[:6], evectors[:, k]),
    })
    return out


def _slice_pairs(pairs: MotionPairs, indices: NDArray[np.int64]) -> MotionPairs:
    return MotionPairs(pairs.Ra[indices], pairs.ta[indices],
                       pairs.Rb[indices], pairs.tb[indices],
                       pairs.indices[indices], pairs.starts_ns[indices])


def block_bootstrap_uncertainty(result: HandEyeResult, *, n_resamples: int = 64,
                                block_duration_s: float = 20.0, seed: int = 20261110,
                                min_valid_fraction: float = 0.75) -> dict[str, Any]:
    """Refit correlated motion pairs by sampling contiguous temporal blocks.

    Does not simulate GLIM/SfM per-pose correlated estimation errors, and
    therefore estimates recording subset stability rather than physical 1-sigma.
    It may not detect common bias. Missing unstable fits are never silently dropped.
    """
    if type(n_resamples) is not int or n_resamples < 3:
        raise ValueError("n_resamples must be integer >=3")
    if not math.isfinite(block_duration_s) or block_duration_s <= 0:
        raise ValueError("block_duration_s must be finite and positive")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0,1]")
    p = result.pairs
    block_ns = max(1, round(block_duration_s * 1e9))
    block_ids = (p.starts_ns - int(p.starts_ns[0])) // block_ns
    blocks = [np.flatnonzero(block_ids == x) for x in np.unique(block_ids)]
    if len(blocks) < 3:
        return {"available": False, "reason": "Less than 3 temporal blocks", "temporal_blocks": len(blocks)}
    rng = np.random.default_rng(seed)
    params = []
    failed = 0
    for _ in range(n_resamples):
        picks = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[int(i)] for i in picks]).astype(np.int64)
        sample = _slice_pairs(p, idx)
        try:
            fit = fit_motion_pairs(sample, result.settings,
                                   initial_rotation=result.rotation_lidar_camera,
                                   initial_translation_m=result.translation_lidar_camera_m,
                                   initial_scale=result.sfm_scale_to_metric)
            # Reject underexcited resamples rather than treating a solver's
            # arbitrary gauge choice as a valid covariance realization.
            if not analyze_local_uncertainty(fit)["full_rank"]:
                failed += 1
                continue
            dr = Rotation.from_matrix(fit.rotation_lidar_camera @ result.rotation_lidar_camera.T).as_rotvec()
            params.append(np.r_[dr, fit.translation_lidar_camera_m - result.translation_lidar_camera_m,
                                math.log(fit.sfm_scale_to_metric / result.sfm_scale_to_metric)])
        except (RuntimeError, ValueError, OverflowError, np.linalg.LinAlgError):
            failed += 1
    valid = len(params)
    report: dict[str, Any] = {
        "method": "resample_contiguous_motion_pair_start_time_blocks_with_replacement",
        "requested_resamples": n_resamples, "valid_resamples": valid,
        "failed_resamples": failed, "temporal_blocks": len(blocks),
        "block_duration_s": block_duration_s, "seed": seed,
        "assumptions": [
            "Resampled blocks retain overlapping motion pairs and their within-block correlation",
            "Common GLIM/SfM systematic bias is NOT captured",
            "This is temporal subset stability, not an independently validated sensor covariance",
        ],
    }
    if valid < max(3, math.ceil(n_resamples * min_valid_fraction)):
        return {**report, "available": False,
                "reason": "Insufficient valid bootstrap fits"}
    sample = np.stack(params)
    cov = np.cov(sample, rowvar=False, ddof=1)
    if cov.shape != (7, 7) or not np.isfinite(cov).all():
        return {**report, "available": False, "reason": "Invalid bootstrap covariance"}
    q = _coordinate_scales(result)
    e,v = np.linalg.eigh(cov[:6, :6] / q[:6, None] / q[None, :6])
    return {
        **report, "available": True,
        "empirical_joint_covariance_7x7": cov.tolist(),
        "empirical_extrinsic_covariance_6x6": cov[:6, :6].tolist(),
        "empirical_extrinsic_standard_deviations": _std_details(cov[:6, :6])["standard_deviations"],
        "empirical_scale_log_std": float(math.sqrt(max(0, cov[6, 6]))),
        "empirical_extrinsic_correlation_matrix_6x6": _std_details(cov[:6, :6])["correlation_matrix"],
        "largest_empirical_uncertainty_direction_normalized": dict(zip(_AXES[:6], v[:, -1].tolist())),
        "largest_empirical_uncertainty_variance_normalized": float(e[-1]),
    }