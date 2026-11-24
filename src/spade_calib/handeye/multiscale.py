"""Shared T_camera_lidar with one positive SfM scale per temporal segment."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from spade_calib.selection.trajectory import MatchedTrajectories, MotionPairs
from spade_calib.selection.scale_segmentation import (
    ScalePartition, ScaleSegmentationSettings, estimate_pair_scales, validate_extrinsic,
    shared_pair_mask,
)
from .residuals import residuals
from .solver import HandEyeSettings


def _subset(pairs: MotionPairs, mask: NDArray[np.bool_]) -> MotionPairs:
    return MotionPairs(pairs.Ra[mask], pairs.ta[mask], pairs.Rb[mask], pairs.tb[mask],
                       pairs.indices[mask], pairs.starts_ns[mask])


@dataclass(frozen=True)
class MultiScaleResult:
    T_camera_lidar: NDArray[np.float64]
    scales: NDArray[np.float64]
    partition: ScalePartition
    pair_mask: NDArray[np.bool_]
    pair_counts: NDArray[np.int64]
    diagnostics: dict[str, Any]
    settings: HandEyeSettings
    segmentation_settings: ScaleSegmentationSettings

    def to_dict(self, samples: MatchedTrajectories | None = None) -> dict[str, Any]:
        """Serializable handoff; endpoints use matched pose indices, not raw frame IDs."""
        if samples is not None and len(samples.timestamps_ns) != self.partition.n_frames:
            raise ValueError("Samples do not match partition length")
        segments = []
        for k, (a, b) in enumerate(self.partition.intervals):
            record = {"segment_id": k, "start_index": a, "stop_index_exclusive": b,
                      "sfm_scale_to_metric": float(self.scales[k]),
                      "motion_pair_count": int(self.pair_counts[k])}
            if samples is not None:
                record.update(start_timestamp_ns=int(samples.timestamps_ns[a]),
                              end_timestamp_ns_inclusive=int(samples.timestamps_ns[b - 1]),
                              image_names=list(samples.image_names[a:b]),
                              timestamps_ns=[int(t) for t in samples.timestamps_ns[a:b]])
            segments.append(record)
        return {"convention": "T_camera_lidar", "scale_units": "m_per_sfm_unit",
                "T_camera_lidar": self.T_camera_lidar.tolist(),
                "T_lidar_camera": np.linalg.inv(self.T_camera_lidar).tolist(),
                "segments": segments, "selected_pair_rows": np.flatnonzero(self.pair_mask).tolist(),
                "settings": asdict(self.settings),
                "segmentation_settings": asdict(self.segmentation_settings),
                "optimization": self.diagnostics,
                "warning": "Relative-motion fit only; no independent physical accuracy guarantee"}


def fit_segmented_motion_pairs(
    pairs: MotionPairs, partition: ScalePartition, initial_X: NDArray,
    settings: HandEyeSettings = HandEyeSettings(),
    segmentation_settings: ScaleSegmentationSettings = ScaleSegmentationSettings(),
    *, pair_mask: NDArray[np.bool_] | None = None,
    initial_scales: NDArray | None = None,
    fix_rotation: bool = False, fix_translation: bool = False,
) -> MultiScaleResult:
    """Fit 6+K unknowns (or fixed R/t ablation) with existing motion residual.

    Gate using the initial X, then keep assignments fixed during optimization.
    A supplied pair_mask is intersected with the gate and boundary mask.
    Two starts (segment OLS and pooled OLS) are compared unless explicit scales
    are supplied. The scale gates are not statistical confidence intervals.
    """
    X0 = validate_extrinsic(initial_X)
    evidence = estimate_pair_scales(pairs, X0, segmentation_settings)
    mask = shared_pair_mask(pairs, evidence, partition)
    if pair_mask is not None:
        provided = np.asarray(pair_mask)
        if provided.shape != mask.shape or provided.dtype != np.bool_:
            raise ValueError("pair_mask must be a boolean array aligned to MotionPairs")
        mask &= provided
    K = partition.count
    groups = partition.groups(pairs)
    sub = [_subset(pairs, mask & (groups == k)) for k in range(K)]
    counts = np.array([p.count for p in sub], dtype=np.int64)
    if np.any(counts < segmentation_settings.min_pairs) or counts.sum() < 7:
        raise ValueError("Insufficient valid within-segment motion pairs")
    def ols(m):
        tb = pairs.tb[m]
        s = float(np.sum(tb * evidence.target_m[m]) / np.sum(tb * tb))
        if not np.isfinite(s) or s <= 0:
            raise ValueError("Nonpositive initial OLS scale")
        return s
    local = np.array([ols(mask & (groups == k)) for k in range(K)])
    starts = [local, np.full(K, ols(mask))]
    if initial_scales is not None:
        supplied = np.asarray(initial_scales, dtype=float)
        if supplied.shape != (K,) or not np.isfinite(supplied).all() or np.any(supplied <= 0):
            raise ValueError("initial_scales must contain K finite positive scales")
        starts = [supplied]

    nr, nt = (0 if fix_rotation else 3), (0 if fix_translation else 3)
    dim = nr + nt + K
    R0, t0 = X0[:3, :3], X0[:3, 3]
    sig_r = np.deg2rad(settings.sigma_relative_rotation_deg)
    sig_t = settings.sigma_relative_translation_m
    # Optimize absolute log-scales with very broad numerical safety bounds,
    # not the evidence gates. Rotation/translation keep the existing chart.
    lower = np.r_[np.full(nr + nt, -np.inf), np.full(K, -30.)]
    upper = np.r_[np.full(nr + nt, np.inf), np.full(K, 30.)]

    def unpack(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix() @ R0 if nr else R0
        t = t0 + x[nr:nr + 3] if nt else t0
        return R, t, np.exp(x[nr + nt:])

    def raw(x):
        R, t, s = unpack(x)
        return np.concatenate([residuals(p, R, t, float(s[k])) for k, p in enumerate(sub)])

    def objective(x):
        r = raw(x)
        r[:, :3] /= sig_r
        r[:, 3:] /= sig_t
        return r.ravel()

    solves = []
    for scales in starts:
        logs = np.log(scales)
        if np.any(logs <= -30) or np.any(logs >= 30):
            raise ValueError("Initial scale outside numerical safety bounds")
        x0 = np.r_[np.zeros(nr + nt), logs]
        solves.append(least_squares(objective, x0, bounds=(lower, upper),
                                   method="trf", loss=settings.robust_loss, x_scale="jac",
                                   max_nfev=settings.max_nfev,
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9))
    successful = [s for s in solves if s.success and np.isfinite(s.cost)]
    if not successful:
        raise RuntimeError("Multi-scale Hand-Eye failed: " + "; ".join(s.message for s in solves))
    sol = min(successful, key=lambda s: s.cost)
    R, t, scales = unpack(sol.x)
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = R, t
    r = raw(sol.x)
    # Unrobustified, scaled local Jacobian for rank diagnostics only.
    units = np.r_[np.full(nr, np.deg2rad(settings.reference_rotation_deg)),
                  np.full(nt, settings.reference_translation_m),
                  np.full(K, settings.reference_log_scale)]
    J = np.empty((r.size, dim))
    for i, unit in enumerate(units):
        dx = np.zeros(dim)
        dx[i] = 1e-5 * unit
        J[:, i] = (objective(sol.x + dx) - objective(sol.x - dx)) / 2e-5
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int(np.count_nonzero(sv > sv[0] * settings.rank_sv_ratio)) if sv.size else 0
    diagnostics = {"converged": True, "nfev": int(sol.nfev), "cost_whitened_robust": float(sol.cost),
                   "attempts": [{"success": bool(s.success), "cost": float(s.cost)} for s in solves],
                   "translation_residual_rms_m": float(np.sqrt(np.mean(np.sum(r[:, 3:]**2, axis=1)))),
                   "rotation_residual_median_deg": float(np.rad2deg(np.median(np.linalg.norm(r[:, :3], axis=1)))),
                   "parameter_count": dim, "normalized_jacobian_rank": rank,
                   "normalized_jacobian_singular_values": sv.tolist(),
                   "full_rank": rank == dim, "scale_at_numerical_bound": bool(np.any(sol.active_mask[nr+nt:])),
                   "rank_is_not_physical_covariance": True, "relative_motion_pairs_may_be_correlated": True,
                   "fix_rotation": fix_rotation, "fix_translation": fix_translation}
    return MultiScaleResult(X, scales, partition, mask, counts, diagnostics,
                            settings, segmentation_settings)
