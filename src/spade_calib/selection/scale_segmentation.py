"""GT-free piecewise scale proposals for synchronized camera/LiDAR trajectories.

Adapted from the KITTI09/10 experimental scripts. Intervals are half-open
indices into MatchedTrajectories, not raw frame numbers or timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from .trajectory import MatchedTrajectories, MotionPairs


@dataclass(frozen=True)
class ScaleSegmentationSettings:
    bin_frames: int = 20
    min_segment_frames: int = 100
    max_segments: int = 8
    min_pairs: int = 3
    min_sfm_baseline: float = 0.08
    min_cosine: float = 0.95
    min_scale: float = 1.0
    max_scale: float = 200.0

    def __post_init__(self) -> None:
        for name in ("bin_frames", "min_segment_frames", "max_segments", "min_pairs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        values = (self.min_sfm_baseline, self.min_cosine, self.min_scale, self.max_scale)
        if not np.isfinite(values).all() or not (
            self.min_sfm_baseline > 0 and -1 <= self.min_cosine < 1
            and 0 < self.min_scale < self.max_scale
        ):
            raise ValueError("Invalid scale gates")


@dataclass(frozen=True)
class ScalePartition:
    """Contiguous nonoverlapping intervals covering [0, n_frames)."""
    boundaries: tuple[int, ...]

    def __post_init__(self) -> None:
        if (len(self.boundaries) < 2 or self.boundaries[0] != 0
                or any(not isinstance(v, (int, np.integer)) for v in self.boundaries)
                or any(b <= a for a, b in zip(self.boundaries, self.boundaries[1:]))):
            raise ValueError("Boundaries must be strictly increasing integers starting at zero")

    @property
    def count(self) -> int:
        return len(self.boundaries) - 1

    @property
    def n_frames(self) -> int:
        return int(self.boundaries[-1])

    @property
    def intervals(self) -> tuple[tuple[int, int], ...]:
        return tuple((int(a), int(b)) for a, b in zip(self.boundaries, self.boundaries[1:]))

    def groups(self, pairs: MotionPairs) -> NDArray[np.int64]:
        ij = validate_pairs(pairs, self.n_frames)
        g = np.searchsorted(self.boundaries[1:], ij[:, 0], side="right").astype(np.int64)
        ends = np.asarray(self.boundaries)[g + 1]
        return np.where(ij[:, 1] < ends, g, -1)


@dataclass(frozen=True)
class ScaleEvidence:
    target_m: NDArray[np.float64]
    scale: NDArray[np.float64]
    cosine: NDArray[np.float64]
    baseline_sfm: NDArray[np.float64]
    valid: NDArray[np.bool_]


def validate_pairs(pairs: MotionPairs, n_frames: int | None = None) -> NDArray[np.int64]:
    ij = np.asarray(pairs.indices)
    if ij.shape != (pairs.count, 2) or ij.dtype.kind not in "iu":
        raise ValueError("MotionPairs.indices must be an integer (N,2) array")
    if (np.any(ij[:, 0] < 0) or np.any(ij[:, 1] <= ij[:, 0])
            or (n_frames is not None and np.any(ij[:, 1] >= n_frames))):
        raise ValueError("Motion pair endpoints outside valid ordered frame range")
    for x, shape in ((pairs.Ra, (pairs.count, 3, 3)), (pairs.Rb, (pairs.count, 3, 3)),
                     (pairs.ta, (pairs.count, 3)), (pairs.tb, (pairs.count, 3))):
        if x.shape != shape or not np.isfinite(x).all():
            raise ValueError("Invalid motion array shape or nonfinite motion")
    return ij


def validate_extrinsic(X: NDArray) -> NDArray[np.float64]:
    X = np.asarray(X, dtype=float)
    if (X.shape != (4, 4) or not np.isfinite(X).all()
            or not np.allclose(X[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(X[:3, :3].T @ X[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(X[:3, :3]), 1., atol=1e-6)):
        raise ValueError("Expected a finite rigid T_camera_lidar")
    return X


def build_frame_lag_pairs(samples: MatchedTrajectories, lag: int = 10) -> MotionPairs:
    """Experimental frame-lag sampling; lag counts matched poses, not raw frames.

    Use existing build_motion_pairs for a fixed physical-time baseline instead.
    """
    n = len(samples.timestamps_ns)
    if type(lag) is not int or not 0 < lag < n:
        raise ValueError("lag must be an integer in [1, number of matched poses)")
    i = np.arange(n - lag, dtype=np.int64)
    j = i + lag
    L0, L1 = samples.lidar_R_world[i], samples.lidar_R_world[j]
    C0, C1 = samples.camera_R_world_sfm[i], samples.camera_R_world_sfm[j]
    Ra = np.einsum("nji,njk->nik", L0, L1)
    Rb = np.einsum("nji,njk->nik", C0, C1)
    ta = np.einsum("nji,nj->ni", L0, samples.lidar_t_world_m[j] - samples.lidar_t_world_m[i])
    tb = np.einsum("nji,nj->ni", C0, samples.camera_t_world_sfm[j] - samples.camera_t_world_sfm[i])
    return MotionPairs(Ra, ta, Rb, tb, np.column_stack((i, j)), samples.timestamps_ns[i])


def estimate_pair_scales(pairs: MotionPairs, initial_X: NDArray,
                         settings: ScaleSegmentationSettings = ScaleSegmentationSettings()
                         ) -> ScaleEvidence:
    """Compute c=R_X t_A-(R_B-I)t_X and s=(t_B dot c)/||t_B||²."""
    validate_pairs(pairs)
    X = validate_extrinsic(initial_X)
    c = pairs.ta @ X[:3, :3].T - np.einsum("nij,j->ni", pairs.Rb - np.eye(3), X[:3, 3])
    sq = np.einsum("ni,ni->n", pairs.tb, pairs.tb)
    dot = np.einsum("ni,ni->n", pairs.tb, c)
    baseline = np.sqrt(sq)
    denom = baseline * np.linalg.norm(c, axis=1)
    s = np.divide(dot, sq, out=np.full(pairs.count, np.nan), where=sq > 1e-16)
    cos = np.divide(dot, denom, out=np.full(pairs.count, np.nan), where=denom > 1e-16)
    valid = ((baseline > settings.min_sfm_baseline) & (cos > settings.min_cosine)
             & (s > settings.min_scale) & (s < settings.max_scale))
    return ScaleEvidence(c, s, cos, baseline, valid)


def fixed_partition(n_frames: int, k: int) -> ScalePartition:
    if type(n_frames) is not int or type(k) is not int or not 1 <= k <= n_frames:
        raise ValueError("Require integer 1 <= k <= n_frames")
    return ScalePartition(tuple(np.rint(np.linspace(0, n_frames, k + 1)).astype(int)))


def shared_pair_mask(pairs: MotionPairs, evidence: ScaleEvidence,
                     *partitions: ScalePartition) -> NDArray[np.bool_]:
    """Identical eligible pair intersection for fair global/fixed/adaptive fits."""
    if evidence.valid.shape != (pairs.count,):
        raise ValueError("Evidence does not match motion pairs")
    mask = evidence.valid.copy()
    for partition in partitions:
        mask &= partition.groups(pairs) >= 0
    return mask


def adaptive_partition(pairs: MotionPairs, n_frames: int, initial_X: NDArray,
                       settings: ScaleSegmentationSettings = ScaleSegmentationSettings()
                       ) -> tuple[ScalePartition, list[dict]]:
    """DP over start-frame bins; pseudo-BIC is a heuristic, NOT statistical BIC.

    Preserve the experimental start-bin SSE criterion on a common gated set.
    Also require enough strictly within-segment pairs for every candidate.
    The final fit excludes crossing pairs; its objective is therefore distinct.
    """
    ij = validate_pairs(pairs, n_frames)
    evidence = estimate_pair_scales(pairs, initial_X, settings)
    if type(n_frames) is not int or n_frames < settings.min_segment_frames:
        raise ValueError("Not enough matched poses for minimum segment length")
    valid = evidence.valid
    N = int(valid.sum())
    if N < settings.min_pairs:
        raise ValueError("Too few valid scale observations")
    cuts = np.r_[np.arange(0, n_frames, settings.bin_frames), n_frames]
    B = len(cuts) - 1
    starts = np.searchsorted(cuts[1:], ij[valid, 0], side="right")
    ends = np.searchsorted(cuts[1:], ij[valid, 1], side="right")
    tb, c = pairs.tb[valid], evidence.target_m[valid]
    stats = np.column_stack((np.sum(tb * tb, axis=1), np.sum(tb * c, axis=1),
                             np.sum(c * c, axis=1)))
    bins = np.zeros((B, 3))
    np.add.at(bins, starts, stats)
    prefix = np.vstack((np.zeros((1, 3)), np.cumsum(bins, axis=0)))
    # 2D integral histogram counts pairs with both endpoints in bin interval.
    hist = np.zeros((B, B), dtype=np.int64)
    np.add.at(hist, (starts, ends), 1)
    integral = np.pad(hist, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    costs = np.full((B + 1, B + 1), np.inf)
    for a in range(B):
        for b in range(a + 1, B + 1):
            if cuts[b] - cuts[a] < settings.min_segment_frames:
                continue
            count = integral[b, b] - integral[a, b] - integral[b, a] + integral[a, a]
            W, V, C = prefix[b] - prefix[a]
            if count >= settings.min_pairs and W > 1e-12 and V > 0:
                costs[a, b] = max(0., float(C - V * V / W))
    max_k = min(settings.max_segments, n_frames // settings.min_segment_frames)
    dp = np.full((max_k + 1, B + 1), np.inf)
    prev = np.full((max_k + 1, B + 1), -1, dtype=int)
    dp[0, 0] = 0.
    candidates = []
    for k in range(1, max_k + 1):
        for b in range(1, B + 1):
            options = dp[k - 1, :b] + costs[:b, b]
            a = int(np.argmin(options))
            if np.isfinite(options[a]):
                dp[k, b], prev[k, b] = options[a], a
        if not np.isfinite(dp[k, B]):
            continue
        last, boundaries = B, [n_frames]
        for step in range(k, 0, -1):
            last = prev[step, last]
            boundaries.append(int(cuts[last]))
        boundaries.reverse()
        sse = float(dp[k, B])
        score = float(3 * N * np.log(max(sse, 1e-12) / (3 * N)) + k * np.log(3 * N))
        candidates.append({"k": k, "boundaries": boundaries, "start_bin_sse": sse,
                           "pseudo_bic": score})
    if not candidates:
        raise ValueError("No feasible scale partition with enough within-segment pairs")
    best = min(candidates, key=lambda x: x["pseudo_bic"])
    return ScalePartition(tuple(best["boundaries"])), candidates
