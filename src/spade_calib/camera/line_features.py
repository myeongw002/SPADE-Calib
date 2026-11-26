"""E1 camera-only line geometry and relative inverse-depth support.

No LiDAR, extrinsic, trajectory, metric depth or physical identity assumption.
The normal matrix is an image geometry proxy, not calibration information.
"""
from __future__ import annotations

import numpy as np
import cv2


def line_geometry(segments, K):
    s = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    K = np.asarray(K, dtype=float)
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or not np.allclose(K[2], [0, 0, 1])
            or K[0, 0] <= 0 or K[1, 1] <= 0 or abs(np.linalg.det(K)) < 1e-12):
        raise ValueError("Invalid rectified camera intrinsic matrix")
    if not np.isfinite(s).all():
        raise ValueError("Nonfinite segments")
    length = np.linalg.norm(s[:, 1] - s[:, 0], axis=1)
    if np.any(length <= 0):
        raise ValueError("Degenerate image line")
    homogeneous = np.concatenate((s, np.ones((len(s), 2, 1))), axis=2)
    lines = np.cross(homogeneous[:, 0], homogeneous[:, 1])
    lines /= np.linalg.norm(lines[:, :2], axis=1)[:, None]
    normals = lines @ K
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return length, normals


def depth_profiles(depth, segments, *, offset_min=3, offset_max=9,
                   stations=17, min_valid=5, range_floor=.03):
    """Return [jump, crease, support, valid_fraction] per original RGB line.

    Fit depth proxy vs outward pixel distance separately on each side.
    Jump = intercept mismatch; crease = sum of outward slopes times span.
    Sign-coherent median over stations, normalized by local observed range
    with a global robust range floor. No metric surface normals inferred.
    Invalid support is NaN, not a negative observation.
    """
    D = np.asarray(depth, dtype=np.float32)
    s = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    if D.ndim != 2 or min(D.shape) < 2 or not np.isfinite(D).all():
        raise ValueError("Expected finite original-resolution HxW depth")
    if not np.isfinite(s).all() or np.any(np.linalg.norm(s[:, 1]-s[:, 0], axis=1) <= 0):
        raise ValueError("Invalid segments")
    if not (1 <= offset_min < offset_max and 1 <= min_valid <= stations
            and 0 < range_floor <= 1):
        raise ValueError("Invalid profile settings")
    scale = float(np.percentile(D, 95) - np.percentile(D, 5))
    if scale <= 1e-8:
        raise ValueError("Relative depth has no robust dynamic range")
    h, w = D.shape
    offsets = np.arange(offset_min, offset_max + 1, dtype=float)
    P = np.linalg.pinv(np.column_stack((np.ones(len(offsets)), offsets)))
    out = np.full((len(s), 4), np.nan)
    for i, p in enumerate(s):
        v = p[1] - p[0]
        v /= np.linalg.norm(v)
        n = np.array([-v[1], v[0]])
        centers = p[0] + np.linspace(.1, .9, stations)[:, None] * (p[1] - p[0])
        pos = centers[:, None] + offsets[None, :, None] * n
        neg = centers[:, None] - offsets[None, :, None] * n
        ok = np.all((pos[..., 0] >= 0) & (pos[..., 0] <= w-1)
                    & (pos[..., 1] >= 0) & (pos[..., 1] <= h-1)
                    & (neg[..., 0] >= 0) & (neg[..., 0] <= w-1)
                    & (neg[..., 1] >= 0) & (neg[..., 1] <= h-1), axis=1)
        out[i, 3] = ok.mean()
        if ok.sum() < min_valid:
            continue

        def sample(x):
            return cv2.remap(D, x[..., 0].astype("float32"),
                             x[..., 1].astype("float32"), cv2.INTER_LINEAR)

        a, b = sample(pos[ok]), sample(neg[ok])
        ca, cb = a @ P.T, b @ P.T
        rng = np.maximum(np.ptp(np.column_stack((a, b)), axis=1), range_floor * scale)
        jump = abs(np.median((ca[:, 0] - cb[:, 0]) / rng))
        crease = abs(np.median((offset_max-offset_min) * (ca[:, 1] + cb[:, 1]) / rng))
        out[i, :3] = jump, crease, min(1., max(jump, crease)) * ok.mean()
    return out


def normal_blocks(normals):
    a = np.asarray(normals, dtype=float)[:, [0, 2]]
    return a[:, :, None] * a[:, None, :]


def greedy_diversity(blocks, budget, weights=None):
    """Greedy E-optimal normal diversity, tie-break by max eigenvalue then ID."""
    M = np.asarray(blocks, dtype=float)
    w = np.ones(len(M)) if weights is None else np.asarray(weights, dtype=float)
    if M.shape != (len(M), 2, 2) or w.shape != (len(M),) or budget < 0:
        raise ValueError("Invalid diversity input")
    if not np.isfinite(M).all() or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Nonfinite or negative diversity weights")
    G = np.zeros((2, 2))
    ids = np.arange(len(M))
    chosen = []
    for _ in range(min(budget, len(M))):
        candidates = G + M[ids] * w[ids, None, None]
        eig = np.linalg.eigvalsh(candidates)
        j = np.lexsort((ids, -eig[:, 1], -eig[:, 0]))[0]
        chosen.append(ids[j])
        G = candidates[j]
        ids = np.delete(ids, j)
    return np.array(chosen, dtype=int)


def select_arms(length, normals, budget, seed, profiles=None):
    """Equal budget and identical pool for all ranked arms.

    With depth enabled, all ranked arms use the same valid-profile pool.
    'all' retains every eligible ELSED line as the descriptive reference.
    """
    length = np.asarray(length, dtype=float)
    M = normal_blocks(normals)
    if budget < 1 or len(M) != len(length):
        raise ValueError("Invalid budget or lengths")
    pool = np.arange(len(length))
    if profiles is not None:
        profiles = np.asarray(profiles, dtype=float)
        if profiles.shape != (len(length), 4):
            raise ValueError("Invalid profile shape")
        pool = pool[np.isfinite(profiles[:, :3]).all(axis=1)]
    k = min(budget, len(pool))

    def top(values):
        return pool[np.argsort(-values[pool], kind="stable")[:k]]

    arms = dict(all=np.arange(len(length)),
                length=top(length),
                random=np.random.default_rng(seed).permutation(pool)[:k],
                normal=pool[greedy_diversity(M[pool], k)])
    if profiles is not None:
        arms.update(jump=top(profiles[:, 0] * profiles[:, 3]),
                    crease=top(profiles[:, 1] * profiles[:, 3]),
                    depth=top(profiles[:, 2]),
                    fusion=pool[greedy_diversity(
                        M[pool], k, .25 + .75 * profiles[pool, 2])])
    return arms, pool


def geometry_metrics(segments, normals, shape):
    s = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    if len(s) == 0:
        return dict(count=0, normal_eigenvalues=None, grid_occupancy=0)
    G = normal_blocks(normals).mean(axis=0)
    h, w = shape
    mid = s.mean(axis=1)
    cells = np.clip(np.floor(mid / np.array([w, h]) * [6, 3]), 0, [5, 2]).astype(int)
    return dict(count=len(s), normal_eigenvalues=np.linalg.eigvalsh(G).tolist(),
                grid_occupancy=len({tuple(row) for row in cells}),
                mean_abs_normal=np.abs(normals).mean(axis=0).tolist())
