"""Point-to-Plane pairwise corrections followed by an anchored SE(3)-linearized PGO.

Port of the user's 2026-11-06 retest_pgo.py. Pair correspondences are frozen
for pairwise estimates; 6D correction graph uses small-angle regularization.
This is NOT historical GLIM's internal pose graph optimizer.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr
from .deskew import AccumulatedCloud
from .joint_solver import (JointSettings, ScanOptimizationResult, _validate_cloud,
                           ptransform, sample_model, match_pairs, apply_corrections,
                           correction_matrices)


@dataclass(frozen=True)
class PGOSettings:
    pairs: JointSettings = JointSettings()
    iterations: int = 4
    pair_max_nfev: int = 38

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.pair_max_nfev < 1:
            raise ValueError('Invalid PGO iteration settings')


def pair_estimation(i: int, j: int, p: NDArray, q: NDArray, n: NDArray,
                    center: NDArray, settings: PGOSettings) -> dict | None:
    if len(p) < 90:
        return None

    def residual(x):
        P = ptransform(p, x, center)
        r = np.einsum('ij,ij->i', n, P - q)
        return np.r_[r, .07 * x[:3] / .18, .07 * x[3:] / np.deg2rad(2.0)]

    sol = least_squares(residual, np.zeros(6), loss='soft_l1', f_scale=.055,
                        max_nfev=settings.pair_max_nfev,
                        bounds=([-0.32] * 3 + [-np.deg2rad(3)] * 3,
                                [.32] * 3 + [np.deg2rad(3)] * 3))
    x = sol.x
    r_before = np.einsum('ij,ij->i', n, p - q)
    r_after = np.einsum('ij,ij->i', n, ptransform(p, x, center) - q)
    delta = np.r_[x[:3] - np.cross(x[3:], center), x[3:]]
    cond = np.linalg.svd(np.einsum('ni,nj->ij', n, n), compute_uv=False)
    return {'i': i, 'j': j, 'delta': delta, 'n': int(len(p)),
            'before_cm': np.sqrt(np.mean(r_before * r_before)) * 100,
            'after_cm': np.sqrt(np.mean(r_after * r_after)) * 100,
            'normal_spread_min_eig': float(cond[-1] / max(cond[0], 1e-10))}


def graph_solve(edges: list[dict], nscan: int, anchor: int, settings: PGOSettings) -> NDArray:
    X = np.zeros((nscan, 6), dtype=float)
    ncols = 6 * (nscan - 1)

    def nidx(s):
        return (s if s < anchor else s - 1) * 6

    for _ in range(settings.iterations):
        ar, rhs, r = [], [], 0
        for e in edges:
            i, j = e['i'], e['j']
            delta = e['delta']
            resid = X[i] - X[j] - delta
            base = np.sqrt(e['n'] / 450) / max(e['after_cm'], 4.5)
            base = np.clip(base, .07, .28)
            norm = np.linalg.norm(resid[:3]) / .14 + np.linalg.norm(resid[3:]) / np.deg2rad(1.4)
            w = base / np.sqrt(1 + (norm / 2) ** 2)
            for d in range(6):
                scale = (1 if d < 3 else 8)
                if i != anchor:
                    ar.append((r, nidx(i) + d, w * scale))
                if j != anchor:
                    ar.append((r, nidx(j) + d, -w * scale))
                rhs.append(-w * scale * resid[d])
                r += 1
        for i in range(nscan):
            if i == anchor:
                continue
            for d in range(6):
                w = .045 if d < 3 else .38
                ar.append((r, nidx(i) + d, w))
                rhs.append(-w * X[i, d])
                r += 1
        for i in range(nscan - 1):
            j = i + 1
            for d in range(6):
                w = .025 if d < 3 else .22
                if i != anchor:
                    ar.append((r, nidx(i) + d, w))
                if j != anchor:
                    ar.append((r, nidx(j) + d, -w))
                rhs.append(-w * (X[i, d] - X[j, d]))
                r += 1
        rows, cols, vals = zip(*ar)
        A = coo_matrix((vals, (rows, cols)), shape=(r, ncols)).tocsr()
        sol = lsmr(A, np.array(rhs), atol=1e-9, btol=1e-9, maxiter=500)
        upd = np.zeros_like(X)
        for i in range(nscan):
            if i != anchor:
                upd[i] = sol[0][nidx(i):nidx(i) + 6]
        X += upd
        if np.linalg.norm(upd) < 1e-7:
            break
    return X


def solve_pgo(cloud: AccumulatedCloud, settings: PGOSettings = PGOSettings()) -> ScanOptimizationResult:
    """Run pairwise point-to-plane constrained PGO without any refined reference input."""
    pair_cfg = settings.pairs
    A, scan = _validate_cloud(cloud, pair_cfg.anchor_scan_id)
    indices, centers, normals, oks = sample_model(A, scan, pair_cfg)
    frozen = match_pairs(A, scan, indices, centers, normals, oks,
                         np.zeros((len(indices), 6)), pair_cfg)
    edges = []
    for i, j, p, q, n in frozen:
        e = pair_estimation(i, j, p, q, n, centers[i], settings)
        if e is not None:
            edges.append(e)
    if not edges:
        raise ValueError('PGO has no valid pairwise scan constraints')
    Xworld = graph_solve(edges, len(indices), pair_cfg.anchor_scan_id, settings)
    X = Xworld.copy()
    X[:, :3] += np.cross(Xworld[:, 3:], centers)
    report = {'pair_edges': len(edges), 'matched_returns': int(sum(x['n'] for x in edges)),
              'pair_after_rmse_cm': float(np.sqrt(np.average(
                  [x['after_cm'] ** 2 for x in edges], weights=[x['n'] for x in edges]))),
              'relative_pose_is_small_angle_approximation': True}
    return ScanOptimizationResult('PGO', apply_corrections(A, scan, X, centers),
                                  np.unique(scan), X, centers, correction_matrices(X, centers),
                                  (report,), pair_cfg.anchor_scan_id)