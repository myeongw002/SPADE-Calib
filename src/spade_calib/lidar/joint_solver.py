"""Robust scan-pair point-to-plane *joint* 6-DoF scan pose refinement.

Ported from the user's 2026-11-06 compa_B_C1_experiment.py. The selected
method uses the B (one rigid SE(3) per deskewed scan) estimate, not C1.
All corrections live in target LiDAR coordinates. D1-specific ROI is an
explicit preset, not a universal physical threshold.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix, vstack as svstack, csr_matrix
from scipy.sparse.linalg import lsmr

from .deskew import AccumulatedCloud


@dataclass(frozen=True)
class JointSettings:
    anchor_scan_id: int = 5
    iterations: int = 5
    seed: int = 20261106
    voxel_m: float = 0.14
    max_samples_per_scan: int = 6000
    k_normals: int = 24
    max_correspondences_per_pair: int = 750
    pair_distance_m: float = 0.42
    pair_cos_normal: float = 0.87
    pair_plane_m: float = 0.15
    irls_scale_m: float = 0.07
    # Only D1 legacy reproduction uses these bounds and wall exclusion.
    d1_roi: bool = True

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.max_samples_per_scan < self.k_normals or self.k_normals < 4:
            raise ValueError('Invalid joint iteration/sample/normal settings')
        if min(self.voxel_m, self.pair_distance_m, self.pair_plane_m, self.irls_scale_m) <= 0:
            raise ValueError('Joint thresholds must be positive')


@dataclass(frozen=True)
class ScanOptimizationResult:
    method: str
    xyz: NDArray[np.float32]
    scan_ids: NDArray[np.int64]
    corrections: NDArray[np.float64]  # (S,6) translation[meters] and SO3 rotvec[radians], around centers
    centers: NDArray[np.float64]  # original deskew scan centers in target frame
    transforms: NDArray[np.float64]  # (S,4,4), column-vector convention, target LiDAR -> target LiDAR
    iterations: tuple[dict, ...]
    anchor_scan_id: int


def voxel_sample(p: NDArray, voxel: float = .14) -> NDArray[np.int64]:
    key = np.floor(p / voxel).astype('int32')
    _, ind = np.unique(key, axis=0, return_index=True)
    return np.sort(ind)


def normals_for_scan(p: NDArray, k: int = 24) -> tuple[NDArray, NDArray]:
    if len(p) < k:
        return np.zeros((len(p), 3)), np.zeros(len(p), dtype=bool)
    tree = cKDTree(p)
    dist, near = tree.query(p, k=k, workers=-1)
    q = p[near]
    cen = q.mean(axis=1)
    arr = q - cen[:, None, :]
    cv = np.einsum('nki,nkj->nij', arr, arr) / k
    eig, vec = np.linalg.eigh(cv)
    ratio = eig[:, 0] / np.maximum(eig[:, 1], 1.e-12)
    normal = vec[:, :, 0]
    ok = (ratio < .28) & (dist[:, -1] < 1.6) & (eig[:, 1] > .0003)
    return normal, ok


def ptransform(p: NDArray, x: NDArray, c: NDArray) -> NDArray:
    return (p - c) @ Rotation.from_rotvec(x[3:]).as_matrix().T + c + x[:3]


def sample_model(A: NDArray, scan: NDArray, settings: JointSettings):
    n_scans = int(np.max(scan)) + 1
    idx, centers, normset, okset = [], [], [], []
    for s in range(n_scans):
        original = np.flatnonzero(scan == s)
        if not len(original):
            raise ValueError(f'Scan {s} has no valid LiDAR points')
        p = A[original]
        if settings.d1_roi:
            mask = (np.abs(p[:, 0]) < 22) & (np.abs(p[:, 1]) < 16) & (p[:, 2] > -2) & (p[:, 2] < 5)
            wall = ((p[:, 0] > 10.3) & (p[:, 0] < 14.9)
                    & (p[:, 1] > 4.3) & (p[:, 1] < 9.4)
                    & (p[:, 2] > -1.2) & (p[:, 2] < 4.6))
            mask &= ~wall
        else:
            mask = np.ones(len(p), dtype=bool)
        valid = original[mask]
        sel = valid[voxel_sample(A[valid], settings.voxel_m)]
        if len(sel) > settings.max_samples_per_scan:
            rng = np.random.default_rng(settings.seed + s)
            sel = np.sort(rng.choice(sel, settings.max_samples_per_scan, replace=False))
        q = A[sel]
        n, ok = normals_for_scan(q, k=settings.k_normals)
        centers.append(A[scan == s].mean(axis=0))
        idx.append(sel)
        normset.append(n)
        okset.append(ok)
    return idx, np.asarray(centers), normset, okset


def match_pairs(A: NDArray, scan: NDArray, inds: Sequence[NDArray], centers: NDArray,
                normals: Sequence[NDArray], oks: Sequence[NDArray], X: NDArray,
                settings: JointSettings):
    n = len(inds)
    anchor = settings.anchor_scan_id
    pairset = ({(i, i + 1) for i in range(n - 1)}
               | {(i, i + 2) for i in range(n - 2)}
               | {(min(i, anchor), max(i, anchor)) for i in range(n) if i != anchor})
    matches = []
    for i, j in sorted(pairset):
        ip, jp = inds[i], inds[j]
        p, q = A[ip], A[jp]
        P, Q = ptransform(p, X[i], centers[i]), ptransform(q, X[j], centers[j])
        valid_j, valid_i = np.flatnonzero(oks[j]), np.flatnonzero(oks[i])
        if len(valid_i) < 50 or len(valid_j) < 50:
            continue
        tree = cKDTree(Q[valid_j])
        dist, loc = tree.query(P[valid_i], workers=1)
        nbr = valid_j[loc]
        nj = normals[j][nbr] @ Rotation.from_rotvec(X[j, 3:]).as_matrix().T
        ni = normals[i][valid_i] @ Rotation.from_rotvec(X[i, 3:]).as_matrix().T
        pln = np.abs(np.einsum('ij,ij->i', nj, P[valid_i] - Q[nbr]))
        accept = ((dist < settings.pair_distance_m)
                  & (np.abs(np.einsum('ij,ij->i', ni, nj)) > settings.pair_cos_normal)
                  & (pln < settings.pair_plane_m))
        indices = np.flatnonzero(accept)
        if len(indices) < 65:
            continue
        pi = valid_i[indices]
        chosen = voxel_sample(P[pi], .25)
        indices = indices[chosen]
        if len(indices) > settings.max_correspondences_per_pair:
            rng = np.random.default_rng(settings.seed + i * 100 + j)
            indices = np.sort(rng.choice(indices, settings.max_correspondences_per_pair, replace=False))
        ui, vj = valid_i[indices], nbr[indices]
        matches.append((i, j, A[ip[ui]], A[jp[vj]], normals[j][vj]))
    return matches


def solve_step(matches: Sequence[tuple], centers: NDArray, X: NDArray, settings: JointSettings):
    nscan = len(X)
    anchor = settings.anchor_scan_id
    ncols = (nscan - 1) * 6
    rows, cols, vals, rhs = [], [], [], []
    rnum = 0
    for i, j, p, q, norig in matches:
        Rj = Rotation.from_rotvec(X[j, 3:]).as_matrix()
        normal = norig @ Rj.T
        P, Q = ptransform(p, X[i], centers[i]), ptransform(q, X[j], centers[j])
        resid = np.einsum('ij,ij->i', normal, P - Q)
        w = 1 / np.sqrt(1 + (resid / settings.irls_scale_m) ** 2)
        jac_i = np.hstack([normal, np.cross(P - centers[i] - X[i, :3], normal)])
        jac_j = -np.hstack([normal, np.cross(Q - centers[j] - X[j, :3], normal)])
        N = len(resid)
        indices = np.arange(rnum, rnum + N)
        for s, mat in ((i, jac_i), (j, jac_j)):
            if s == anchor:
                continue
            k = s if s < anchor else s - 1
            for d in range(6):
                rows.extend(indices.tolist())
                cols.extend(np.full(N, k * 6 + d, dtype=int).tolist())
                vals.extend((mat[:, d] * w).tolist())
        rhs.extend((-resid * w).tolist())
        rnum += N
    if not matches:
        raise ValueError('No valid scan-pair correspondences; cannot refine poses')
    data = coo_matrix((vals, (rows, cols)), shape=(rnum, ncols)).tocsr()
    y = np.array(rhs)
    Aextra, bextra = [], []

    def mkreg(sd_t, sd_r):
        return [1 / sd_t] * 3 + [1 / sd_r] * 3

    def ridx(s):
        return (s if s < anchor else s - 1) * 6

    for s in range(nscan):
        if s == anchor:
            continue
        for d, v in enumerate(mkreg(.22, np.deg2rad(3.5))):
            row = np.zeros(ncols)
            row[ridx(s) + d] = v
            Aextra.append(row)
            bextra.append(-X[s, d] * v)
    for i in range(nscan - 1):
        j = i + 1
        for d, v in enumerate(mkreg(.22, np.deg2rad(3.0))):
            row = np.zeros(ncols)
            if i != anchor:
                row[ridx(i) + d] += v
            if j != anchor:
                row[ridx(j) + d] -= v
            Aextra.append(row)
            bextra.append(-(X[i, d] - X[j, d]) * v)
    a = svstack([data, csr_matrix(np.array(Aextra))], format='csr')
    b = np.r_[y, bextra]
    result = lsmr(a, b, atol=1e-8, btol=1e-8, maxiter=450)
    dx = np.zeros((nscan, 6))
    for s in range(nscan):
        if s != anchor:
            dx[s] = result[0][ridx(s):ridx(s) + 6]
    for s in range(nscan):
        t, r = np.linalg.norm(dx[s, :3]), np.linalg.norm(dx[s, 3:])
        if t > .10:
            dx[s, :3] *= .10 / t
        if r > np.deg2rad(1.5):
            dx[s, 3:] *= np.deg2rad(1.5) / r
    rmses = [np.einsum('ij,ij->i', n @ Rotation.from_rotvec(X[j, 3:]).as_matrix().T,
                                ptransform(p, X[i], centers[i]) - ptransform(q, X[j], centers[j]))
             for i, j, p, q, n in matches]
    report = dict(matches=sum(len(p) for i, j, p, q, n in matches), pairs=len(matches),
                  resid_rmse_cm=float(np.sqrt(np.mean(np.hstack(rmses) ** 2)) * 100))
    return dx, report


def apply_corrections(A: NDArray, scan: NDArray, X: NDArray, centers: NDArray) -> NDArray[np.float32]:
    out = np.empty(A.shape, dtype=np.float32)
    for s in range(len(X)):
        mask = scan == s
        out[mask] = ptransform(A[mask], X[s], centers[s]).astype(np.float32)
    return out


def correction_matrices(X: NDArray, centers: NDArray) -> NDArray[np.float64]:
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], len(X), axis=0)
    for s in range(len(X)):
        R = Rotation.from_rotvec(X[s, 3:]).as_matrix()
        transforms[s, :3, :3] = R
        transforms[s, :3, 3] = centers[s] + X[s, :3] - R @ centers[s]
    return transforms


def _validate_cloud(cloud: AccumulatedCloud, anchor: int) -> tuple[NDArray, NDArray]:
    A = np.asarray(cloud.xyz, dtype=np.float64)
    scan = np.asarray(cloud.scan_id, dtype=np.int64)
    unique = np.unique(scan)
    if len(unique) < 2 or not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError('Expected >=2 contiguous scan IDs starting from zero')
    if anchor < 0 or anchor >= len(unique):
        raise ValueError('Anchor scan not present')
    if not np.isfinite(A).all():
        raise ValueError('Non-finite points')
    return A, scan


def solve_joint(cloud: AccumulatedCloud, settings: JointSettings = JointSettings()) -> ScanOptimizationResult:
    A, scan = _validate_cloud(cloud, settings.anchor_scan_id)
    indices, centers, normals, oks = sample_model(A, scan, settings)
    X = np.zeros((len(indices), 6))
    reports = []
    for iteration in range(settings.iterations):
        pairs = match_pairs(A, scan, indices, centers, normals, oks, X, settings)
        dx, report = solve_step(pairs, centers, X, settings)
        X += dx
        report.update(iteration=iteration,
                      translation_step_max_cm=float(np.linalg.norm(dx[:, :3], axis=1).max() * 100),
                      rotation_step_max_deg=float(np.rad2deg(np.linalg.norm(dx[:, 3:], axis=1).max())))
        reports.append(report)
        if np.linalg.norm(dx) < 1e-5:
            break
    return ScanOptimizationResult('Joint', apply_corrections(A, scan, X, centers),
                                  np.unique(scan), X, centers, correction_matrices(X, centers),
                                  tuple(reports), settings.anchor_scan_id)