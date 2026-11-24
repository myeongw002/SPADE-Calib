"""Metric segment geometry and provisional plane-based calibration.

No ground truth is used here. Global SfM landmarks are filtered by in-segment
track support, not locally bundle adjusted or independently triangulated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Iterable, Iterator

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from spade_calib.correspondence.sfm_lidar_planes import LidarPatch
from spade_calib.selection.scale_segmentation import validate_extrinsic


@dataclass(frozen=True)
class SparseTrack:
    point_id: int
    xyz: np.ndarray
    error_px: float
    image_ids: tuple[int, ...]


def read_sparse_tracks(path: str | Path) -> Iterator[SparseTrack]:
    """Read COLMAP points3D.txt/bin, including track image IDs, once per model."""
    path = Path(path)
    if path.suffix == ".txt":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                v = line.split()
                if len(v) < 8 or (len(v) - 8) % 2:
                    raise ValueError("Malformed COLMAP points3D.txt track")
                yield SparseTrack(int(v[0]), np.array(v[1:4], dtype=float),
                                  float(v[7]), tuple(int(x) for x in v[8::2]))
    elif path.suffix == ".bin":
        def read(f, size):
            b = f.read(size)
            if len(b) != size:
                raise ValueError("Truncated COLMAP points3D.bin")
            return b
        header = struct.Struct("<QdddBBBdQ")
        with path.open("rb") as f:
            count = struct.unpack("<Q", read(f, 8))[0]
            if count > path.stat().st_size // header.size:
                raise ValueError("Invalid COLMAP point count")
            for _ in range(count):
                pid, x, y, z, _, _, _, error, length = header.unpack(read(f, header.size))
                if length > (path.stat().st_size - f.tell()) // 8:
                    raise ValueError("Invalid COLMAP track length")
                ij = np.frombuffer(read(f, int(length) * 8), dtype="<i4").reshape(-1, 2)
                yield SparseTrack(pid, np.array([x, y, z]), error,
                                  tuple(int(i) for i in ij[:, 0]))
            if f.read(1):
                raise ValueError("Unexpected trailing COLMAP points3D bytes")
    else:
        raise ValueError("Use COLMAP points3D.bin or points3D.txt")


def metric_local_points(tracks: Iterable[SparseTrack], centers: dict[int, np.ndarray],
                        reference_R_world_camera: np.ndarray,
                        reference_t_world_camera: np.ndarray, scale: float,
                        *, min_track: int = 2, max_error_px: float = 2.,
                        min_parallax_deg: float = 1.) -> tuple[np.ndarray, np.ndarray]:
    """Select points with local track support; map world XYZ into anchor camera.

    Parallax is a conservative lower bound: maximum angle to first observed
    camera ray. Reported COLMAP reprojection error is global, not local.
    """
    if (not np.isfinite([scale, max_error_px, min_parallax_deg]).all()
            or scale <= 0 or min_track < 2 or max_error_px <= 0
            or not 0 <= min_parallax_deg < 180):
        raise ValueError("Invalid SfM filter/scale")
    R = np.asarray(reference_R_world_camera, dtype=float)
    t = np.asarray(reference_t_world_camera, dtype=float)
    if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("Invalid reference pose")
    points, ids = [], []
    for p in tracks:
        if not np.isfinite(p.xyz).all() or not 0 <= p.error_px <= max_error_px:
            continue
        seen = sorted(set(p.image_ids).intersection(centers))
        if len(seen) < min_track:
            continue
        rays = p.xyz - np.stack([centers[i] for i in seen])
        norm = np.linalg.norm(rays, axis=1)
        if np.any(norm < 1e-9):
            continue
        rays /= norm[:, None]
        angle = float(np.rad2deg(np.arccos(np.clip(np.min(rays @ rays[0]), -1., 1.))))
        if angle < min_parallax_deg:
            continue
        points.append(scale * ((p.xyz - t) @ R))
        ids.append(p.point_id)
    return np.asarray(points, dtype=float).reshape(-1, 3), np.asarray(ids, dtype=np.uint64)


@dataclass(frozen=True)
class PlanePair:
    segment_id: int
    camera: LidarPatch  # generic finite metric patch, in anchor camera coordinates
    lidar: LidarPatch
    angle_deg: float
    overlap_camera: float
    overlap_lidar: float


def associate_patches(camera: list[LidarPatch], lidar: list[LidarPatch], X: np.ndarray,
                      segment_id: int, *, max_angle_deg: float = 15.,
                      max_offset_m: float = 1.5, support_radius_m: float = 1.,
                      min_overlap: float = .1, min_margin: float = .12
                      ) -> list[PlanePair]:
    """Mutual-best finite-support candidates under initial X; not verified matches."""
    X = validate_extrinsic(X)
    if (not np.isfinite([max_angle_deg, max_offset_m, support_radius_m,
                         min_overlap, min_margin]).all()
            or not 0 < max_angle_deg < 90 or min(max_offset_m, support_radius_m) <= 0
            or not 0 < min_overlap <= 1 or min_margin < 0):
        raise ValueError("Invalid plane association gates")
    R, t = X[:3, :3], X[:3, 3]
    candidates = {}
    for i, c in enumerate(camera):
        pc = c.support_xyz_lidar
        tc = cKDTree(pc)
        for j, l in enumerate(lidar):
            pl = l.support_xyz_lidar @ R.T + t
            nl = R @ l.normal
            dl = l.offset_m - nl @ t
            angle = float(np.rad2deg(np.arccos(np.clip(abs(c.normal @ nl), 0, 1))))
            offset = max(abs(c.normal @ pl.mean(0) + c.offset_m),
                         abs(nl @ pc.mean(0) + dl))
            if angle > max_angle_deg or offset > max_offset_m:
                continue
            oc = float(np.mean(cKDTree(pl).query(pc, workers=1)[0] <= support_radius_m))
            ol = float(np.mean(tc.query(pl, workers=1)[0] <= support_radius_m))
            if min(oc, ol) < min_overlap:
                continue
            score = angle / max_angle_deg + offset / max_offset_m + 1 - (oc + ol) / 2
            candidates[i, j] = (score, PlanePair(segment_id, c, l, angle, oc, ol))
    def unique_best(rows):
        rows = sorted(rows)
        if not rows or (len(rows) > 1 and rows[1][0] - rows[0][0] < min_margin):
            return None
        return rows[0][1]
    out = []
    for i in range(len(camera)):
        j = unique_best([(v[0], b) for (a, b), v in candidates.items() if a == i])
        if j is not None and unique_best([(v[0], a) for (a, b), v in candidates.items()
                                          if b == j]) == i:
            out.append(candidates[i, j][1])
    return out


def plane_pair_diagnostics(pairs: list[PlanePair], X: np.ndarray) -> list[dict]:
    X = validate_extrinsic(X)
    R, t = X[:3, :3], X[:3, 3]
    results = []
    for p in pairs:
        c, l = p.camera, p.lidar
        pc = c.support_xyz_lidar
        pl = l.support_xyz_lidar @ R.T + t
        r1 = pl @ c.normal + c.offset_m
        r2 = ((pc - t) @ R) @ l.normal + l.offset_m
        results.append({"segment_id": p.segment_id, "camera_patch": c.patch_id,
                        "lidar_patch": l.patch_id, "association_angle_deg": p.angle_deg,
                        "association_overlap_camera": p.overlap_camera,
                        "association_overlap_lidar": p.overlap_lidar,
                        "normal_angle_deg": float(np.rad2deg(np.arccos(np.clip(
                            abs(c.normal @ (R @ l.normal)), 0, 1)))),
                        "camera_to_lidar_median_abs_m": float(np.median(abs(r2))),
                        "lidar_to_camera_median_abs_m": float(np.median(abs(r1)))})
    return results


def refine_shared_extrinsic(pairs: list[PlanePair], X0: np.ndarray, *,
                            sigma_m: float = .15, max_rotation_deg: float = 5.,
                            max_translation_m: float = 1., max_points: int = 500,
                            seed: int = 42) -> dict:
    """Fixed-association bidirectional point-to-plane 6DoF diagnostic.

    Equal segment, plane and direction weights. Robust loss is applied to
    raw normalized residuals BEFORE weighting. Rank is a numerical proxy.
    No motion term, scale update, correspondence rematching or accuracy claim.
    """
    X0 = validate_extrinsic(X0)
    if (not np.isfinite([sigma_m, max_rotation_deg, max_translation_m]).all()
            or min(sigma_m, max_rotation_deg, max_translation_m) <= 0
            or max_rotation_deg >= 90 or max_points < 3):
        raise ValueError("Invalid refinement settings")
    if not pairs:
        return {"success": False, "reason": "no_plane_pairs"}
    rng = np.random.default_rng(seed)
    counts = {s: sum(p.segment_id == s for p in pairs) for s in {p.segment_id for p in pairs}}
    blocks = []
    for p in pairs:
        def sample(x):
            return x if len(x) <= max_points else x[rng.choice(len(x), max_points, replace=False)]
        pc, pl = sample(p.camera.support_xyz_lidar), sample(p.lidar.support_xyz_lidar)
        w = 1. / (len(counts) * counts[p.segment_id] * 2)
        blocks.append((p, pc, pl, w))
    def state(delta):
        X = X0.copy()
        X[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix() @ X0[:3, :3]
        X[:3, 3] += delta[3:]
        return X
    def objective(delta, robust=True):
        X = state(delta)
        R, t = X[:3, :3], X[:3, 3]
        out = []
        for p, pc, pl, w in blocks:
            r1 = ((pl @ R.T + t) @ p.camera.normal + p.camera.offset_m) / sigma_m
            r2 = (((pc - t) @ R) @ p.lidar.normal + p.lidar.offset_m) / sigma_m
            for r in (r1, r2):
                # Squared transformed residual is soft-L1 rho(r^2).
                if robust:
                    r = r * np.sqrt(2. / (np.hypot(1., r) + 1.))
                out.append(np.sqrt(w / len(r)) * r)
        return np.concatenate(out)
    def rank_at(delta):
        step = np.eye(6) * np.array([np.deg2rad(1.)]*3 + [.1]*3) * 1e-5
        J = np.column_stack([(objective(delta+d, False)-objective(delta-d, False))/2e-5
                             for d in step])
        sv = np.linalg.svd(J, compute_uv=False)
        return int(np.sum(sv > max(sv[0], 1e-15)*1e-8)), sv.tolist()
    zero = np.zeros(6)
    rank, sv = rank_at(zero)
    if rank < 6:
        return {"success": False, "reason": "rank_deficient", "initial_rank": rank,
                "initial_singular_values": sv}
    bound = np.array([np.deg2rad(max_rotation_deg)]*3 + [max_translation_m]*3)
    sol = least_squares(objective, zero, bounds=(-bound, bound), x_scale="jac",
                        max_nfev=200, ftol=1e-9, gtol=1e-9, xtol=1e-9)
    final_rank, final_sv = rank_at(sol.x)
    X = state(sol.x)
    return {"success": bool(sol.success and final_rank == 6), "solver_converged": bool(sol.success),
            "message": str(sol.message), "T_camera_lidar": X.tolist(),
            "initial_rank": rank, "initial_singular_values": sv,
            "final_rank": final_rank, "final_singular_values": final_sv,
            "bound_hit": bool(np.any(sol.active_mask)),
            "cost_initial": float(.5*np.sum(objective(zero)**2)), "cost_final": float(sol.cost),
            "plane_pairs": len(pairs), "segments": len(counts),
            "warning": "Provisional fixed correspondences; rank does not prove physical accuracy",
            "before": plane_pair_diagnostics(pairs, X0), "after": plane_pair_diagnostics(pairs, X)}
