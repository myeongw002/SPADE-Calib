#!/usr/bin/env python3
"""KITTI 07 fixed-frame vs fixed-distance scan accumulation pilot.

Run against the installed SPADE-Calib repository. The registration and rigid
accumulation are the repository's ORIGINAL solve_joint()/accumulate_glim().
No pointwise deskew is attempted for KITTI's XYZI-only .bin scans.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spade_calib.io.kitti_bin import load_kitti_bin_scan
from spade_calib.lidar import (GlimTrajectory, accumulate_glim, JointSettings,
                              PGOSettings, solve_joint, solve_pgo)
from spade_calib.lidar.joint_solver import voxel_sample

LOG = logging.getLogger('spade_accum')
# Intervals from uploaded KITTI07 sfm_scene_selection(2).zip (seconds relative to times.txt).
WINDOWS = {'005': (37.5, 52.5), '012': (90.0, 105.0)}


@dataclass(frozen=True)
class Experiment:
    name: str
    selected: tuple[int, ...]
    method: str
    parameter: float
    target: int
    window: str
    start_s: float
    stop_s: float


def load_kitti_glim(times_file: Path, odom_file: Path, pose_index_offset: int = 0
                    ) -> tuple[NDArray[np.float64], GlimTrajectory, dict[str, Any]]:
    """Use explicit row-index association, NOT GLIM's often synthetic timestamps.

    Pose row j is assigned to KITTI scan (j + pose_index_offset). Out-of-range
    frames are not silently given the last available pose.
    """
    times = np.atleast_1d(np.loadtxt(times_file, dtype=np.float64))
    odom = np.atleast_2d(np.loadtxt(odom_file, dtype=np.float64))
    if times.ndim != 1 or len(times) < 2 or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('KITTI times.txt must contain >=2 strictly increasing finite seconds')
    if odom.ndim != 2 or odom.shape[1] != 8 or len(odom) < 2 or not np.isfinite(odom).all():
        raise ValueError('GLIM odom_lidar.txt must be Nx8: time x y z qx qy qz qw')
    if np.any(np.diff(odom[:, 0]) <= 0):
        raise ValueError('Original GLIM timestamps must increase strictly even when index mapping is selected')
    if type(pose_index_offset) is not int or pose_index_offset < 0 or pose_index_offset + len(odom) > len(times):
        raise ValueError('pose-index-offset must map every GLIM pose to an existing KITTI frame')
    idx = np.arange(len(odom)) + pose_index_offset
    # Valid mapping is deliberately explicit. Preserve the GLIM quaternion/translation
    # and reindex temporal labels so the existing rigid accumulation can be reused.
    aligned = np.column_stack((times[idx], odom[:, 1:]))
    traj = GlimTrajectory.from_array(aligned)
    meta = {
        'mapping': 'GLIM row j -> KITTI frame j + pose_index_offset',
        'pose_index_offset': int(pose_index_offset),
        'glim_original_first_timestamp_s': float(odom[0, 0]),
        'kitti_first_timestamp_s': float(times[0]),
        'number_of_scans_in_times_file': len(times),
        'number_of_glim_pose_rows': len(odom),
        'first_mapped_frame': int(idx[0]),
        'last_mapped_frame': int(idx[-1]),
        'warning': 'GLIM timestamps replaced solely for index-association; verify row-index correspondence for any other dataset',
    }
    return times, traj, meta


def frames_in_window(times: NDArray, lo: float, hi: float, last_pose_frame: int,
                     first_pose_frame: int = 0) -> NDArray[np.int64]:
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError('Invalid scene boundaries')
    frames = np.flatnonzero((times >= lo) & (times <= hi) &
                            (np.arange(len(times)) >= first_pose_frame) &
                            (np.arange(len(times)) <= last_pose_frame))
    if len(frames) == 0:
        raise ValueError('No pose-covered frames inside requested scene window')
    return frames


def select_fixed_frame(candidates: NDArray, target: int, half_frames: int) -> NDArray:
    if half_frames < 0:
        raise ValueError('half_frames must be nonnegative')
    if target not in candidates:
        raise ValueError('Target not available in candidate window')
    chosen = candidates[np.abs(candidates - target) <= half_frames]
    if target not in chosen:
        raise ValueError('Target not available in candidate window')
    return chosen


def select_fixed_distance(candidates: NDArray, target: int, radius_m: float,
                          positions: NDArray, *, position_offset: int = 0) -> NDArray:
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError('radius_m must be positive and finite')
    if target not in candidates:
        raise ValueError('Target not available in candidate window')
    offsets = candidates - position_offset
    k = target - position_offset
    if np.any(offsets < 0) or np.any(offsets >= len(positions)) or k < 0 or k >= len(positions):
        raise ValueError('Selected frames are outside mapped pose rows')
    dist = np.linalg.norm(positions[offsets] - positions[k], axis=1)
    chosen = candidates[dist <= radius_m + 1e-10]
    if target not in chosen:
        raise RuntimeError('Internal error: target missing from distance selection')
    return chosen


def select_distance_budget(candidates: NDArray, target: int, budget: int,
                           positions: NDArray, *, position_offset: int = 0) -> NDArray:
    """Controlled comparison: N spatial-nearest frames (NOT fixed radius)."""
    if target not in candidates or budget < 2 or budget > len(candidates):
        raise ValueError('Invalid target/budget for distance-budget method')
    d = np.linalg.norm(positions[candidates-position_offset] -
                       positions[target-position_offset], axis=1)
    # Stable ties: temporal proximity and original frame ID. Keep target.
    order = np.lexsort((candidates, np.abs(candidates-target), d))
    selected = np.sort(candidates[order[:budget]])
    assert target in selected
    return selected


def travel_distance(positions: NDArray, frames: NDArray, pose_index_offset: int) -> float:
    if len(frames) < 2:
        return 0.0
    # Path length over all *intervening* odometry frames, not merely endpoints.
    f = np.arange(int(frames[0]), int(frames[-1]) + 1) - pose_index_offset
    return float(np.linalg.norm(np.diff(positions[f], axis=0), axis=1).sum())


def prepare_experiments(args: argparse.Namespace, times: NDArray, traj: GlimTrajectory,
                        pose_index_offset: int) -> list[Experiment]:
    if args.window in WINDOWS and args.start_s is None and args.stop_s is None:
        start, stop = WINDOWS[args.window]
    elif args.start_s is not None and args.stop_s is not None:
        start, stop = args.start_s, args.stop_s
    else:
        raise ValueError('Use window 005/012, or provide both --start-s and --stop-s')
    first_pose = pose_index_offset
    last_pose = pose_index_offset + len(traj.timestamps) - 1
    candidates = frames_in_window(times, start, stop, last_pose, first_pose)
    if args.target_frame not in candidates:
        raise ValueError(f'Target frame {args.target_frame} is outside scene [{start}, {stop}] '
                         f'or outside GLIM pose coverage [{first_pose}, {last_pose}]')
    plans = []
    for n in args.half_frames:
        chosen = select_fixed_frame(candidates, args.target_frame, n)
        if len(chosen) >= 2:
            plans.append(Experiment(f'fixed_frame_n{n}', tuple(map(int, chosen)),
                                    'fixed_frame', float(n), args.target_frame,
                                    args.window, start, stop))
        else:
            LOG.warning('Skipping frame half-width %d: fewer than two scans', n)
    for d in args.radii_m:
        chosen = select_fixed_distance(candidates, args.target_frame, d, traj.positions,
                                       position_offset=pose_index_offset)
        if len(chosen) >= 2:
            plans.append(Experiment(f'fixed_distance_{d:g}m'.replace('.', 'p'),
                                    tuple(map(int, chosen)), 'fixed_distance', d,
                                    args.target_frame, args.window, start, stop))
        else:
            LOG.warning('Skipping radius %g m: fewer than two scans', d)
    if not plans:
        raise ValueError('No experiment contains at least two scans')
    if args.match_scan_budget:
        for n in args.half_frames:
            target_plan = next((x for x in plans if x.method == 'fixed_frame' and x.parameter == n), None)
            if target_plan is None:
                continue
            chosen = select_distance_budget(candidates, args.target_frame, len(target_plan.selected),
                                            traj.positions, position_offset=pose_index_offset)
            plans.append(Experiment(f'distance_budget_n{n}', tuple(map(int,chosen)),
                                    'distance_budget', float(len(chosen)), args.target_frame,
                                    args.window, start, stop))
    if len({x.name for x in plans}) != len(plans):
        raise ValueError('Duplicate half-frame/radius experiment names')
    return plans


def load_scans(velodyne: Path, indices: tuple[int, ...], times: NDArray,
               min_range_m: float, max_range_m: float):
    scans = []
    for local_idx, original_frame in enumerate(indices):
        path = velodyne / f'{original_frame:06d}.bin'
        if not path.is_file():
            raise FileNotFoundError(f'Missing KITTI scan: {path}')
        stamp = int(round(float(times[original_frame]) * 1e9))
        scans.append(load_kitti_bin_scan(path, scan_id=local_idx, timestamp_ns=stamp,
                                         min_range_m=min_range_m, max_range_m=max_range_m))
        if len(scans[-1].xyz) == 0:
            raise ValueError(f'No in-range LiDAR points in {path}')
    return scans


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        return
    fieldnames = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def voxel_downsample_with_labels(points: NDArray, labels: NDArray, voxel_m: float,
                                  max_points: int, seed: int) -> tuple[NDArray, NDArray]:
    if voxel_m <= 0 or max_points < 100:
        raise ValueError('Invalid voxel or point cap')
    ids = voxel_sample(points, voxel_m)
    if len(ids) > max_points:
        rng = np.random.default_rng(seed)
        ids = np.sort(rng.choice(ids, size=max_points, replace=False))
    return np.asarray(points[ids], dtype=np.float64), np.asarray(labels[ids], dtype=np.int64)


def write_ply_binary(path: Path, xyz: NDArray, scan_ids: NDArray) -> None:
    if len(xyz) != len(scan_ids):
        raise ValueError('PLY points and labels have different lengths')
    xyz = np.asarray(xyz, dtype='<f4')
    scan_ids = np.asarray(scan_ids, dtype=np.int64)
    # Deterministic color assignment, independent of random hash seeds.
    r = ((scan_ids * 73 + 42) % 230 + 20).astype('u1')
    g = ((scan_ids * 137 + 77) % 230 + 20).astype('u1')
    b = ((scan_ids * 31 + 163) % 230 + 20).astype('u1')
    dat = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for k, key in enumerate('xyz'):
        dat[key] = xyz[:, k]
    dat['red'], dat['green'], dat['blue'] = r, g, b
    with path.open('wb') as fh:
        fh.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(dat)}\n'
                  'property float x\nproperty float y\nproperty float z\n'
                  'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                  'end_header\n').encode('ascii'))
        dat.tofile(fh)



def save_preview(path: Path, xyz: NDArray, scan_ids: NDArray, title: str) -> None:
    """Two simple orthographic views; geometric diagnostics only."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        LOG.warning('matplotlib unavailable; skipping %s', path.name)
        return
    n = len(xyz)
    ids = np.arange(n)
    if n > 22000:
        ids = np.random.default_rng(20261116).choice(n, 22000, replace=False)
    sample = xyz[ids]
    lab = scan_ids[ids]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), constrained_layout=True)
    for ax, h, v, xname, yname in ((axes[0], 0, 1, 'X [m]', 'Y [m]'),
                                    (axes[1], 0, 2, 'X [m]', 'Z [m]')):
        ax.scatter(sample[:, h], sample[:, v], c=lab, cmap='turbo', s=.25,
                   alpha=.6, rasterized=True)
        ax.scatter([0], [0], marker='+', c='black', s=50, label='Target LiDAR')
        ax.set_xlabel(xname); ax.set_ylabel(yname); ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.2)
    axes[0].set_title('Top view (scan ID colors)')
    axes[1].set_title('Side view')
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)

def plane_pca(points: NDArray) -> tuple[NDArray, float, NDArray, NDArray]:
    if len(points) < 3:
        raise ValueError('Need >=3 points to estimate plane')
    c = np.mean(points, axis=0)
    eig, vec = np.linalg.eigh(np.cov(points - c, rowvar=False, bias=True))
    n = vec[:, 0]
    return n, -float(n @ c), c, eig


def angle_between_unoriented(a: NDArray, b: NDArray) -> float:
    return float(np.rad2deg(np.arccos(np.clip(abs(float(a @ b)), 0.0, 1.0))))


def detect_ransac_plane(points: NDArray, *, threshold: float, iters: int,
                        up: NDArray, kind: str, rng: np.random.Generator,
                        min_inliers: int, prior_normal: NDArray | None = None,
                        prior_max_angle_deg: float = 30.0) -> tuple[NDArray, float, NDArray] | None:
    """RANSAC with normal filtering; return plane and input-point inlier mask.

    The score is support count; physical correctness requires visual verification.
    """
    n_total = len(points)
    if n_total < min_inliers:
        return None
    best_idx = None
    best_count = 0
    cos_ground = math.cos(math.radians(30.0))
    cos_prior = math.cos(math.radians(prior_max_angle_deg))
    for _ in range(iters):
        tri = points[rng.choice(n_total, 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        mag = float(np.linalg.norm(n))
        if mag < 1e-5:
            continue
        n /= mag
        vertical = abs(float(n @ up))
        if kind == 'ground' and vertical < cos_ground:
            continue
        if kind == 'wall' and vertical > 0.65:
            continue
        if prior_normal is not None and abs(float(n @ prior_normal)) < cos_prior:
            continue
        d = -float(n @ tri[0])
        # Work on the entire current candidate set; avoid unintended random
        # score inflation due to a varying subset of points.
        mask = np.abs(points @ n + d) < threshold
        cnt = int(mask.sum())
        if cnt > best_count:
            best_count, best_idx = cnt, mask
    if best_idx is None or best_count < min_inliers:
        return None
    # Refine PCA and repeat support assignment, maintaining normal-type prior.
    n, d, _, _ = plane_pca(points[best_idx])
    if kind == 'ground' and abs(float(n @ up)) < cos_ground:
        return None
    if kind == 'wall' and abs(float(n @ up)) > 0.65:
        return None
    if prior_normal is not None and abs(float(n @ prior_normal)) < cos_prior:
        return None
    mask = np.abs(points @ n + d) < threshold
    if int(mask.sum()) < min_inliers:
        return None
    n, d, _, _ = plane_pca(points[mask])
    return n, d, mask


def plane_metrics(points: NDArray, scan_ids: NDArray, normal: NDArray, offset: float,
                  *, kind: str, plane_id: str, stage: str, window: str, experiment: str,
                  min_split_points: int = 40) -> dict[str, Any]:
    # Plane parameters and all residuals are in the SAME target LiDAR frame.
    r = points @ normal + offset
    tangent = np.eye(3) - normal[:, None] @ normal[None, :]
    pc = (points - np.mean(points, axis=0)) @ tangent
    eig = np.linalg.eigvalsh(np.cov(pc, rowvar=False, bias=True))
    two = np.maximum(np.sort(eig)[-2:], 0.0)
    angle_split = None
    offset_split = None
    unique_scans = np.unique(scan_ids)
    if len(unique_scans) >= 4:
        # Split by original scan ID parity; not statistically independent
        # because both halves use the same odometry/refinement front-end.
        groups = [points[(scan_ids % 2) == parity] for parity in (0, 1)]
        if all(len(g) >= min_split_points for g in groups):
            n0, d0, _, _ = plane_pca(groups[0])
            n1, d1, _, _ = plane_pca(groups[1])
            if n0 @ n1 < 0:
                n1, d1 = -n1, -d1
            angle_split = angle_between_unoriented(n0, n1)
            offset_split = abs(float(d0 - d1))
    return {
        'window': window, 'experiment': experiment, 'stage': stage,
        'plane_id': plane_id, 'kind': kind, 'inliers': len(points),
        'support_scans': len(unique_scans),
        'nx': float(normal[0]), 'ny': float(normal[1]), 'nz': float(normal[2]),
        'offset_m': float(offset),
        'rmse_m': float(np.sqrt(np.mean(r * r))),
        'median_abs_residual_m': float(np.median(np.abs(r))),
        'coverage_eig_major_m2': float(two[1]),
        'coverage_eig_minor_m2': float(two[0]),
        'coverage_area_proxy_m2': float(4 * np.sqrt(two[0] * two[1])),
        'normal_split_deg': angle_split,
        'offset_split_m': offset_split,
        'split_is_diagnostic_not_ground_truth': True,
    }


def extract_planes(xyz: NDArray, labels: NDArray, args: argparse.Namespace,
                   *, experiment: str, stage: str) -> list[dict[str, Any]]:
    # Plane extraction always uses identically parameterized, uniformly
    # downsampled point sets for both methods.
    up = np.asarray(args.up, dtype=np.float64)
    up /= np.linalg.norm(up)
    mask = np.ones(len(xyz), dtype=bool)
    if args.roi is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = args.roi
        mask &= ((xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
                 (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax) &
                 (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax))
    work = xyz[mask]
    lab = labels[mask]
    if len(work) < args.min_plane_points:
        return []
    rng = np.random.default_rng(args.seed)
    rows = []
    # Ground: constrain to physically plausible near-sensor heights in LiDAR z.
    gmask = ((work[:, 2] >= args.ground_z_min) &
             (work[:, 2] <= args.ground_z_max))
    candidates = work[gmask]
    if len(candidates) >= args.min_plane_points:
        result = detect_ransac_plane(candidates, threshold=args.ransac_threshold_m,
                                     iters=args.ransac_iters, up=up, kind='ground',
                                     rng=rng, min_inliers=args.min_plane_points)
        if result is not None:
            n, d, _ = result
            if n @ up < 0:
                n, d = -n, -d
            gm = np.abs(work @ n + d) < args.ransac_threshold_m
            rows.append(plane_metrics(work[gm], lab[gm], n, d,
                                      kind='ground', plane_id='ground_0',
                                      stage=stage, window=args.window, experiment=experiment))
            # Remove all near-ground support before searching non-ground planes.
            keep = ~gm
            work, lab = work[keep], lab[keep]
    prior = None
    if args.wall_normal is not None:
        prior = np.asarray(args.wall_normal, dtype=np.float64)
        prior /= np.linalg.norm(prior)
    for j in range(args.max_walls):
        res = detect_ransac_plane(work, threshold=args.ransac_threshold_m,
                                  iters=args.ransac_iters, up=up, kind='wall', rng=rng,
                                  min_inliers=args.min_plane_points,
                                  prior_normal=prior, prior_max_angle_deg=args.wall_normal_tol_deg)
        if res is None:
            break
        n, d, m = res
        if n[np.argmax(np.abs(n))] < 0:
            n, d = -n, -d
        row = plane_metrics(work[m], lab[m], n, d, kind='non_ground',
                            plane_id=f'wall_{j}', stage=stage,
                            window=args.window, experiment=experiment)
        # Suppress probable duplicate plane hypotheses. Distinct coplanar patches
        # may still remain indistinguishable; this does not establish identity.
        duplicate = any(
            prev['kind'] == 'non_ground' and
            angle_between_unoriented(n, np.array([prev['nx'], prev['ny'], prev['nz']])) < 5.0 and
            abs(float(d - prev['offset_m'])) < .18
            for prev in rows
        )
        if not duplicate:
            rows.append(row)
        work, lab = work[~m], lab[~m]
    return rows


def validate_cli(args: argparse.Namespace) -> None:
    if args.window not in WINDOWS and (args.start_s is None or args.stop_s is None):
        raise ValueError('Window must be 005/012, or both --start-s and --stop-s')
    if not (0.0 < args.min_range_m < args.max_range_m):
        raise ValueError('Invalid LiDAR range')
    if args.pose_index_offset < 0 or args.target_frame < 0:
        raise ValueError('Offsets/frame must be nonnegative')
    for values, name, low in ((args.half_frames, 'half-frames', 0), (args.radii_m, 'radii-m', 0)):
        if not values:
            raise ValueError(f'{name} cannot be empty')
        if any(not math.isfinite(float(v)) or float(v) <= low for v in values):
            raise ValueError(f'Invalid {name}')
    if args.max_output_points < 100 or args.max_plane_points < 100 or args.min_plane_points < 3 or args.max_walls < 0:
        raise ValueError('Invalid point caps')
    if (args.ransac_iters < 1 or args.ransac_threshold_m <= 0 or args.voxel_m <= 0
            or args.refine_iterations < 1 or args.refine_max_samples < 24
            or args.refine_voxel_m <= 0 or args.pgo_iterations < 1 or args.pgo_pair_max_nfev < 1
            or not (0 < args.wall_normal_tol_deg <= 90)):
        raise ValueError('Invalid RANSAC/voxel settings')
    if args.roi is not None and any(args.roi[i] >= args.roi[i+1] for i in (0, 2, 4)):
        raise ValueError('ROI bounds must be min < max')
    for vec, name in ((args.up, 'up'), (args.wall_normal, 'wall-normal')):
        if vec is not None and (not np.isfinite(vec).all() or np.linalg.norm(vec) < 1e-12):
            raise ValueError(f'{name} must be nonzero finite vector')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--velodyne', type=Path, help='KITTI seq07/velodyne folder (not required for --dry-run)')
    p.add_argument('--times', type=Path, required=True, help='KITTI seq07/times.txt')
    p.add_argument('--odom', type=Path, required=True, help='GLIM odom_lidar.txt (Nx8)')
    p.add_argument('--pose-index-offset', type=int, default=0,
                   help='Map GLIM pose row j to KITTI scan j+offset; verify correspondence for your GLIM log')
    p.add_argument('--window', default='005', help='005 or 012, alternatively supply explicit start/stop')
    p.add_argument('--start-s', type=float, help='Override scene start time, in KITTI times.txt seconds')
    p.add_argument('--stop-s', type=float, help='Override scene stop time')
    p.add_argument('--target-frame', type=int, default=384, help='KITTI scan index and reference LiDAR frame')
    p.add_argument('--half-frames', type=int, nargs='+', default=[5, 10], help='±N scan bounds')
    p.add_argument('--radii-m', type=float, nargs='+', default=[3.0, 6.0], help='Euclidean distance radii from target [m]')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true', help='Select frames without loading LiDAR data')
    p.add_argument('--match-scan-budget', action='store_true',
                   help='Add closest-pose selection with exactly the same scan budget as Fixed Frame; NOT fixed-radius')
    p.add_argument('--refiner', choices=['joint', 'pgo', 'none'], default='joint')
    p.add_argument('--refine-iterations', type=int, default=4)
    p.add_argument('--refine-voxel-m', type=float, default=.16)
    p.add_argument('--refine-max-samples', type=int, default=2500)
    p.add_argument('--pgo-iterations', type=int, default=4)
    p.add_argument('--pgo-pair-max-nfev', type=int, default=38)
    p.add_argument('--min-range-m', type=float, default=1.0)
    p.add_argument('--max-range-m', type=float, default=70.0)
    p.add_argument('--voxel-m', type=float, default=.14, help='Plane/PLY uniform voxel sampling')
    p.add_argument('--max-plane-points', type=int, default=45000)
    p.add_argument('--max-output-points', type=int, default=180000)
    p.add_argument('--ransac-iters', type=int, default=300)
    p.add_argument('--ransac-threshold-m', type=float, default=.14)
    p.add_argument('--min-plane-points', type=int, default=160)
    p.add_argument('--max-walls', type=int, default=2)
    p.add_argument('--ground-z-min', type=float, default=-3.4)
    p.add_argument('--ground-z-max', type=float, default=.2)
    p.add_argument('--up', type=float, nargs=3, default=[0, 0, 1], metavar=('UX', 'UY', 'UZ'))
    p.add_argument('--wall-normal', type=float, nargs=3, metavar=('NX', 'NY', 'NZ'),
                   help='Optional rough target non-ground normal, expressed in target LiDAR frame')
    p.add_argument('--wall-normal-tol-deg', type=float, default=35.)
    p.add_argument('--roi', type=float, nargs=6,
                   metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX', 'ZMIN', 'ZMAX'),
                   help='Optional same target-LiDAR-frame ROI for all experiments')
    p.add_argument('--seed', type=int, default=20261116)
    p.add_argument('--no-preview', action='store_true', help='Do not render top/side PNG previews')
    p.add_argument('--verbose', action='store_true')
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    validate_cli(args)
    args.out.mkdir(parents=True, exist_ok=True)
    times, traj, mapping = load_kitti_glim(args.times, args.odom, args.pose_index_offset)
    plans = prepare_experiments(args, times, traj, args.pose_index_offset)
    manifest = {'mapping': mapping, 'scene': {'window': args.window, 'target_frame': args.target_frame},
                'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'experiments': []}
    scan_rows, quality_rows, summary_rows = [], [], []
    for plan in plans:
        t0 = time.perf_counter()
        out = args.out / plan.name
        out.mkdir(parents=True, exist_ok=True)
        frames = np.asarray(plan.selected, dtype=np.int64)
        anchor_local = plan.selected.index(plan.target)
        distance_traveled = travel_distance(traj.positions, frames, args.pose_index_offset)
        displacement = float(np.linalg.norm(traj.positions[frames[-1] - args.pose_index_offset] -
                                            traj.positions[frames[0] - args.pose_index_offset]))
        for j, frame in enumerate(frames):
            scan_rows.append({'experiment': plan.name, 'method': plan.method,
                              'param': plan.parameter, 'local_scan_id': j,
                              'kitti_frame': int(frame), 'timestamp_s': float(times[frame]),
                              'is_target': bool(frame == plan.target),
                              'distance_to_target_m': float(np.linalg.norm(
                                  traj.positions[frame - args.pose_index_offset] -
                                  traj.positions[plan.target - args.pose_index_offset]))})
        row = {'experiment': plan.name, 'method': plan.method, 'parameter': plan.parameter,
               'target_frame': plan.target, 'scan_count': len(frames),
               'first_frame': int(frames[0]), 'last_frame': int(frames[-1]),
               'path_length_m': distance_traveled, 'endpoint_distance_m': displacement,
               'anchor_local_scan_id': anchor_local, 'status': 'selected_only'}
        LOG.info('%s: %d scans, anchor=%d, path=%.2fm', plan.name, len(frames), anchor_local, distance_traveled)
        if not args.dry_run:
            if args.velodyne is None:
                raise ValueError('--velodyne required without --dry-run')
            scans = load_scans(args.velodyne, plan.selected, times, args.min_range_m, args.max_range_m)
            target_stamp = int(round(float(times[plan.target]) * 1e9))
            raw = accumulate_glim(scans, traj, target_timestamp_ns=target_stamp, mode='rigid')
            source_count = len(raw.xyz)
            corrections = None
            refine_info: list[dict[str, Any]] = []
            datasets = [('initial', raw.xyz, raw.scan_id)]
            if args.refiner in ('joint', 'pgo'):
                cfg = JointSettings(anchor_scan_id=anchor_local, legacy_roi=False,
                                    iterations=args.refine_iterations,
                                    voxel_m=args.refine_voxel_m,
                                    max_samples_per_scan=args.refine_max_samples,
                                    seed=args.seed)
                if args.refiner == 'joint':
                    result = solve_joint(raw, settings=cfg)
                    solver_name = 'spade_calib.lidar.joint_solver.solve_joint'
                    solver_settings = asdict(cfg)
                else:
                    pgo_cfg = PGOSettings(pairs=cfg, iterations=args.pgo_iterations,
                                          pair_max_nfev=args.pgo_pair_max_nfev)
                    result = solve_pgo(raw, settings=pgo_cfg)
                    solver_name = 'spade_calib.lidar.pgo_solver.solve_pgo'
                    solver_settings = asdict(pgo_cfg)
                if (result.anchor_scan_id != anchor_local or
                        result.transforms.shape != (len(frames), 4, 4) or
                        result.xyz.shape != raw.xyz.shape or
                        not np.isfinite(result.transforms).all() or
                        not np.allclose(result.transforms[anchor_local], np.eye(4), atol=1e-8)):
                    raise RuntimeError(f'{args.refiner}: invalid scan correction transforms')
                # Joint uses 'pairs'/'matches'; PGO uses 'pair_edges'/'matched_returns'.
                refine_info = list(result.iterations)
                n_edges = (refine_info[0].get('pairs', refine_info[0].get('pair_edges', 0))
                           if refine_info else 0)
                if n_edges < len(frames) - 1:
                    LOG.warning('%s: only %s matched pair edges for %s scans; graph cannot be connected',
                                plan.name, n_edges, len(frames))
                corrections = result.corrections
                datasets.append(('refined', result.xyz, raw.scan_id))
                np.save(out / 'refined_corrections.npy', result.corrections)
                np.save(out / 'refined_transforms.npy', result.transforms)
                (out / 'refinement.json').write_text(json.dumps({
                    'solver': solver_name,
                    'original_scan_frames': list(plan.selected),
                    'anchor_scan_id': anchor_local,
                    'scan_centered_transform_convention': 'p_new = R (p - center) + center + t',
                    'iterations': refine_info,
                    'settings': solver_settings,
                    'note': 'Registration quality is not extrinsic calibration ground truth',
                }, indent=2, allow_nan=False), encoding='utf-8')
            for stage, pts, ids in datasets:
                p, l = voxel_downsample_with_labels(np.asarray(pts), np.asarray(ids),
                                                     args.voxel_m, args.max_plane_points, args.seed)
                detected = extract_planes(p, l, args, experiment=plan.name, stage=stage)
                quality_rows.extend(detected)
                (out / f'{stage}_planes.json').write_text(json.dumps(detected, indent=2, allow_nan=False),
                                                           encoding='utf-8')
                vis, vis_ids = voxel_downsample_with_labels(np.asarray(pts), np.asarray(ids),
                                                            args.voxel_m, args.max_output_points, args.seed)
                write_ply_binary(out / f'{stage}_scan_colors.ply', vis, vis_ids)
                if not args.no_preview:
                    save_preview(out / f'{stage}_preview.png', vis, vis_ids,
                                 f'{plan.name} | {stage} | scans={len(frames)}')
                # Keep point provenance for reproducibility without gigantic raw dumps.
                np.savez_compressed(out / f'{stage}_sample.npz', xyz=p.astype('f4'),
                                    local_scan_id=l.astype('u2'),
                                    source_frame_indices=np.asarray(plan.selected, dtype=np.int32))
                row[f'{stage}_plane_count'] = len(detected)
                row[f'{stage}_ground_count'] = sum(r['kind']=='ground' for r in detected)
                row[f'{stage}_non_ground_count'] = sum(r['kind']=='non_ground' for r in detected)
            row.update(status='ok', source_points=source_count,
                       registration_pair_edges_first=(refine_info[0].get('pairs', refine_info[0].get('pair_edges'))
                                                      if refine_info else None),
                       registration_pair_matches_first=(refine_info[0].get('matches', refine_info[0].get('matched_returns'))
                                                        if refine_info else None),
                       graph_connectivity_verified=False)
        row['runtime_sec'] = time.perf_counter() - t0
        summary_rows.append(row)
        manifest['experiments'].append(asdict(plan))
        write_csv(args.out / 'selected_scans.csv', scan_rows)
        write_csv(args.out / 'summary.csv', summary_rows)
        write_csv(args.out / 'plane_quality.csv', quality_rows)
        (args.out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
        LOG.info('Finished %s in %.1fs status=%s', plan.name, row['runtime_sec'], row['status'])
    LOG.info('Output: %s', args.out.resolve())
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        LOG.error('%s: %s', type(exc).__name__, exc)
        sys.exit(2)
