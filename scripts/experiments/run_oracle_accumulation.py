#!/usr/bin/env python3
"""KITTI07 GT-pose reference for frame/radius accumulation (NOT an extrinsic oracle).

Requires the adjacent run_accumulation_experiment.py in a SPADE-Calib checkout.
KITTI Odometry poses are T_world_camera0 and calib 'Tr' is T_camera0_lidar.
No deskew: KITTI Odometry .bin contains neither ring nor per-return timestamps.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from spade_calib.evaluation.kitti import kitti_gt_camera_lidar
from spade_calib.io.kitti_bin import load_kitti_bin_scan
from spade_calib.lidar import (
    GlimTrajectory, JointSettings, PGOSettings, accumulate_glim, solve_joint, solve_pgo,
)

# Reuse precisely the same scan selection and row-index mapping as the pilot.
from run_accumulation_experiment import (  # noqa: E402
    load_kitti_glim,
    load_scans,
    prepare_experiments,
)

LOG = logging.getLogger('spade_oracle')


def _validate_transform(t: NDArray, label: str) -> None:
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError(f'{label} must be finite 4x4')
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(f'{label}: invalid homogeneous bottom row')
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=2e-3) or abs(np.linalg.det(r) - 1) > 2e-3:
        raise ValueError(f'{label}: rotation is not in SO(3)')


def load_gt_poses(poses_file: Path, calib_file: Path, frame_count: int) -> NDArray[np.float64]:
    """T_world_lidar via KITTI Odometry camera-0 ground truth and calibration.

    KITTI 'poses/07.txt' rows encode rectified camera-0 -> world in 3x4.
    'sequences/07/calib.txt' Tr encodes Velodyne -> rectified camera-0.
    """
    rows = np.loadtxt(poses_file, dtype=np.float64, ndmin=2)
    if rows.shape != (frame_count, 12) or not np.isfinite(rows).all():
        raise ValueError(f'KITTI GT poses must contain {frame_count} finite rows of 12 values')
    twc = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    twc[:, :3, :4] = rows.reshape(-1, 3, 4)
    for i, pose in enumerate(twc):
        _validate_transform(pose, f'KITTI camera0 pose {i}')
    tcl = kitti_gt_camera_lidar(calib_file, camera_index=0)
    _validate_transform(tcl, 'KITTI Tr (camera0<-lidar)')
    twl = twc @ tcl
    return twl


def glim_poses(traj: GlimTrajectory) -> NDArray[np.float64]:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(traj.timestamps), axis=0)
    poses[:, :3, :3] = Rotation.from_quat(traj.quaternions_xyzw).as_matrix()
    poses[:, :3, 3] = traj.positions
    return poses


def gt_selection_trajectory(twl: NDArray, times: NDArray, offset: int,
                            n_glim: int) -> GlimTrajectory:
    subset = twl[offset: offset + n_glim]
    quats = Rotation.from_matrix(subset[:, :3, :3]).as_quat()
    return GlimTrajectory(times[offset:offset + n_glim], subset[:, :3, 3], quats)


def relative_poses(world_poses: NDArray, frames: NDArray, target: int,
                   *, offset: int = 0) -> NDArray[np.float64]:
    """Per-scan T_Ltarget_Lframe, column-vector convention."""
    k = target - offset
    f = frames - offset
    if k < 0 or k >= len(world_poses) or (f < 0).any() or (f >= len(world_poses)).any():
        raise ValueError('Frame outside pose coverage')
    return np.linalg.inv(world_poses[k])[None] @ world_poses[f]


def roi_mask(xyz: NDArray, roi: NDArray) -> NDArray[np.bool_]:
    return ((xyz[:, 0] >= roi[0]) & (xyz[:, 0] < roi[1]) &
            (xyz[:, 1] >= roi[2]) & (xyz[:, 1] < roi[3]) &
            (xyz[:, 2] >= roi[4]) & (xyz[:, 2] < roi[5]))


def voxel_keys(xyz: NDArray, roi: NDArray, voxel_m: float) -> NDArray[np.int32]:
    """Unique occupied 3D voxels anchored at the SAME fixed ROI origin."""
    valid = np.asarray(xyz, dtype=np.float64)
    valid = valid[roi_mask(valid, roi)]
    if len(valid) == 0:
        return np.empty((0, 3), dtype=np.int32)
    raw_indices = np.floor((valid - roi[[0, 2, 4]]) / voxel_m)
    if (raw_indices > np.iinfo(np.int32).max).any() or (raw_indices < 0).any():
        raise ValueError('Voxel indices overflow / invalid ROI')
    return np.unique(raw_indices.astype(np.int32), axis=0)


def _row_view(keys: NDArray[np.int32]) -> NDArray:
    return np.ascontiguousarray(keys).view(np.dtype((np.void, keys.dtype.itemsize * 3))).ravel()


def occupancy_metrics(keys: NDArray, reference: NDArray) -> dict[str, float | int]:
    if not len(reference):
        raise ValueError('Oracle reference has no voxels in ROI')
    common = len(np.intersect1d(_row_view(keys), _row_view(reference), assume_unique=True))
    return {
        'occupied_voxels': int(len(keys)),
        'reference_voxels': int(len(reference)),
        'common_voxels': int(common),
        'reference_recall': float(common / len(reference)),
        'reference_precision': float(common / len(keys)) if len(keys) else 0.0,
    }


def pose_errors(estimate: NDArray, gt: NDArray, anchor_local: int) -> dict[str, float]:
    """Target-frame relative transform comparison; omit identity anchor from stats."""
    delta = np.swapaxes(gt[:, :3, :3], 1, 2) @ estimate[:, :3, :3]
    rot = np.rad2deg(Rotation.from_matrix(delta).magnitude())
    trans = np.linalg.norm(estimate[:, :3, 3] - gt[:, :3, 3], axis=1)
    mask = np.ones(len(rot), dtype=bool)
    mask[anchor_local] = False
    if not mask.any():
        raise ValueError('At least two scans are required')
    return {
        'pose_rot_mean_deg': float(np.mean(rot[mask])),
        'pose_rot_p95_deg': float(np.percentile(rot[mask], 95)),
        'pose_rot_max_deg': float(np.max(rot[mask])),
        'pose_trans_mean_m': float(np.mean(trans[mask])),
        'pose_trans_p95_m': float(np.percentile(trans[mask], 95)),
        'pose_trans_max_m': float(np.max(trans[mask])),
    }


def point_displacement(estimate_xyz: NDArray, gt_xyz: NDArray,
                       roi: NDArray) -> dict[str, float | int]:
    """Identical original returns, ROI frozen from GT, not nearest neighbors.

    This measures pose-induced displacement, NOT independent surface accuracy.
    """
    if estimate_xyz.shape != gt_xyz.shape:
        raise ValueError('Point order/count mismatch: cannot use paired displacement')
    mask = roi_mask(gt_xyz, roi)
    if not mask.any():
        return {'paired_points': 0, 'paired_rmse_m': float('nan'),
                'paired_p95_m': float('nan')}
    diff = np.asarray(estimate_xyz[mask], dtype=np.float64) - gt_xyz[mask]
    norm = np.linalg.norm(diff, axis=1)
    return {'paired_points': int(len(norm)),
            'paired_rmse_m': float(np.sqrt(np.mean(norm ** 2))),
            'paired_p95_m': float(np.percentile(norm, 95))}


def _xyz_for(scans: list, transforms: NDArray) -> NDArray[np.float64]:
    if len(scans) != len(transforms):
        raise ValueError('Transforms and scans must agree')
    return np.concatenate([scan.xyz @ mat[:3, :3].T + mat[:3, 3]
                           for scan, mat in zip(scans, transforms)], axis=0)


def build_reference(velodyne: Path, frames: NDArray, times: NDArray,
                    gt_world: NDArray, target: int, roi: NDArray,
                    voxel_m: float, min_range_m: float, max_range_m: float) -> NDArray:
    """Oracle-pose union of ALL eligible scene scans, not one strategy's subset."""
    transform = relative_poses(gt_world, frames, target)
    pieces = []
    for i, frame in enumerate(frames):
        path = velodyne / f'{int(frame):06d}.bin'
        scan = load_kitti_bin_scan(path, scan_id=0,
                                   timestamp_ns=int(round(float(times[frame]) * 1e9)),
                                   min_range_m=min_range_m, max_range_m=max_range_m)
        xyz = scan.xyz @ transform[i, :3, :3].T + transform[i, :3, 3]
        piece = voxel_keys(xyz, roi, voxel_m)
        if len(piece):
            pieces.append(piece)
    if not pieces:
        raise ValueError('No reference points in fixed ROI; inspect ROI / scan data')
    return np.unique(np.vstack(pieces), axis=0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as fh:
        cols = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--velodyne', type=Path, required=True)
    p.add_argument('--times', type=Path, required=True)
    p.add_argument('--odom', type=Path, required=True)
    p.add_argument('--poses', type=Path, required=True, help='KITTI odometry poses/07.txt (camera0 -> world)')
    p.add_argument('--calib', type=Path, required=True, help='KITTI sequences/07/calib.txt')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--window', default='012', choices=['005', '012'])
    p.add_argument('--start-s', type=float)
    p.add_argument('--stop-s', type=float)
    p.add_argument('--target-frame', type=int, default=898)
    p.add_argument('--pose-index-offset', type=int, default=0)
    p.add_argument('--half-frames', type=int, nargs='+', default=[5])
    p.add_argument('--radii-m', type=float, nargs='+', default=[5.0])
    p.add_argument('--match-scan-budget', action='store_true')
    p.add_argument('--selection-source', choices=['glim', 'gt'], default='glim',
                   help='Fixed radius uses which trajectory? Keep glim for real-policy baseline')
    p.add_argument('--refiner', choices=['none', 'joint', 'pgo', 'both'], default='none',
                   help='Run independent solvers on identical GLIM-accumulated scans')
    p.add_argument('--refine-iterations', type=int, default=4)
    p.add_argument('--refine-voxel-m', type=float, default=.16)
    p.add_argument('--refine-max-samples', type=int, default=2500)
    p.add_argument('--pgo-iterations', type=int, default=4,
                   help='PGO graph IRLS iterations (not Joint iterations)')
    p.add_argument('--pgo-pair-max-nfev', type=int, default=38,
                   help='Max evaluations in each PGO pairwise least-squares fit')
    p.add_argument('--seed', type=int, default=20261116)
    p.add_argument('--min-range-m', type=float, default=1.0)
    p.add_argument('--max-range-m', type=float, default=70.0)
    p.add_argument('--voxel-m', type=float, default=.5, help='Common ROI voxel edge in meters')
    p.add_argument('--roi', type=float, nargs=6, required=True,
                   metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX', 'ZMIN', 'ZMAX'),
                   help='Fixed ROI in TARGET LiDAR coordinates, shared by all strategies')
    p.add_argument('--verbose', action='store_true')
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(levelname)s %(message)s')
    roi = np.asarray(args.roi, dtype=np.float64)
    if not np.isfinite(roi).all() or any(roi[k] >= roi[k + 1] for k in (0, 2, 4)):
        raise ValueError('ROI needs finite xmin<xmax, ymin<ymax, zmin<zmax')
    if not np.isfinite(args.voxel_m) or args.voxel_m <= 0:
        raise ValueError('voxel-m must be finite and >0')
    if not (0 <= args.min_range_m < args.max_range_m and
            np.isfinite([args.min_range_m, args.max_range_m]).all()):
        raise ValueError('Invalid LiDAR range')
    if (args.refine_iterations < 1 or args.refine_voxel_m <= 0 or
            args.refine_max_samples < 24 or args.pgo_iterations < 1 or
            args.pgo_pair_max_nfev < 1):
        raise ValueError('Invalid Joint/PGO refinement settings')
    if not args.velodyne.is_dir():
        raise ValueError(f'No Velodyne directory: {args.velodyne}')
    times, glim, mapping = load_kitti_glim(args.times, args.odom, args.pose_index_offset)
    gt_world = load_gt_poses(args.poses, args.calib, len(times))
    glim_world = glim_poses(glim)
    selection_trajectory = (glim if args.selection_source == 'glim' else
                            gt_selection_trajectory(gt_world, times, args.pose_index_offset, len(glim.timestamps)))
    plans = prepare_experiments(args, times, selection_trajectory, args.pose_index_offset)
    if not plans:
        raise ValueError('No valid plans')
    # Use EXACTLY the same candidate scene as the selection code; the union is
    # independent of the method. Do not use an unrelated global 07 scene.
    scene_frames = np.arange(len(times), dtype=np.int64)
    scene_frames = scene_frames[(times[scene_frames] >= plans[0].start_s) &
                                 (times[scene_frames] <= plans[0].stop_s) &
                                 (scene_frames >= args.pose_index_offset) &
                                 (scene_frames < args.pose_index_offset + len(glim.timestamps))]
    if args.target_frame not in scene_frames:
        raise ValueError('Target outside reference candidate scene')
    args.out.mkdir(parents=True, exist_ok=True)
    reference = build_reference(args.velodyne, scene_frames, times, gt_world,
                                args.target_frame, roi, args.voxel_m,
                                args.min_range_m, args.max_range_m)
    np.savez_compressed(args.out / 'reference_voxels.npz', voxel_indices=reference,
                        roi=roi, voxel_size_m=args.voxel_m, scene_frames=scene_frames)
    summary: list[dict[str, Any]] = []
    per_scan: list[dict[str, Any]] = []
    for plan in plans:
        frames = np.asarray(plan.selected, dtype=np.int64)
        anchor = plan.selected.index(plan.target)
        scans = load_scans(args.velodyne, plan.selected, times, args.min_range_m, args.max_range_m)
        gt_rel = relative_poses(gt_world, frames, plan.target)
        glim_rel = relative_poses(glim_world, frames, plan.target,
                                  offset=args.pose_index_offset)
        target_stamp = int(round(float(times[plan.target]) * 1e9))
        raw = accumulate_glim(scans, glim, target_timestamp_ns=target_stamp, mode='rigid')
        oracle_xyz = _xyz_for(scans, gt_rel)
        if len(raw.xyz) != len(oracle_xyz):
            raise RuntimeError('LiDAR return count changed during accumulation')
        models = [('oracle', oracle_xyz, gt_rel), ('glim', raw.xyz, glim_rel)]
        methods = (['joint', 'pgo'] if args.refiner == 'both'
                   else [] if args.refiner == 'none' else [args.refiner])
        # Joint and PGO MUST start from the same unrefined GLIM cloud.
        # Neither solver is initialized using the other solver's output.
        for method in methods:
            pair_cfg = JointSettings(anchor_scan_id=anchor, d1_roi=False,
                                     iterations=args.refine_iterations,
                                     voxel_m=args.refine_voxel_m,
                                     max_samples_per_scan=args.refine_max_samples,
                                     seed=args.seed)
            if method == 'joint':
                result = solve_joint(raw, settings=pair_cfg)
                solver = 'spade_calib.lidar.joint_solver.solve_joint'
            else:
                pgo_cfg = PGOSettings(pairs=pair_cfg, iterations=args.pgo_iterations,
                                      pair_max_nfev=args.pgo_pair_max_nfev)
                result = solve_pgo(raw, settings=pgo_cfg)
                solver = 'spade_calib.lidar.pgo_solver.solve_pgo'
            if (result.anchor_scan_id != anchor or
                    result.transforms.shape != (len(frames), 4, 4) or
                    result.xyz.shape != raw.xyz.shape or
                    not np.isfinite(result.transforms).all() or
                    not np.allclose(result.transforms[anchor], np.eye(4), atol=1e-8)):
                raise RuntimeError(f'{method} produced inconsistent scan corrections')
            # Both solvers return T_target_new_target_GLIM per scan.
            # Do not interpret result.transforms as the raw LiDAR pose.
            refined_rel = result.transforms @ glim_rel
            models.append((method, result.xyz, refined_rel))
            np.save(args.out / f'{plan.name}_{method}_corrections.npy', result.transforms)
            (args.out / f'{plan.name}_{method}_diagnostics.json').write_text(
                json.dumps({'solver': solver, 'selected_frames': list(plan.selected),
                            'anchor_local_scan_id': anchor,
                            'iterations': list(result.iterations),
                            'initialization': 'independent_original_GLIM_accumulation',
                            'note': 'Pair residuals are training diagnostics, not GT accuracy'},
                           indent=2, allow_nan=False), encoding='utf-8')
        for stage, xyz, transforms in models:
            keys = voxel_keys(xyz, roi, args.voxel_m)
            row = {'experiment': plan.name, 'selection': plan.method,
                   'selection_parameter': plan.parameter, 'selection_pose': args.selection_source,
                   'stage': stage, 'target_frame': plan.target, 'scan_count': len(frames),
                   'original_points': int(sum(len(s.xyz) for s in scans)),
                   'points_in_roi': int(np.count_nonzero(roi_mask(xyz, roi))),
                   **occupancy_metrics(keys, reference)}
            if stage != 'oracle':
                row.update(pose_errors(transforms, gt_rel, anchor))
                row.update(point_displacement(np.asarray(xyz), oracle_xyz, roi))
            summary.append(row)
            LOG.info('%s/%s: scans=%d recall=%.3f precision=%.3f',
                     plan.name, stage, len(frames), row['reference_recall'],
                     row['reference_precision'])
            if stage != 'oracle':
                delta = np.swapaxes(gt_rel[:, :3, :3], 1, 2) @ transforms[:, :3, :3]
                angles = np.rad2deg(Rotation.from_matrix(delta).magnitude())
                td = np.linalg.norm(transforms[:, :3, 3] - gt_rel[:, :3, 3], axis=1)
                for j, f in enumerate(frames):
                    per_scan.append({'experiment': plan.name, 'stage': stage,
                                     'local_scan_id': j, 'frame': int(f),
                                     'is_target': int(j == anchor),
                                     'rotation_error_deg': float(angles[j]),
                                     'translation_error_m': float(td[j])})
    write_csv(args.out / 'summary.csv', summary)
    write_csv(args.out / 'per_scan_pose_errors.csv', per_scan)
    (args.out / 'manifest.json').write_text(json.dumps({
        'input': {k: str(getattr(args, k)) for k in ('velodyne', 'times', 'odom', 'poses', 'calib')},
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'mapping': mapping,
        'coordinate_convention': 'T_world_lidar = T_world_camera0 @ T_camera0_lidar; T_target_i = inv(T_world_target) @ T_world_i',
        'selection_source': args.selection_source,
        'reference': {'type': 'KITTI_gt_pose_scan_union', 'scene_frames': scene_frames.tolist(),
                      'voxel_count': int(len(reference)),
                      'not_surface_ground_truth': True, 'contains_evaluated_scan_points': True},
        'comparisons': [{'name': p.name, 'scan_frames': list(p.selected),
                         'method': p.method, 'parameter': p.parameter} for p in plans],
        'limitations': ['No per-return deskew', 'KITTI poses are not error-free',
                        'All candidate scans form reference (not independent holdout)',
                        'Voxel occupancy is not calibrated surface completeness',
                        'Paired point displacement measures relative pose effect, not independent geometry GT',
                        'Different scan counts require budget-matched comparisons'],
    }, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    LOG.info('Results saved: %s', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
