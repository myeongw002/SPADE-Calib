"""CLI for common-input, fixed-scale Hand-Eye + Ground 3-DoF/6-DoF comparison."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from spade_calib.calibration.ground import load_ground_matches
from spade_calib.calibration.solver import GroundRefinementSettings, compare_modes
from spade_calib.selection.trajectory import SelectionSettings, build_motion_pairs


def run_ground_refinement(args: Any) -> int:
    with Path(args.handeye).open('r', encoding='utf-8') as handle:
        report = json.load(handle)
    handeye = report.get('handeye')
    if not isinstance(handeye, dict) or 'T_camera_lidar' not in handeye:
        raise ValueError('Expected a spade-calib handeye JSON with T_camera_lidar')
    if handeye.get('convention') != 'T_camera_lidar (multiply lidar point to camera frame)':
        raise ValueError('Hand-Eye must use the explicit LiDAR-to-Camera convention')
    parameters = handeye.get('settings', {})
    scale = float(handeye['sfm_scale_to_metric'])
    initial = np.asarray(handeye['T_camera_lidar'], dtype=float)

    if args.kitti_times is not None:
        if args.image_timestamps is not None or args.camera_to_lidar_offset_ns != 0:
            raise ValueError('KITTI frame matching cannot use timestamp CSV or clock offset')
        from spade_calib.io.kitti import load_kitti_trajectory_data
        samples = load_kitti_trajectory_data(
            args.glim, args.sfm, args.kitti_times,
            glim_first_frame=args.glim_first_frame,
        )
    else:
        if args.glim_first_frame != 0:
            raise ValueError('--glim-first-frame requires --kitti-times')
        from spade_calib.selection.trajectory import load_trajectory_data
        samples = load_trajectory_data(
            args.glim, args.sfm,
            camera_to_lidar_offset_ns=args.camera_to_lidar_offset_ns,
            image_timestamps_csv=args.image_timestamps,
        )

    pair_settings = SelectionSettings(
        pair_baseline_s=float(parameters.get('pair_baseline_s', 2.0)),
        pair_stride_s=float(parameters.get('pair_stride_s', 1.0)),
        pair_tolerance_s=float(parameters.get('pair_tolerance_s', 0.12)),
    )
    pairs = build_motion_pairs(samples, int(samples.timestamps_ns[0]),
                               int(samples.timestamps_ns[-1]), pair_settings)
    saved_count = int(handeye.get('number_of_motion_pairs', -1))
    if pairs.count != saved_count:
        raise ValueError(f'Hand-Eye motion pairs changed: saved {saved_count}, '
                         f'rebuilt {pairs.count}; check input trajectories/timestamps')
    matches = load_ground_matches(args.ground_matches)
    sigma_rot = float(parameters.get('sigma_relative_rotation_deg', 0.5))
    sigma_trans = float(parameters.get('sigma_relative_translation_m', 0.05))
    settings = GroundRefinementSettings(
        sigma_relative_rotation_deg=sigma_rot,
        sigma_relative_translation_m=sigma_trans,
        handeye_weight=args.handeye_weight,
        ground_weight=args.ground_weight,
        max_translation_delta_m=args.max_translation_delta_m,
        max_rotation_delta_deg=args.max_rotation_delta_deg,
        max_initial_normal_angle_deg=args.max_initial_normal_angle_deg,
        robust_loss=args.robust_loss,
    )
    result = compare_modes(initial, scale, pairs, matches, settings=settings)
    result['source'] = {
        'handeye_report': str(Path(args.handeye).resolve()),
        'glim_trajectory': str(Path(args.glim).resolve()),
        'sfm_trajectory': str(Path(args.sfm).resolve()),
        'ground_matches': str(Path(args.ground_matches).resolve()),
        'motion_pairs_rebuilt_equal_count': True,
        'motion_pairs_rebuilt_equal_content': False,
        'note': 'Only pair count is checked; preserve source hashes to ensure identical inputs.',
    }
    result['input_ground_correspondence_count'] = len(matches)
    result['gt_used_for_fitting'] = False
    if args.calib is not None:
        from spade_calib.evaluation.kitti import evaluate_camera_lidar, kitti_gt_camera_lidar
        gt = kitti_gt_camera_lidar(args.calib, camera_index=args.camera_index)
        result['kitti_evaluation_only'] = {
            'baseline': evaluate_camera_lidar(initial, gt),
            'fixed_rotation': evaluate_camera_lidar(
                np.asarray(result['fixed_rotation']['T_camera_lidar']), gt),
            'optimize_rotation': evaluate_camera_lidar(
                np.asarray(result['optimize_rotation']['T_camera_lidar']), gt),
            'camera_index': args.camera_index,
            'gt_source': str(Path(args.calib).resolve()),
            'gt_used_for_fitting': False,
        }
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=destination.parent,
                                         prefix='.ground_', suffix='.tmp',
                                         delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({
        'output': str(destination), 'motion_pairs': pairs.count,
        'ground_matches': len(matches),
        'fixed_rotation_converged': result['fixed_rotation']['optimizer']['success'],
        'optimize_rotation_converged': result['optimize_rotation']['optimizer']['success'],
        'gt_evaluated': args.calib is not None,
    }, indent=2))
    return 0
