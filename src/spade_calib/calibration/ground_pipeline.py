"""KITTI-ready metric-depth + accumulated LiDAR -> GroundMatch -> 3/6-DoF CLI.

No GT in extraction or fitting. Explicit frame-index and KITTI scan-time
checks prevent a silently mismatched camera depth / LiDAR accumulation.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from spade_calib.camera.planes import CameraGeometry, CameraPlaneSettings, extract_camera_planes
from spade_calib.lidar.ground_planes import LidarGroundSettings, extract_lidar_ground
from spade_calib.correspondence.ground_matcher import GroundMatchingSettings, match_ground


def _atomic_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent,
            prefix='.ground_pipeline_', suffix='.tmp', delete=False,
        ) as f:
            temporary = Path(f.name)
            json.dump(doc, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_frame_cloud(path: Path, frame_index: int, times_path: Path
                      ) -> tuple[np.ndarray, dict]:
    from spade_calib.io.kitti import read_kitti_times_ns
    with np.load(path, allow_pickle=False) as archive:
        required = ('xyz', 'frame_id', 'reference_timestamp_ns',
                    'frame_ids', 'motion_model')
        for key in required:
            if key not in archive.files:
                raise ValueError(f'Accumulated KITTI NPZ missing {key}')
        frame_id = str(archive['frame_id'].item())
        if frame_id != f'kitti_lidar_frame_{frame_index}':
            raise ValueError(f'Cloud frame {frame_id!r} != requested target {frame_index}')
        ref_ts = int(archive['reference_timestamp_ns'].item())
        source_frames = np.asarray(archive['frame_ids'], dtype=int)
        model = str(archive['motion_model'].item())
        xyz = np.asarray(archive['xyz'], dtype=np.float64)
    times = read_kitti_times_ns(times_path)
    if not (0 <= frame_index < len(times)):
        raise ValueError('KITTI camera/target frame index outside times.txt')
    if ref_ts != int(times[frame_index]):
        raise ValueError('LiDAR reference timestamp differs from KITTI times.txt for target frame')
    if not len(source_frames) or not np.all(np.diff(source_frames) > 0):
        raise ValueError('Accumulation source frame metadata invalid')
    if model != 'rigid':
        raise ValueError('This initial KITTI pipeline supports rigid accumulation only')
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Accumulation contains invalid points')
    return xyz, {
        'target_lidar_frame': frame_id, 'target_timestamp_ns': ref_ts,
        'source_frames': source_frames.tolist(),
        'source_frame_count': len(source_frames), 'point_count': len(xyz),
        'motion_model': model,
        'deskew_applied': False,
    }


def prepare_ground_inputs(args: argparse.Namespace) -> tuple[dict | None, dict]:
    if args.frame_index < 0 or args.depth_frame_index < 0:
        raise ValueError('Frame indices must be nonnegative')
    if args.depth_frame_index != args.frame_index:
        raise ValueError('Camera depth and LiDAR accumulation must refer to the SAME frame')
    geometry = CameraGeometry.from_json(args.camera_json)
    xyz, lidar_meta = _load_frame_cloud(args.cloud, args.frame_index, args.kitti_times)
    if Path(args.metric_depth).suffix.lower() != '.npy':
        raise ValueError('Metric depth must be a dense .npy camera-Z array')
    depth = np.load(args.metric_depth, allow_pickle=False)
    if depth.ndim != 2 or depth.shape != (geometry.height, geometry.width):
        raise ValueError('Metric depth must be HxW matching CameraGeometry')
    with args.handeye.open('r', encoding='utf-8') as f:
        handeye_doc = json.load(f)
    h = handeye_doc.get('handeye', {})
    if h.get('convention') != 'T_camera_lidar (multiply lidar point to camera frame)':
        raise ValueError('Hand-Eye must be a LiDAR->Camera estimation JSON')
    T0 = np.asarray(h.get('T_camera_lidar'), dtype=np.float64)
    if T0.shape != (4, 4) or not np.isfinite(T0).all():
        raise ValueError('Invalid saved Hand-Eye transform')
    if abs(np.linalg.det(T0[:3, :3]) - 1) > 1e-3:
        raise ValueError('Saved Hand-Eye transform has invalid rotation')

    cam_settings = CameraPlaneSettings(
        processing_width=args.processing_width, processing_height=args.processing_height,
        min_depth_m=args.min_depth_m, max_depth_m=args.max_depth_m)
    lidar_settings = LidarGroundSettings(
        max_sampled_points=args.max_lidar_points, iterations=args.lidar_ransac_iterations,
        min_inliers=args.min_lidar_inliers, min_height_m=args.min_ground_height_m,
        max_height_m=args.max_ground_height_m)
    match_settings = GroundMatchingSettings(
        mask_dilation_grid_px=args.mask_dilation_px,
        min_guided_points=args.min_guided_points,
    )
    camera_planes, cam_info = extract_camera_planes(depth, geometry, cam_settings)
    lidar_ground, lidar_info = extract_lidar_ground(xyz, lidar_settings)
    match, match_info = match_ground(camera_planes, lidar_ground, depth, geometry,
                                    T0, match_settings)
    diag = {
        'schema_version': 1, 'kind': 'auto_metric_ground_correspondence_diagnostic',
        'camera': cam_info, 'lidar': lidar_info, 'matching': match_info,
        'target_frame_index': args.frame_index, 'depth_frame_index': args.depth_frame_index,
        'frame_alignment': lidar_meta,
        'source': {
            'metric_depth': str(args.metric_depth.resolve()),
            'camera_intrinsics': str(args.camera_json.resolve()),
            'accumulated_lidar': str(args.cloud.resolve()),
            'handeye': str(args.handeye.resolve()),
            'kitti_times': str(args.kitti_times.resolve()),
        },
        'camera_ground_candidate_count': len(camera_planes),
        'accepted_ground_hypotheses': 1 if match is not None else 0,
        'gt_used_for_matching': False,
        'warning': (
            'AUTO-GENERATED PROVISIONAL correspondence: geometric matching '
            'does NOT establish independent physical correspondence truth. '
            'Depth must already be calibrated metric Camera optical Z.'
        ),
    }
    if match is None:
        return None, diag
    payload = {
        'schema_version': 1,
        'convention': 'T_camera_lidar',
        'source_type': 'automatic_provisional_camera_guided_ground',
        'verified_by_independent_review': False,
        'target_frame_index': args.frame_index,
        'matches': [{
            'id': match.match_id,
            'lidar': {
                'frame': 'lidar_target',
                'normal': match.lidar.normal.tolist(),
                'offset_m': match.lidar.offset_m,
            },
            'camera': {
                'frame': 'camera',
                'normal': match.camera.normal.tolist(),
                'offset_m': match.camera.offset_m,
            },
            'sigma_normal_deg': match.sigma_normal_deg,
            'sigma_offset_m': match.sigma_offset_m,
        }],
    }
    return payload, diag


def run_prepare_ground(args: argparse.Namespace) -> int:
    if args.run_refinement and not args.allow_provisional:
        raise ValueError('--run-refinement requires --allow-provisional (automatic matches are not GT)')
    if args.run_refinement and (args.glim is None or args.sfm is None):
        raise ValueError('--run-refinement requires --glim and --sfm')
    out = args.output_dir.expanduser().resolve()
    ground, diag = prepare_ground_inputs(args)
    _atomic_json(out / 'ground_diagnostics.json', diag)
    if ground is None:
        raise ValueError(f'No Ground Plane match passed gates; diagnostics at {out / "ground_diagnostics.json"}')
    matches_file = out / 'ground_matches.json'
    _atomic_json(matches_file, ground)
    result = {
        'ground_matches': str(matches_file),
        'diagnostics': str(out / 'ground_diagnostics.json'),
        'matching_status': 'provisional_requires_independent_review',
        'run_refinement': bool(args.run_refinement),
    }
    if args.run_refinement:
        from spade_calib.calibration.cli import run_ground_refinement
        subset = argparse.Namespace(
            handeye=args.handeye, ground_matches=matches_file,
            glim=args.glim, sfm=args.sfm,
            kitti_times=args.kitti_times, glim_first_frame=args.glim_first_frame,
            image_timestamps=None, camera_to_lidar_offset_ns=0,
            handeye_weight=args.handeye_weight, ground_weight=args.ground_weight,
            max_translation_delta_m=args.max_translation_delta_m,
            max_rotation_delta_deg=args.max_rotation_delta_deg,
            max_initial_normal_angle_deg=args.max_initial_normal_angle_deg,
            robust_loss=args.robust_loss, calib=args.calib,
            camera_index=args.camera_index, output=out / 'ground_refinement.json',
        )
        run_ground_refinement(subset)
        result['ground_refinement'] = str(subset.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
