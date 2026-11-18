"""KITTI-indexed SfM-point / finite-LiDAR-plane prototype CLI adapters."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from spade_calib.correspondence.sfm_lidar_planes import (
    MatchSettings, PatchSettings, SfMPlaneMatches, extract_lidar_patches,
    match_sfm_to_patches,
)


def _read_handeye(path: Path) -> tuple[np.ndarray, float, dict]:
    doc = json.loads(path.read_text(encoding='utf-8'))
    h = doc.get('handeye', {})
    if h.get('convention') != 'T_camera_lidar (multiply lidar point to camera frame)':
        raise ValueError('Hand-Eye JSON must use T_camera_lidar convention')
    T = np.asarray(h['T_camera_lidar'], dtype=float)
    scale = float(h['sfm_scale_to_metric'])
    if (T.shape != (4, 4) or not np.isfinite(T).all() or
            not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8) or
            not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5) or
            abs(np.linalg.det(T[:3, :3])-1) > 1e-5 or
            not np.isfinite(scale) or scale <= 0):
        raise ValueError('Invalid Hand-Eye transform / SfM scale')
    return T, scale, doc


def _load_target_cloud(path: Path, frame_index: int, times_path: Path
                       ) -> tuple[np.ndarray, np.ndarray, dict]:
    from spade_calib.io.kitti import read_kitti_times_ns

    with np.load(path, allow_pickle=False) as cloud:
        required = {'xyz', 'scan_id', 'frame_ids', 'reference_timestamp_ns', 'frame_id'}
        if not required.issubset(cloud.files):
            raise ValueError('Cloud must be accumulate-kitti NPZ with target frame metadata')
        xyz = np.asarray(cloud['xyz'], dtype=np.float64)
        scan_ids = np.asarray(cloud['scan_id'], dtype=np.int64)
        frames = np.asarray(cloud['frame_ids'], dtype=np.int64)
        source_frame = str(cloud['frame_id'].item())
        stamp = int(cloud['reference_timestamp_ns'].item())
        mode = str(cloud['motion_model'].item()) if 'motion_model' in cloud else 'unknown'
    allowed = {f'kitti_lidar_frame_{frame_index}', 'lidar_target'}
    if source_frame not in allowed or frame_index not in frames:
        raise ValueError('Accumulated LiDAR cloud frame does not match target camera frame')
    if xyz.ndim != 2 or xyz.shape[1] != 3 or scan_ids.shape != (len(xyz),):
        raise ValueError('Invalid accumulated point shape or scan_id')
    if mode != 'rigid':
        raise ValueError('This KITTI prototype accepts rigid accumulation only')
    times = read_kitti_times_ns(times_path)
    if frame_index >= len(times) or stamp != int(times[frame_index]):
        raise ValueError('Accumulated point timestamp differs from KITTI frame time')
    return xyz, scan_ids, {'source_frame': source_frame, 'reference_timestamp_ns': stamp,
                           'scan_count': int(len(frames)), 'point_count': int(len(xyz)),
                           'motion_model': mode}


def run_prepare_sfm_plane_matches(args) -> int:
    """Read COLMAP tracked points and generate unverified finite-patch associations."""
    from spade_calib.camera.planes import CameraGeometry
    from spade_calib.camera.sfm_anchors import build_sfm_metric_anchors

    if args.frame_index < 0 or Path(args.image_name).stem != f'{args.frame_index:06d}':
        raise ValueError('image-name must be the same KITTI frame as frame-index')
    if args.output.suffix != '.npz':
        raise ValueError('--output must end in .npz')
    T0, scale, _ = _read_handeye(args.handeye)
    camera = CameraGeometry.from_json(args.camera_json)
    fields, anchor_info = build_sfm_metric_anchors(
        args.colmap_model, args.image_name, camera, scale,
        max_reprojection_error_px=args.max_reprojection_error_px,
        min_track_length=args.min_track_length,
    )
    # SfM anchors already encode scaled optical Z; undo Hand-Eye scale once
    # to keep the joint optimizer's scale parameter identifiable.
    undistorted = cv2.undistortPoints(fields['xy'].reshape(-1, 1, 2),
                                      camera.K, camera.distortion)[:, 0, :]
    rays = np.column_stack((undistorted, np.ones(len(undistorted))))
    xyz_cam_sfm = rays * (fields['depth_m']/scale)[:, None]
    cloud, ids, cloud_meta = _load_target_cloud(args.cloud, args.frame_index,
                                               args.kitti_times)
    patch_settings = PatchSettings(
        voxel_size_m=args.voxel_m, max_sampled_points=args.max_lidar_points,
        max_planes=args.max_planes, ransac_iterations=args.ransac_iterations,
        distance_threshold_m=args.plane_threshold_m,
        min_inliers=args.min_plane_inliers,
        connectivity_radius_m=args.connectivity_radius_m,
    )
    patches, patch_info = extract_lidar_patches(cloud, ids, patch_settings)
    matching_settings = MatchSettings(
        max_normal_distance_m=args.max_normal_distance_m,
        max_support_distance_m=args.max_support_distance_m,
        min_matches_per_patch=args.min_matches_per_patch,
        max_matches_per_patch=args.max_matches_per_patch,
    )
    matches, match_info = match_sfm_to_patches(
        xyz_cam_sfm, fields['xy'], patches, T0, scale, matching_settings)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if matches is None:
        # Deliberately do not emit an empty, apparently usable match file.
        raise ValueError('No reliable candidate matches. Inspect point/plane diagnostics '
                         f'(patches={len(patches)}): {match_info}')
    np.savez_compressed(
        output, camera_xyz_sfm=matches.camera_xyz_sfm,
        feature_xy=matches.feature_xy, normal_lidar=matches.normal_lidar,
        offset_lidar_m=matches.offset_lidar_m, patch_id=matches.patch_id,
        feature_index=matches.feature_index,
    )
    report = {
        'schema_version': 1, 'kind': 'sfm_point_finite_lidar_patch_matches',
        'status': 'automatic_provisional_not_ground_truth',
        'verified_by_independent_review': False,
        'frame_index': args.frame_index, 'image_name': args.image_name,
        'frame_convention': 'T_camera_lidar; SfM points in camera axes UNSCALED',
        'sfm_scale_to_metric_at_matching': scale,
        'source_handeye': str(args.handeye.resolve()),
        'source_colmap_model': str(args.colmap_model.resolve()),
        'source_lidar_cloud': str(args.cloud.resolve()),
        'source_kitti_times': str(args.kitti_times.resolve()),
        'camera_anchors': anchor_info, 'cloud': cloud_meta,
        'patch_extraction': patch_info, 'patch_settings': asdict(patch_settings),
        'patches': [{'id': p.patch_id, 'point_count': len(p.support_xyz_lidar),
                     'normal_lidar': p.normal.tolist(), 'offset_m': p.offset_m,
                     'rmse_m': p.rmse_m, 'contributing_scan_ids': p.scan_ids.tolist()}
                    for p in patches],
        'matching': match_info, 'match_settings': asdict(matching_settings),
        'gt_used_for_matching': False,
        'warning': 'Automatically associated finite patches are not independently verified '
                   'and may be incorrect under an inaccurate initial extrinsic.',
    }
    output.with_suffix('.json').write_text(
        json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'matches': str(output), 'report': str(output.with_suffix('.json')),
                      'matched_points': matches.count,
                      'matched_patches': match_info['matched_patch_count'],
                      'provisional': True}, indent=2))
    return 0


def _read_matches(path: Path) -> SfMPlaneMatches:
    with np.load(path, allow_pickle=False) as f:
        required = ('camera_xyz_sfm', 'feature_xy', 'normal_lidar',
                    'offset_lidar_m', 'patch_id', 'feature_index')
        if any(key not in f.files for key in required):
            raise ValueError('SfM point-plane NPZ missing required arrays')
        m = SfMPlaneMatches(*(np.asarray(f[k]).copy() for k in required))
    m.validate()
    return m


def run_refine_sfm_planes(args) -> int:
    from spade_calib.calibration.sfm_plane_solver import (
        SfMPlaneRefinementSettings, refine_joint_7dof,
    )
    from spade_calib.io.kitti import load_kitti_trajectory_data
    from spade_calib.selection.trajectory import SelectionSettings, build_motion_pairs

    meta_path = args.matches.with_suffix('.json')
    doc = json.loads(meta_path.read_text(encoding='utf-8'))
    if doc.get('schema_version') != 1 or doc.get('kind') != 'sfm_point_finite_lidar_patch_matches':
        raise ValueError('Unexpected match report format')
    if doc.get('status') != 'automatic_provisional_not_ground_truth':
        raise ValueError('Unexpected correspondence status')
    if not args.allow_provisional:
        raise ValueError('Generated matches are PROVISIONAL; explicitly pass '
                         '--allow-provisional to run research-only refinement')
    T0, scale, he_doc = _read_handeye(args.handeye)
    if (not np.isclose(float(doc['sfm_scale_to_metric_at_matching']), scale,
                       rtol=1e-5, atol=0) or
            args.handeye.resolve() != Path(doc['source_handeye']).resolve()):
        raise ValueError('Hand-Eye source / initial scale mismatch from match preparation')
    from spade_calib.io.colmap import resolve_colmap_images
    if resolve_colmap_images(Path(doc['source_colmap_model'])) != resolve_colmap_images(args.sfm):
        raise ValueError('Match preparation and Hand-Eye refinement must use the SAME SfM model')
    matches = _read_matches(args.matches)
    samples = load_kitti_trajectory_data(
        args.glim, args.sfm, args.kitti_times, glim_first_frame=args.glim_first_frame)
    if doc['image_name'] not in samples.image_names and not any(
            Path(n).name == Path(doc['image_name']).name for n in samples.image_names):
        raise ValueError('Match image is absent from registered Hand-Eye trajectory')
    hsettings = he_doc['handeye']['settings']
    ssettings = SelectionSettings(
        pair_baseline_s=float(hsettings['pair_baseline_s']),
        pair_stride_s=float(hsettings['pair_stride_s']),
        pair_tolerance_s=float(hsettings['pair_tolerance_s']),
    )
    motion_pairs = build_motion_pairs(
        samples, int(samples.timestamps_ns[0]), int(samples.timestamps_ns[-1]), ssettings)
    settings = SfMPlaneRefinementSettings(
        sigma_motion_rotation_deg=float(hsettings['sigma_relative_rotation_deg']),
        sigma_motion_translation_m=float(hsettings['sigma_relative_translation_m']),
        sigma_point_plane_m=args.sigma_plane_m, motion_weight=args.motion_weight,
        plane_weight=args.plane_weight, max_rotation_delta_deg=args.max_rotation_delta_deg,
        max_translation_delta_m=args.max_translation_delta_m,
        max_log_scale_delta=args.max_log_scale_delta, robust_loss=args.robust_loss,
    )
    result = refine_joint_7dof(T0, scale, motion_pairs, matches, settings)
    result['source'] = {'handeye': str(args.handeye.resolve()),
                        'matches': str(args.matches.resolve()),
                        'glim': str(args.glim.resolve()),
                        'sfm': str(args.sfm.resolve()),
                        'kitti_times': str(args.kitti_times.resolve())}
    result['settings'] = asdict(settings)
    result['handeye'] = {'convention': 'T_camera_lidar (multiply lidar point to camera frame)',
                         'T_camera_lidar': result['T_camera_lidar'],
                         'sfm_scale_to_metric': result['sfm_scale_to_metric']}
    # Compatibility with evaluate-kitti-handeye; calibration GT is NOT loaded here.
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'success': result['optimizer']['success'],
                      'matched_points': matches.count,
                      'full_local_rank': result['local_observability']['full_rank'],
                      'any_bound_hit': result['optimizer']['any_bound_hit']}, indent=2))
    return 0 if result['optimizer']['success'] else 2