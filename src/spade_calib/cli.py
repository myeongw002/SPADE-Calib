"""SPADE-Calib CLI: legacy inspect, recording index, scene selection and Hand-Eye."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from spade_calib.io.dataset import inspect_dataset, load_dataset_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='spade-calib')
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect', help='Validate an extracted legacy dataset')
    inspect.add_argument('--config', type=Path, required=True)
    inspect.add_argument('--output', type=Path)
    inspect.add_argument('--no-hashes', action='store_true')
    index = sub.add_parser('index', help='Index a full recording without loading all point clouds')
    index.add_argument('--config', type=Path, required=True)
    index.add_argument('--output', type=Path, required=True)
    select = sub.add_parser('select-scenes', help='Trajectory observability and scene selection')
    source = select.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', type=Path, help='Full recording index config (schema v2)')
    source.add_argument('--glim', type=Path, help='GLIM trajectory TXT (trajectory-only archival mode)')
    select.add_argument('--sfm', type=Path, help='COLMAP images.bin/images.txt/camera_poses.csv, needed with --glim')
    select.add_argument('--camera-to-lidar-offset-ns', type=int, default=0,
                        help='Explicit image-clock to GLIM-clock offset, with --glim only')
    select.add_argument('--image-timestamps', type=Path,
                        help='CSV file,filename + timestamp_ns for custom non-timestamped images')
    select.add_argument('--kitti-times', type=Path,
                        help='KITTI Odometry times.txt; pair poses by frame index rather than clock')
    select.add_argument('--glim-first-frame', type=int, default=0,
                        help='Frame number corresponding to first GLIM pose in KITTI frame mode')
    select.add_argument('--output', type=Path, required=True)
    select.add_argument('--window-s', type=float, default=30.0)
    select.add_argument('--window-stride-s', type=float, default=15.0)
    select.add_argument('--pair-baseline-s', type=float, default=2.0)
    select.add_argument('--pair-stride-s', type=float, default=1.0)
    select.add_argument('--min-pairs', type=int, default=10)
    select.add_argument('--min-lidar-scans', type=int, default=10)
    select.add_argument('--max-lidar-gap-s', type=float, default=1.5)
    select.add_argument('--regularizer', type=float, default=1e-3)
    select.add_argument('--max-scenes', type=int, default=4)
    handeye = sub.add_parser('handeye', help='Full-trajectory 7-DoF Hand-Eye and uncertainty')
    source = handeye.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', type=str, help='Full recording schema-version 2 YAML')
    source.add_argument('--glim', type=str, help='GLIM trajectory TXT (archived trajectory-only mode)')
    handeye.add_argument('--sfm', type=str, help='COLMAP images file or pycolmap camera_poses.csv')
    handeye.add_argument('--camera-to-lidar-offset-ns', type=int, default=0)
    handeye.add_argument('--image-timestamps', type=Path,
                         help='CSV file,filename + timestamp_ns for custom image names')
    handeye.add_argument('--kitti-times', type=Path,
                         help='KITTI Odometry times.txt; pair by frame index')
    handeye.add_argument('--glim-first-frame', type=int, default=0,
                         help='Frame corresponding to the first GLIM pose in KITTI mode')
    handeye.add_argument('--output', type=str, required=True)
    handeye.add_argument('--pair-baseline-s', type=float, default=2.0)
    handeye.add_argument('--pair-stride-s', type=float, default=1.0)
    handeye.add_argument('--pair-tolerance-s', type=float, default=0.12)
    handeye.add_argument('--sigma-relative-rotation-deg', type=float, default=0.5,
                         help='ASSUMED relative-motion noise, not measured sensor sigma')
    handeye.add_argument('--sigma-relative-translation-m', type=float, default=0.05,
                         help='ASSUMED relative-motion noise, not measured sensor sigma')
    handeye.add_argument('--robust-loss', choices=('linear', 'soft_l1', 'huber', 'cauchy'),
                         default='soft_l1')
    handeye.add_argument('--bootstrap-resamples', type=int, default=0)
    handeye.add_argument('--bootstrap-block-s', type=float, default=20.0)
    handeye.add_argument('--bootstrap-seed', type=int, default=20261110)
    evaluation = sub.add_parser('evaluate-kitti-handeye',
                                help='Evaluate saved T_camera_lidar against KITTI Odometry GT')
    evaluation.add_argument('--result', type=Path, required=True,
                            help='JSON output of spade-calib handeye')
    evaluation.add_argument('--calib', type=Path, required=True,
                            help='KITTI Odometry sequence calib.txt; evaluation only')
    evaluation.add_argument('--camera-index', type=int, choices=range(4), default=2)
    evaluation.add_argument('--output', type=Path)
    ground = sub.add_parser('refine-ground',
        help='Compare 3-DoF fixed-R and 6-DoF joint Hand-Eye + Ground constraints')
    ground.add_argument('--handeye', required=True, type=Path,
                        help='Saved full-trajectory handeye.json')
    ground.add_argument('--ground-matches', required=True, type=Path,
                        help='Verified metric camera/LiDAR ground plane pairs JSON')
    ground.add_argument('--glim', required=True, type=Path,
                        help='Same GLIM trajectory used for --handeye')
    ground.add_argument('--sfm', required=True, type=Path,
                        help='Same COLMAP/pycolmap camera trajectory used for --handeye')
    ground.add_argument('--kitti-times', type=Path,
                        help='KITTI Odometry times.txt: match by frame index')
    ground.add_argument('--glim-first-frame', type=int, default=0)
    ground.add_argument('--image-timestamps', type=Path)
    ground.add_argument('--camera-to-lidar-offset-ns', type=int, default=0)
    ground.add_argument('--handeye-weight', type=float, default=1.0)
    ground.add_argument('--ground-weight', type=float, default=1.0)
    ground.add_argument('--max-translation-delta-m', type=float, default=2.0)
    ground.add_argument('--max-rotation-delta-deg', type=float, default=10.0)
    ground.add_argument('--max-initial-normal-angle-deg', type=float, default=45.0)
    ground.add_argument('--robust-loss', choices=('linear','soft_l1','huber','cauchy'),
                        default='soft_l1')
    ground.add_argument('--calib', type=Path,
                        help='Optional KITTI GT calib.txt, evaluation ONLY after fitting')
    ground.add_argument('--camera-index', type=int, choices=range(4), default=2)
    ground.add_argument('--output', required=True, type=Path)
    da2 = sub.add_parser('infer-depth',
        help='Run official Depth Anything V2 relative model; preserve raw float32 HxW')
    da2.add_argument('--image', type=Path, required=True)
    da2.add_argument('--checkpoint', type=Path, required=True)
    da2.add_argument('--da2-repo', type=Path,
                     help='Trusted clone of official Depth-Anything-V2 Python source')
    da2.add_argument('--encoder', choices=('vits','vitb','vitl','vitg'), default='vitl')
    da2.add_argument('--input-size', type=int, default=518)
    da2.add_argument('--device', choices=('auto','cpu','cuda','mps'), default='auto')
    da2.add_argument('--output', type=Path, required=True,
                     help='Raw relative inverse-depth proxy .npy; NOT metric Camera Z')

    align = sub.add_parser('align-depth',
        help='Align DA2 relative inverse-depth to SfM Camera Z using training anchors')
    align.add_argument('--relative-depth', type=Path, required=True)
    align.add_argument('--handeye', type=Path, required=True,
                       help='Saved Hand-Eye report whose SfM scale is metric')
    anchor_source = align.add_mutually_exclusive_group(required=True)
    anchor_source.add_argument('--anchors', type=Path,
                              help='Precomputed SfM metric anchor NPZ')
    anchor_source.add_argument('--colmap-model', type=Path,
                              help='Original sparse COLMAP reconstruction with images/points3D')
    align.add_argument('--image-name', type=str,
                       help='Target Camera image filename in COLMAP model')
    align.add_argument('--camera-json', type=Path,
                       help='CameraGeometry JSON with same K/rectification as COLMAP/image')
    align.add_argument('--max-reprojection-error-px', type=float, default=3.)
    align.add_argument('--min-train-anchors', type=int, default=40)
    align.add_argument('--min-holdout-anchors', type=int, default=10)
    align.add_argument('--max-holdout-median-relative-error', type=float, default=.30)
    align.add_argument('--output', type=Path, required=True,
                       help='Metric optical-Z depth .npy for prepare-ground')
    prepare_ground = sub.add_parser('prepare-ground',
        help='Extract metric Camera/LiDAR Ground match and optionally run 3/6-DoF refinement')
    prepare_ground.add_argument('--handeye', type=Path, required=True)
    prepare_ground.add_argument('--cloud', type=Path, required=True,
                                help='accumulate-kitti NPZ in target-frame LiDAR axes')
    prepare_ground.add_argument('--metric-depth', type=Path, required=True,
                                help='Camera-aligned metric optical-Z depth .npy (NOT raw DA2)')
    prepare_ground.add_argument('--camera-json', type=Path, required=True,
                                help='K, distortion, image width/height, depth_type=camera_z_m')
    prepare_ground.add_argument('--frame-index', type=int, required=True,
                                help='Target KITTI LiDAR frame')
    prepare_ground.add_argument('--depth-frame-index', type=int, required=True,
                                help='KITTI Camera image index used for the metric depth')
    prepare_ground.add_argument('--kitti-times', type=Path, required=True,
                                help='KITTI Odometry times.txt to verify accumulation timestamp')
    prepare_ground.add_argument('--output-dir', type=Path, required=True)
    prepare_ground.add_argument('--processing-width', type=int, default=320)
    prepare_ground.add_argument('--processing-height', type=int, default=180)
    prepare_ground.add_argument('--min-depth-m', type=float, default=1.)
    prepare_ground.add_argument('--max-depth-m', type=float, default=55.)
    prepare_ground.add_argument('--max-lidar-points', type=int, default=60000)
    prepare_ground.add_argument('--lidar-ransac-iterations', type=int, default=420)
    prepare_ground.add_argument('--min-lidar-inliers', type=int, default=180)
    prepare_ground.add_argument('--min-ground-height-m', type=float, default=.35)
    prepare_ground.add_argument('--max-ground-height-m', type=float, default=3.5)
    prepare_ground.add_argument('--mask-dilation-px', type=int, default=10)
    prepare_ground.add_argument('--min-guided-points', type=int, default=35)
    prepare_ground.add_argument('--run-refinement', action='store_true',
                                help='Use auto-generated PROVISIONAL matches in 3/6-DoF optimizer')
    prepare_ground.add_argument('--allow-provisional', action='store_true',
                                help='Explicitly acknowledge generated matches are not verified GT')
    prepare_ground.add_argument('--glim', type=Path)
    prepare_ground.add_argument('--sfm', type=Path)
    prepare_ground.add_argument('--glim-first-frame', type=int, default=0)
    prepare_ground.add_argument('--calib', type=Path,
                                help='Optional KITTI GT calib.txt: evaluation AFTER fitting only')
    prepare_ground.add_argument('--camera-index', type=int, choices=range(4), default=2)
    prepare_ground.add_argument('--handeye-weight', type=float, default=1.)
    prepare_ground.add_argument('--ground-weight', type=float, default=1.)
    prepare_ground.add_argument('--max-translation-delta-m', type=float, default=2.)
    prepare_ground.add_argument('--max-rotation-delta-deg', type=float, default=10.)
    prepare_ground.add_argument('--max-initial-normal-angle-deg', type=float, default=45.)
    prepare_ground.add_argument('--robust-loss', choices=('linear','soft_l1','huber','cauchy'),
                                default='soft_l1')
    sfmp = sub.add_parser('prepare-sfm-planes',
        help='Propose COLMAP SfM points -> finite LiDAR plane patches, without DA2')
    sfmp.add_argument('--handeye', type=Path, required=True)
    sfmp.add_argument('--colmap-model', type=Path, required=True,
                      help='Real COLMAP sparse model containing 3D tracks')
    sfmp.add_argument('--image-name', type=str, required=True)
    sfmp.add_argument('--frame-index', type=int, required=True)
    sfmp.add_argument('--camera-json', type=Path, required=True)
    sfmp.add_argument('--cloud', type=Path, required=True,
                      help='accumulate-kitti NPZ in the target LiDAR sensor frame')
    sfmp.add_argument('--kitti-times', type=Path, required=True)
    sfmp.add_argument('--max-reprojection-error-px', type=float, default=3.0)
    sfmp.add_argument('--min-track-length', type=int, default=2)
    sfmp.add_argument('--voxel-m', type=float, default=0.18)
    sfmp.add_argument('--max-lidar-points', type=int, default=30000)
    sfmp.add_argument('--max-planes', type=int, default=8)
    sfmp.add_argument('--ransac-iterations', type=int, default=180)
    sfmp.add_argument('--plane-threshold-m', type=float, default=0.14)
    sfmp.add_argument('--min-plane-inliers', type=int, default=80)
    sfmp.add_argument('--connectivity-radius-m', type=float, default=1.0)
    sfmp.add_argument('--no-orientation-aware', action='store_true',
                      help='Ablation: disable local-PCA orientation-diverse plane proposals')
    sfmp.add_argument('--diverse-normal-min-angle-deg', type=float, default=25.0,
                      help='Minimum angle from dominant plane normal for diverse proposals')
    sfmp.add_argument('--novel-normal-min-angle-deg', type=float, default=25.0,
                      help='Minimum angle from already accepted normals for novel priority')
    sfmp.add_argument('--local-normal-neighbors', type=int, default=24)
    sfmp.add_argument('--local-normal-max-rmse-m', type=float, default=0.10)
    sfmp.add_argument('--diverse-hypotheses-per-patch', type=int, default=120)
    sfmp.add_argument('--max-normal-distance-m', type=float, default=1.5)
    sfmp.add_argument('--max-support-distance-m', type=float, default=2.5)
    sfmp.add_argument('--min-matches-per-patch', type=int, default=8)
    sfmp.add_argument('--max-matches-per-patch', type=int, default=200)
    sfmp.add_argument('--output', type=Path, required=True, help='Provisional match NPZ')

    sfmr = sub.add_parser('refine-sfm-planes',
        help='Joint 7-DoF Hand-Eye + provisional SfM point / LiDAR plane refinement')
    sfmr.add_argument('--handeye', type=Path, required=True)
    sfmr.add_argument('--matches', type=Path, required=True)
    sfmr.add_argument('--glim', type=Path, required=True)
    sfmr.add_argument('--sfm', type=Path, required=True,
                      help='SAME COLMAP sparse model used in prepare-sfm-planes')
    sfmr.add_argument('--kitti-times', type=Path, required=True)
    sfmr.add_argument('--glim-first-frame', type=int, default=0)
    sfmr.add_argument('--sigma-plane-m', type=float, default=0.20)
    sfmr.add_argument('--motion-weight', type=float, default=1.0)
    sfmr.add_argument('--plane-weight', type=float, default=1.0)
    sfmr.add_argument('--max-rotation-delta-deg', type=float, default=10.)
    sfmr.add_argument('--max-translation-delta-m', type=float, default=2.)
    sfmr.add_argument('--max-log-scale-delta', type=float, default=0.35)
    sfmr.add_argument('--robust-loss', choices=('linear','soft_l1','huber','cauchy'),
                      default='soft_l1')
    sfmr.add_argument('--allow-provisional', action='store_true',
                      help='Explicitly accept that automatic matches are NOT ground truth')
    sfmr.add_argument('--output', type=Path, required=True)

    kitti = sub.add_parser('accumulate-kitti',
                           help='Rigid accumulation of KITTI .bin scans using indexed GLIM Odom poses')
    kitti.add_argument('--lidar-dir', type=Path, required=True)
    kitti.add_argument('--glim', type=Path, required=True)
    kitti.add_argument('--kitti-times', type=Path, required=True)
    kitti.add_argument('--start-frame', type=int, required=True)
    kitti.add_argument('--stop-frame', type=int, required=True)
    kitti.add_argument('--target-frame', type=int, required=True)
    kitti.add_argument('--glim-first-frame', type=int, default=0)
    kitti.add_argument('--min-range-m', type=float, default=0.5)
    kitti.add_argument('--max-range-m', type=float, default=100.0)
    kitti.add_argument('--output', type=Path, required=True, help='Result .npz path')
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == 'inspect':
            config = load_dataset_config(args.config)
            report = inspect_dataset(config, include_hashes=not args.no_hashes)
            rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)+'\n'
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding='utf-8')
            else:
                sys.stdout.write(rendered)
        elif args.command == 'index':
            from spade_calib.io.recording import load_recording_config, index_recording, write_recording_index
            index = index_recording(load_recording_config(args.config))
            report = write_recording_index(index, args.output)
            print(json.dumps({
                'dataset': report['dataset']['name'],
                'output': str(args.output),
                'summary': report['summary'],
                'warnings': report['warnings'],
            }, ensure_ascii=False, indent=2))
        elif args.command == 'select-scenes':
            from spade_calib.selection import (
                SelectionSettings, load_trajectory_data, from_recording_index,
                select_scenes, write_scene_selection,
            )
            cfg = SelectionSettings(
                window_s=args.window_s, window_stride_s=args.window_stride_s,
                pair_baseline_s=args.pair_baseline_s, pair_stride_s=args.pair_stride_s,
                min_pairs=args.min_pairs, max_scenes=args.max_scenes,
                min_lidar_scans_per_window=args.min_lidar_scans,
                max_lidar_gap_s=args.max_lidar_gap_s,
                regularizer=args.regularizer,
            )
            if args.config is not None:
                if (args.sfm is not None or args.camera_to_lidar_offset_ns != 0
                        or args.image_timestamps is not None or args.kitti_times is not None
                        or args.glim_first_frame != 0):
                    raise ValueError('With --config, input and timestamp policy must come from recording YAML')
                from spade_calib.io.recording import load_recording_config, index_recording
                index = index_recording(load_recording_config(args.config))
                samples = from_recording_index(index)
                stamps = [x.timestamp_ns for x in index.lidar_scans]
            else:
                if args.sfm is None:
                    raise ValueError('--sfm is required with --glim')
                index = None
                if args.kitti_times is not None:
                    from spade_calib.io.kitti import load_kitti_trajectory_data
                    if args.image_timestamps is not None or args.camera_to_lidar_offset_ns != 0:
                        raise ValueError('KITTI frame mode cannot combine with clock offset or timestamp CSV')
                    samples = load_kitti_trajectory_data(
                        args.glim, args.sfm, args.kitti_times,
                        glim_first_frame=args.glim_first_frame,
                    )
                else:
                    if args.glim_first_frame != 0:
                        raise ValueError('--glim-first-frame requires --kitti-times')
                    samples = load_trajectory_data(
                        args.glim, args.sfm,
                        camera_to_lidar_offset_ns=args.camera_to_lidar_offset_ns,
                        image_timestamps_csv=args.image_timestamps,
                    )
                stamps = None
            report = select_scenes(samples, cfg, lidar_scan_stamps_ns=stamps)
            write_scene_selection(report, args.output, index=index)
            print(json.dumps({
                'matched_trajectory_poses': report['recording']['matched_camera_glim_poses'],
                'candidates': len(report['candidates']),
                'selected': report['selected_count'],
                'output': str(args.output),
            }, indent=2))
        elif args.command == 'handeye':
            from spade_calib.handeye.cli import run_handeye
            return run_handeye(args)
        elif args.command == 'prepare-sfm-planes':
            from spade_calib.calibration.sfm_plane_cli import run_prepare_sfm_plane_matches
            return run_prepare_sfm_plane_matches(args)
        elif args.command == 'refine-sfm-planes':
            from spade_calib.calibration.sfm_plane_cli import run_refine_sfm_planes
            return run_refine_sfm_planes(args)
        elif args.command == 'refine-ground':
            from spade_calib.calibration.cli import run_ground_refinement
            return run_ground_refinement(args)
        elif args.command == 'infer-depth':
            from spade_calib.camera.depth_cli import run_infer_depth
            return run_infer_depth(args)
        elif args.command == 'align-depth':
            from spade_calib.camera.depth_cli import run_align_depth
            return run_align_depth(args)
        elif args.command == 'prepare-ground':
            from spade_calib.calibration.ground_pipeline import run_prepare_ground
            return run_prepare_ground(args)
        elif args.command == 'accumulate-kitti':
            import os
            import tempfile
            import numpy as np
            from spade_calib.io.kitti import accumulate_kitti_rigid_window, read_kitti_times_ns
            if args.start_frame < 0 or args.stop_frame < args.start_frame:
                raise ValueError('start-frame must be >=0 and stop-frame must be >=start-frame')
            if args.output.suffix.lower() != '.npz':
                raise ValueError('KITTI rigid accumulation output must end with .npz')
            frame_ids = list(range(args.start_frame, args.stop_frame + 1))
            cloud = accumulate_kitti_rigid_window(
                args.lidar_dir, args.glim, args.kitti_times, frame_ids,
                target_frame_id=args.target_frame,
                glim_first_frame=args.glim_first_frame,
                min_range_m=args.min_range_m, max_range_m=args.max_range_m,
            )
            stamps = read_kitti_times_ns(args.kitti_times)[frame_ids]
            dest = args.output.expanduser().resolve()
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp_file = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode='wb', dir=dest.parent, suffix='.npz',
                    prefix='.kitti_', delete=False,
                ) as handle:
                    temp_file = Path(handle.name)
                    np.savez_compressed(
                        handle,
                        xyz=cloud.xyz, intensity=cloud.intensity,
                        scan_id=cloud.scan_id,
                        source_return_index=cloud.source_return_index,
                        frame_ids=np.asarray(frame_ids, dtype=np.int64),
                        scan_timestamps_ns=stamps,
                        reference_timestamp_ns=np.int64(cloud.reference_timestamp_ns),
                        frame_id=np.str_(cloud.frame_id),
                        motion_model=np.str_('rigid'),
                        deskew_applied=np.bool_(False),
                    )
                os.replace(temp_file, dest)
            finally:
                if temp_file is not None:
                    temp_file.unlink(missing_ok=True)
            report = {
                'output': str(dest),
                'point_count': len(cloud.xyz),
                'frame_id': cloud.frame_id,
                'source_frames': len(frame_ids),
                'motion_model': 'rigid',
                'deskew_applied': False,
                'per_return_timestamps_available': False,
                'ring_available': False,
                'reflectance_available': True,
                'scan_timestamp_source': 'KITTI times.txt',
                'pose_mapping': 'GLIM row index, not GLIM timestamp',
                'source_return_index_preserved': True,
                'warning': 'Within-scan motion distortion remains uncorrected',
            }
            sys.stdout.write(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        elif args.command == 'evaluate-kitti-handeye':
            from spade_calib.evaluation.kitti import evaluate_handeye_json
            report = evaluate_handeye_json(args.result, args.calib,
                                           camera_index=args.camera_index)
            rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(rendered, encoding='utf-8')
            else:
                sys.stdout.write(rendered)
        else:
            parser.error('Unknown command')
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, yaml.YAMLError) as exc:
        print(f'spade-calib: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())