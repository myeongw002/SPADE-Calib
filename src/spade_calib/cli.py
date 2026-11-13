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
        elif args.command == 'evaluate-kitti-handeye':
            from spade_calib.evaluation.kitti import evaluate_handeye_json
            report = evaluate_handeye_json(args.result, args.calib,
                                           camera_index=args.camera_index)
            rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\\n'
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