"""CLI subcommand implementation, kept separate from legacy commands."""
from __future__ import annotations

import argparse
import json

from spade_calib.handeye.report import report_handeye, write_handeye_report
from spade_calib.handeye.solver import HandEyeSettings, fit_full_trajectory
from spade_calib.selection.trajectory import load_trajectory_data, match_trajectories


def run_handeye(args: argparse.Namespace) -> int:
    if args.config is not None:
        if (args.sfm is not None or args.camera_to_lidar_offset_ns != 0
                or args.kitti_times is not None or args.image_timestamps is not None
                or args.glim_first_frame != 0):
            raise ValueError("Direct input options require --glim; --config uses YAML")
        from spade_calib.io.recording import index_recording, load_recording_config
        index = index_recording(load_recording_config(args.config))
        # Hand-Eye uses ALL registered image poses, not only those with a nearby PCD scan.
        # Important: avoid throwing away valid trajectory samples when raw scans are sparse.
        poses = [a.sfm for a in index.associations if a.sfm is not None]
        image_timestamps = {a.sfm.name: int(a.image.timestamp_ns)
                            for a in index.associations if a.sfm is not None}
        samples = match_trajectories(
            index.glim_trajectory, poses,
            camera_to_lidar_offset_ns=index.config.camera_to_lidar_offset_ns,
            image_timestamp_map=image_timestamps,
        )
        source = str(index.config.source_config)
    else:
        if args.sfm is None:
            raise ValueError("--sfm is required with --glim")
        if args.kitti_times is not None:
            from spade_calib.io.kitti import load_kitti_trajectory_data
            if args.camera_to_lidar_offset_ns != 0 or args.image_timestamps is not None:
                raise ValueError("KITTI frame mode is incompatible with clock offsets and timestamp CSV")
            samples = load_kitti_trajectory_data(
                args.glim, args.sfm, args.kitti_times,
                glim_first_frame=args.glim_first_frame,
            )
        else:
            samples = load_trajectory_data(
                args.glim, args.sfm,
                camera_to_lidar_offset_ns=args.camera_to_lidar_offset_ns,
                image_timestamps_csv=args.image_timestamps,
            )
        source = "GLIM: " + args.glim + "; SfM: " + args.sfm
    settings = HandEyeSettings(
        pair_baseline_s=args.pair_baseline_s,
        pair_stride_s=args.pair_stride_s,
        pair_tolerance_s=args.pair_tolerance_s,
        sigma_relative_rotation_deg=args.sigma_relative_rotation_deg,
        sigma_relative_translation_m=args.sigma_relative_translation_m,
        robust_loss=args.robust_loss,
    )
    result = fit_full_trajectory(samples, settings)
    report = report_handeye(result, bootstrap_resamples=args.bootstrap_resamples,
                            bootstrap_block_s=args.bootstrap_block_s,
                            bootstrap_seed=args.bootstrap_seed,
                            source_description=source)
    write_handeye_report(report, args.output)
    print(json.dumps({
        "output": args.output,
        "matched_pose_count": len(samples.timestamps_ns),
        "motion_pairs": result.pairs.count,
        "sfm_scale_to_metric": result.sfm_scale_to_metric,
        "optimization_converged": result.diagnostics["converged"],
        "joint_rank": report["observability_and_model_covariance"]["rank"],
        "model_based_covariance_available": report["observability_and_model_covariance"]["covariance_available"],
        "bootstrap_available": report["bootstrap"]["available"],
    }, ensure_ascii=False, indent=2))
    return 0