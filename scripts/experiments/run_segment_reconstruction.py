#!/usr/bin/env python3
"""KITTI scale-segment reconstruction ablation: A0 rigid vs A1 refined LiDAR.

Uses the same global SfM model, segment scale, scans and anchor in both arms.
No BA, no GT during fitting, no invented per-return timestamps.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from spade_calib.calibration.segment_geometry import (
    read_sparse_tracks, metric_local_points, associate_patches,
    plane_pair_diagnostics, refine_shared_extrinsic,
)
from spade_calib.correspondence.sfm_lidar_planes import PatchSettings, extract_lidar_patches
from spade_calib.handeye.multiscale import fit_segmented_motion_pairs
from spade_calib.handeye.solver import HandEyeSettings
from spade_calib.io.colmap import load_registered_images, resolve_colmap_images
from spade_calib.io.kitti import (
    accumulate_kitti_rigid_window, frame_id_from_name, load_kitti_trajectory_data,
)
from spade_calib.lidar.joint_solver import JointSettings, solve_joint, voxel_sample
from spade_calib.lidar.pgo_solver import PGOSettings, solve_pgo
from spade_calib.selection.scale_segmentation import (
    ScalePartition, ScaleSegmentationSettings, adaptive_partition,
    fixed_partition, build_frame_lag_pairs, validate_extrinsic,
)

LOG = logging.getLogger(__name__)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_ply(path, xyz):
    points = np.asarray(xyz, dtype="<f4")
    with path.open("wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(points)}\nproperty float x\n"
                 "property float y\nproperty float z\nend_header\n").encode("ascii"))
        points.tofile(f)


def export_geometry(folder, xyz, patches, voxel):
    folder.mkdir(parents=True, exist_ok=True)
    write_ply(folder / "cloud_voxel.ply", xyz[voxel_sample(xyz, voxel)])
    rows = []
    for p in patches:
        write_ply(folder / f"plane_{p.patch_id:02d}.ply", p.support_xyz_lidar)
        rows.append({"patch_id": int(p.patch_id), "normal": p.normal.tolist(),
                     "offset_m": p.offset_m, "rmse_m": p.rmse_m,
                     "support_count": len(p.support_xyz_lidar),
                     "scan_ids": p.scan_ids.tolist()})
    write_json(folder / "planes.json", rows)
    return rows


def scan_frames(lo, hi, anchor, stride, max_scans):
    """Select identical scan IDs in both arms; always include the anchor."""
    frames = sorted(set(range(lo, hi + 1, stride)) | {anchor})
    if max_scans and len(frames) > max_scans:
        idx = np.rint(np.linspace(0, len(frames)-1, max_scans)).astype(int)
        frames = sorted({frames[i] for i in idx} | {anchor})
        if len(frames) > max_scans:
            # Drop nearest non-anchor frame, retain original anchor and order.
            frames.remove(min((f for f in frames if f != anchor), key=lambda f: abs(f-anchor)))
    return frames


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--sequence", type=Path, required=True, help="KITTI sequences/07 with times.txt")
    p.add_argument("--lidar-dir", type=Path, required=True, help="Explicit .bin input (16ch or 64ch)")
    p.add_argument("--glim", type=Path, required=True, help="TUM GLIM LiDAR pose file; contiguous frame rows")
    p.add_argument("--sfm", type=Path, required=True, help="Same COLMAP sparse model as Hand-Eye")
    p.add_argument("--handeye", type=Path, required=True, help="Initial handeye.json with T_camera_lidar")
    p.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    p.add_argument("--glim-first-frame", type=int, default=0)
    p.add_argument("--partition", choices=["adaptive", "fixed", "global"], default="adaptive")
    p.add_argument("--segments", type=int, default=8, help="Fixed K or adaptive maximum K")
    p.add_argument("--bin-frames", type=int, default=20)
    p.add_argument("--min-segment-frames", type=int, default=100)
    p.add_argument("--lag", type=int, default=10)
    p.add_argument("--min-sfm-baseline", type=float, default=.08)
    p.add_argument("--min-scale", type=float, default=1.)
    p.add_argument("--max-scale", type=float, default=200.)
    p.add_argument("--min-cosine", type=float, default=.95)
    p.add_argument("--segment-id", type=int, nargs="+", help="Only process selected segment IDs")
    p.add_argument("--contains-frame", type=int, nargs="+", help="Only process segments containing these KITTI frames")
    p.add_argument("--scan-stride", type=int, default=1)
    p.add_argument("--max-scans", type=int, default=0, help="Optional cap per segment; 0 uses all selected scans")
    p.add_argument("--method", choices=["joint", "pgo"], default="joint")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--voxel", type=float, default=.18)
    p.add_argument("--max-range", type=float, default=65.)
    p.add_argument("--plane-threshold", type=float, default=.14)
    p.add_argument("--max-planes", type=int, default=12)
    p.add_argument("--lidar-min-inliers", type=int, default=80)
    p.add_argument("--sfm-min-inliers", type=int, default=25)
    p.add_argument("--min-track", type=int, default=2)
    p.add_argument("--max-reprojection", type=float, default=2.)
    p.add_argument("--min-parallax", type=float, default=1.)
    p.add_argument("--refine-extrinsic", action="store_true", help="Opt in to provisional fixed plane association solve")
    p.add_argument("--gt-calib", type=Path, help="Optional KITTI GT, evaluation only")
    p.add_argument("--camera-index", type=int, choices=[0,1,2,3], default=2)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if (args.scan_stride < 1 or args.max_scans < 0 or args.max_scans == 1
            or args.iterations < 1 or args.voxel <= 0 or args.max_range <= 1):
        raise ValueError("Invalid scan/voxel/iteration settings")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    images = resolve_colmap_images(args.sfm)
    if images.name not in ("images.bin", "images.txt"):
        raise ValueError("Full COLMAP sparse model required, not only camera poses")
    points = images.parent / ("points3D" + images.suffix)
    if not points.is_file():
        raise FileNotFoundError(points)
    samples = load_kitti_trajectory_data(args.glim, images, args.sequence / "times.txt",
                                         glim_first_frame=args.glim_first_frame)
    records = load_registered_images(images)
    frame_ids = [frame_id_from_name(n) for n in samples.image_names]
    X0 = validate_extrinsic(np.array(json.loads(args.handeye.read_text())["handeye"]["T_camera_lidar"]))
    cfg = ScaleSegmentationSettings(bin_frames=args.bin_frames,
          min_segment_frames=args.min_segment_frames, max_segments=args.segments,
          min_sfm_baseline=args.min_sfm_baseline, min_cosine=args.min_cosine,
          min_scale=args.min_scale, max_scale=args.max_scale)
    pairs = build_frame_lag_pairs(samples, args.lag)
    n = len(frame_ids)
    candidates = []
    if args.partition == "adaptive":
        partition, candidates = adaptive_partition(pairs, n, X0, cfg)
    elif args.partition == "fixed":
        partition = fixed_partition(n, args.segments)
    else:
        partition = ScalePartition((0, n))
    fit = fit_segmented_motion_pairs(pairs, partition, X0, HandEyeSettings(),
                                    segmentation_settings=cfg)
    X = fit.T_camera_lidar
    selected = set(range(partition.count))
    if args.segment_id is not None:
        if not set(args.segment_id) <= selected:
            raise ValueError("Unknown segment ID")
        selected &= set(args.segment_id)
    if args.contains_frame is not None:
        matched = {k for k, (a,b) in enumerate(partition.intervals)
                   if any(frame_ids[a] <= f <= frame_ids[b-1] for f in args.contains_frame)}
        selected &= matched
    if not selected:
        raise ValueError("No selected segments")
    args.out.mkdir(parents=True, exist_ok=True)
    report = fit.to_dict(samples)
    write_json(args.out / "segmented_handeye.json",
               {"fit": report, "candidates": candidates})
    write_json(args.out / "config.json", {k: str(v) if isinstance(v, Path) else v
                                         for k,v in vars(args).items()})
    # Read the sparse track file once, distribute by image membership.
    centers_by_seg = {}
    for k in selected:
        a,b = partition.intervals[k]
        centers_by_seg[k] = {records[samples.image_names[i]].image_id:
                             samples.camera_t_world_sfm[i] for i in range(a,b)}
    by_image = {im:k for k, centers in centers_by_seg.items() for im in centers}
    tracks_by_seg = {k: [] for k in selected}
    for point in read_sparse_tracks(points):
        counts = {}
        for im in set(point.image_ids):
            if im in by_image:
                k = by_image[im]
                counts[k] = counts.get(k, 0) + 1
        for k, count in counts.items():
            if count >= args.min_track:
                tracks_by_seg[k].append(point)
    all_pairs = {"A0": [], "A1": []}
    summary = {"segments": [], "motion_model": "rigid_no_deskew",
               "warning": "A0/A1 fixed original HE scales; no local BA or scale re-fit after pose refinement"}
    def patch_config(min_inliers):
        return PatchSettings(voxel_size_m=args.voxel, max_range_m=args.max_range,
                             distance_threshold_m=args.plane_threshold,
                             max_planes=args.max_planes, min_inliers=min_inliers)
    for k in sorted(selected):
        a,b = partition.intervals[k]
        ref = (a+b-1)//2
        anchor = frame_ids[ref]
        frames = scan_frames(frame_ids[a], frame_ids[b-1], anchor,
                             args.scan_stride, args.max_scans)
        if len(frames) < 2:
            raise ValueError("Need at least two scans per segment")
        base = args.out / f"segment_{k:02d}"
        base.mkdir()
        LOG.info("Segment %d: frames %d..%d, anchor %d, %d scans, scale %.6g",
                 k, frame_ids[a], frame_ids[b-1], anchor, len(frames), fit.scales[k])
        xyz_c, point_ids = metric_local_points(
            tracks_by_seg.pop(k), centers_by_seg[k], samples.camera_R_world_sfm[ref],
            samples.camera_t_world_sfm[ref], float(fit.scales[k]),
            min_track=args.min_track, max_error_px=args.max_reprojection,
            min_parallax_deg=args.min_parallax)
        if len(xyz_c) == 0:
            raise ValueError(f"Segment {k}: no qualifying SfM points")
        cp, cd = extract_lidar_patches(xyz_c, settings=patch_config(args.sfm_min_inliers))
        cam_planes = export_geometry(base / "sfm_camera", xyz_c, cp, args.voxel)
        np.savez_compressed(base / "sfm_camera" / "points.npz", xyz_m=xyz_c, point_ids=point_ids)
        cloud = accumulate_kitti_rigid_window(args.lidar_dir, args.glim,
                    args.sequence / "times.txt", frames, target_frame_id=anchor,
                    glim_first_frame=args.glim_first_frame, max_range_m=args.max_range)
        joint = JointSettings(anchor_scan_id=frames.index(anchor), legacy_roi=False,
                              iterations=args.iterations)
        entry = {"segment_id": k, "anchor_frame": anchor, "scan_frames": frames,
                 "scale": float(fit.scales[k]), "sfm_points": len(xyz_c),
                 "sfm_patch_diagnostics": cd, "sfm_planes": cam_planes, "arms": {}}
        # Complete and persist the baseline first, even if refinement later fails.
        for arm in ("A0", "A1"):
            LOG.info("Segment %d / %s", k, arm)
            folder = base / arm
            if arm == "A0":
                xyz = cloud.xyz
                pose_report = {"method": "GLIM_rigid", "anchor_scan_id": frames.index(anchor)}
            else:
                try:
                    refined = (solve_joint(cloud, joint) if args.method == "joint" else
                               solve_pgo(cloud, PGOSettings(pairs=joint, iterations=args.iterations)))
                except (ValueError, RuntimeError) as exc:
                    entry["arms"][arm] = {"status": "failed", "reason": str(exc)}
                    LOG.error("Segment %d refinement failed: %s", k, exc)
                    break
                xyz = refined.xyz
                pose_report = {"method": refined.method, "iterations": list(refined.iterations),
                               "anchor_scan_id": refined.anchor_scan_id}
                folder.mkdir()
                np.savez_compressed(folder / "pose_corrections.npz", frame_ids=frames,
                                    transforms=refined.transforms, corrections=refined.corrections)
            lp, ld = extract_lidar_patches(xyz, cloud.scan_id, patch_config(args.lidar_min_inliers))
            planes = export_geometry(folder, xyz, lp, args.voxel)
            # Camera-coordinate visualization uses HE only; not used by extraction.
            R,t = X[:3,:3], X[:3,3]
            idx = voxel_sample(xyz, args.voxel)
            write_ply(folder / "lidar_in_camera.ply", xyz[idx] @ R.T + t)
            matches = associate_patches(cp, lp, X, k)
            all_pairs[arm].extend(matches)
            entry["arms"][arm] = {"status": "ok", "lidar_points": len(xyz),
                  "patch_diagnostics": ld, "planes": planes, "pose_refinement": pose_report,
                  "provisional_matches": plane_pair_diagnostics(matches, X)}
        summary["segments"].append(entry)
        write_json(base / "report.json", entry)
        write_json(args.out / "summary.json", summary)
    # Compare refinements only on segment IDs with successful reconstruction in both arms.
    common_segments = {s["segment_id"] for s in summary["segments"]
                       if all(s["arms"].get(a, {}).get("status") == "ok" for a in ("A0", "A1"))}
    summary["common_reconstructed_segments"] = sorted(common_segments)
    if args.refine_extrinsic:
        summary["extrinsic_refinement"] = {}
        for arm in ("A0", "A1"):
            usable = [p for p in all_pairs[arm] if p.segment_id in common_segments]
            summary["extrinsic_refinement"][arm] = refine_shared_extrinsic(usable, X)
    if args.gt_calib is not None:
        # Explicit evaluation-only path. GT never enters the optimizer or association.
        from spade_calib.evaluation.kitti import kitti_gt_camera_lidar, evaluate_camera_lidar
        gt = kitti_gt_camera_lidar(args.gt_calib, args.camera_index)
        summary["gt_evaluation_only"] = {"initial_handeye": evaluate_camera_lidar(X0, gt),
                                        "multi_scale_handeye": evaluate_camera_lidar(X, gt)}
        for arm, r in summary.get("extrinsic_refinement", {}).items():
            if r.get("success"):
                summary["gt_evaluation_only"][arm] = evaluate_camera_lidar(np.array(r["T_camera_lidar"]), gt)
        for arm, matches in all_pairs.items():
            summary["gt_evaluation_only"][arm+"_fixed_match_geometry"] = plane_pair_diagnostics(matches, gt)
    write_json(args.out / "summary.json", summary)
    LOG.info("Completed: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
