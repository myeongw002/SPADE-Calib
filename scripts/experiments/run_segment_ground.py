#!/usr/bin/env python3
"""B4 ground + motion refinement from existing segment reconstruction outputs.

Read A0 by default. Rebuild only trajectory motion pairs, not accumulated
point clouds. Original GLIM/COLMAP and times.txt must still be available.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import numpy as np

from spade_calib.calibration.segmented_ground import (
    GroundSettings, extract_segment_ground, fit_ground_handeye, ground_residuals,
)
from spade_calib.handeye.residuals import residuals
from spade_calib.handeye.solver import HandEyeSettings
from spade_calib.io.kitti import load_kitti_trajectory_data, frame_id_from_name
from spade_calib.selection.scale_segmentation import (
    ScalePartition, build_frame_lag_pairs, validate_extrinsic,
)
from spade_calib.selection.trajectory import MotionPairs

LOG = logging.getLogger(__name__)


def read_xyz_ply(path):
    """Strict reader for XYZ-only binary little-endian PLY written by previous run."""
    with Path(path).open("rb") as f:
        lines = []
        for _ in range(32):
            line = f.readline()
            if not line:
                raise ValueError(f"Truncated PLY header: {path}")
            lines.append(line.strip().decode("ascii"))
            if lines[-1]=="end_header":
                break
        if (len(lines)!=7 or lines[0]!="ply" or lines[1]!="format binary_little_endian 1.0"
                or not lines[2].startswith("element vertex ")
                or lines[3:]!=["property float x","property float y","property float z","end_header"]):
            raise ValueError(f"Expected previous runner XYZ-only PLY: {path}")
        n = int(lines[2].split()[-1])
        if n<0 or Path(path).stat().st_size-f.tell()!=12*n:
            raise ValueError(f"PLY size mismatch: {path}")
        return np.fromfile(f,dtype="<f4",count=n*3).reshape(-1,3)


def write_ply(path, xyz):
    with Path(path).open("wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(xyz)}\nproperty float x\nproperty float y\n"
                 "property float z\nend_header\n").encode("ascii"))
        np.asarray(xyz,dtype="<f4").tofile(f)


def write_json(path, data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run",type=Path,required=True,help="Existing segmentXX_trial1 directory")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--sequence",type=Path,help="Override stored sequence directory")
    p.add_argument("--glim",type=Path,help="Override stored GLIM path")
    p.add_argument("--sfm",type=Path,help="Override stored SfM path")
    p.add_argument("--arm",choices=["A0","A1"],default="A0")
    p.add_argument("--weights",type=float,nargs="+",default=[.1,1.,10.])
    p.add_argument("--ground-mode", choices=["both","normal-offset","offset-only"], default="both",
                   help="Compare B4-a and B4-c on the same fixed supports by default")
    p.add_argument("--refit-scales",action="store_true",help="Also run B4-b with bounded scale update")
    p.add_argument("--image-dir",type=Path,help="Original rectified KITTI image_2 (or selected camera) directory; enables projection")
    p.add_argument("--intrinsics-calib",type=Path,help="KITTI rectified P matrices; defaults to --gt-calib or sequence/calib.txt")
    p.add_argument("--projection-lidar-dir",type=Path,help="Override single-scan directory saved in config; may use original 64ch for visual evaluation only")
    p.add_argument("--projection-cloud",choices=["single","accumulated","both"],default="both")
    p.add_argument("--projection-frames",type=int,nargs="+",help="Extra raw frame IDs for single-scan projections (all segment anchors included)")
    p.add_argument("--projection-max-depth",type=float,default=65.)
    p.add_argument("--sigma-normal-deg",type=float,default=3.)
    p.add_argument("--sigma-offset-m",type=float,default=.15)
    p.add_argument("--max-log-scale-delta",type=float,default=.15)
    p.add_argument("--max-rotation-deg",type=float,default=5.)
    p.add_argument("--max-translation-m",type=float,default=1.)
    p.add_argument("--min-depth",type=float,default=3.)
    p.add_argument("--max-depth",type=float,default=25.)
    p.add_argument("--half-width",type=float,default=3.)
    p.add_argument("--min-sfm-points",type=int,default=25)
    p.add_argument("--min-lidar-points",type=int,default=100)
    p.add_argument("--gt-calib",type=Path,help="Optional explicit evaluation-only KITTI calibration")
    p.add_argument("--camera-index",type=int,choices=[0,1,2,3])
    p.add_argument("--oracle",action="store_true",help="Explicit diagnostic GT start, no re-matching")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    if args.refit_scales and args.ground_mode=="offset-only":
        raise ValueError("--refit-scales is a separate B4-b normal+offset comparison; use both or normal-offset")
    if args.image_dir is None and (args.intrinsics_calib or args.projection_lidar_dir or args.projection_frames):
        raise ValueError("Projection options require --image-dir")
    if args.projection_frames and args.projection_cloud=="accumulated":
        raise ValueError("Extra frame projections require single or both")
    if args.oracle and args.gt_calib is None:
        raise ValueError("--oracle requires explicit --gt-calib")
    if any(not np.isfinite(w) or w<=0 for w in args.weights) or len(set(args.weights))!=len(args.weights):
        raise ValueError("Ground weights must be distinct positive finite values")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    run = args.run
    config = json.loads((run/"config.json").read_text())
    saved = json.loads((run/"segmented_handeye.json").read_text())["fit"]
    old_summary = json.loads((run/"summary.json").read_text())
    X = validate_extrinsic(np.array(saved["T_camera_lidar"]))
    camera_index = int(config["camera_index"])
    if args.camera_index is not None and args.camera_index != camera_index:
        raise ValueError("Cannot change camera index of saved HE / reconstruction")
    rows = saved["segments"]
    partition = ScalePartition(tuple([0]+[int(r["stop_index_exclusive"]) for r in rows]))
    s = np.array([r["sfm_scale_to_metric"] for r in rows])
    for k,(a,b) in enumerate(partition.intervals):
        if rows[k]["segment_id"]!=k or rows[k]["start_index"]!=a:
            raise ValueError("Saved partition mismatch")
    samples = load_kitti_trajectory_data(args.glim or config["glim"],args.sfm or config["sfm"],
                    Path(args.sequence or config["sequence"])/"times.txt",
                    glim_first_frame=int(config["glim_first_frame"]))
    if (list(samples.image_names)!=[n for r in rows for n in r["image_names"]]
            or samples.timestamps_ns.tolist()!=[n for r in rows for n in r["timestamps_ns"]]):
        raise ValueError("Original matched images/timestamps disagree with stored run")
    pairs = build_frame_lag_pairs(samples,int(config["lag"]))
    selected = np.asarray(saved["selected_pair_rows"])
    if (selected.ndim!=1 or selected.dtype.kind not in "iu" or len(np.unique(selected))!=len(selected)
            or np.any(selected<0) or np.any(selected>=pairs.count)):
        raise ValueError("Invalid saved motion-pair rows")
    mask = np.zeros(pairs.count,dtype=bool)
    mask[selected] = True
    groups = partition.groups(pairs)
    if np.any(mask & (groups<0)):
        raise ValueError("Saved pairs cross boundaries")
    he = HandEyeSettings(**saved["settings"])
    # Verify recomputed motion residuals against archived cost. Same pose names
    # alone do not establish that the original SfM/GLIM poses are unchanged.
    cost = 0.
    for k in range(partition.count):
        m = mask & (groups==k)
        if m.sum()!=rows[k]["motion_pair_count"]:
            raise ValueError("Saved pair count mismatch")
        p = MotionPairs(pairs.Ra[m],pairs.ta[m],pairs.Rb[m],pairs.tb[m],pairs.indices[m],pairs.starts_ns[m])
        r = residuals(p,X[:3,:3],X[:3,3],float(s[k]))
        r[:,:3] /= np.deg2rad(he.sigma_relative_rotation_deg)
        r[:,3:] /= he.sigma_relative_translation_m
        from spade_calib.calibration.segmented_ground import _robust
        cost += .5*float(np.sum(_robust(r,he.robust_loss)**2))
    archived = saved["optimization"]["cost_whitened_robust"]
    if not np.isclose(cost,archived,rtol=1e-5,atol=1e-7):
        raise ValueError(f"Archived HE cost mismatch: {archived} vs {cost}; check original pose inputs")
    cfg = GroundSettings(min_depth_m=args.min_depth,max_depth_m=args.max_depth,
                         half_width_m=args.half_width,min_sfm_points=args.min_sfm_points,
                         min_lidar_points=args.min_lidar_points)
    args.out.mkdir(parents=True,exist_ok=True)
    matches, diagnostics = [],[]
    for entry in old_summary["segments"]:
        k = int(entry["segment_id"])
        if entry["arms"].get(args.arm,{}).get("status")!="ok":
            diagnostics.append({"segment_id":k,"accepted":False,"reason":"missing_successful_arm"})
            continue
        a,b = partition.intervals[k]
        expected_anchor = frame_id_from_name(samples.image_names[(a+b-1)//2])
        if entry["anchor_frame"]!=expected_anchor or not np.isclose(entry["scale"],s[k],rtol=1e-10):
            raise ValueError("Saved cloud anchor/scale mismatch")
        folder = run/f"segment_{k:02d}"
        with np.load(folder/"sfm_camera"/"points.npz",allow_pickle=False) as f:
            c = f["xyz_m"]
        l = read_xyz_ply(folder/args.arm/"cloud_voxel.ply")
        match, report, cp, lp = extract_segment_ground(c,l,X,k,cfg)
        diagnostics.append(report)
        LOG.info("Segment %d ground: %s",k,report["reason"])
        if match is not None:
            matches.append(match)
            target = args.out/f"segment_{k:02d}"
            target.mkdir()
            write_ply(target/"ground_camera.ply",cp)
            write_ply(target/"ground_lidar.ply",lp)
            write_ply(target/"ground_lidar_in_camera_HE.ply",lp@X[:3,:3].T+X[:3,3])
    summary = {"run":str(run),"arm":args.arm,"motion_pair_count":int(mask.sum()),
               "archived_HE_cost":archived,"recomputed_HE_cost":cost,
               "ground_settings":asdict(cfg),"ground_diagnostics":diagnostics,
               "accepted_ground_segments":[m.segment_id for m in matches],
               "baseline":{"T_camera_lidar":X.tolist(),"scales":s.tolist()},
               "experiments":{},"oracle_experiments":{}}
    write_json(args.out/"summary.json",summary)
    if args.gt_calib:
        from spade_calib.evaluation.kitti import kitti_gt_camera_lidar,evaluate_camera_lidar
        gt = kitti_gt_camera_lidar(args.gt_calib,args.camera_index if args.camera_index is not None
                                    else int(config["camera_index"]))
        summary["baseline"]["gt_evaluation"]=evaluate_camera_lidar(X,gt)
        raw = ground_residuals(matches,gt,s,s)
        summary["GT_fixed_ground_diagnostic"]=[
            {"segment_id":m.segment_id,"offset_error_m":float(r[3]),
             "normal_angle_deg":float(np.rad2deg(np.arccos(np.clip(
                 m.camera.normal@(gt[:3,:3]@m.lidar.normal),-1,1))))}
            for m,r in zip(matches,raw)]
    variants = []
    if args.ground_mode in ("both","normal-offset"):
        variants.append(("a",False,True))
    if args.ground_mode in ("both","offset-only"):
        variants.append(("c",False,False))
    if args.refit_scales:
        variants.append(("b",True,True))
    for variant,refit,normal_enabled in variants:
        for weight in args.weights:
            name = f"B4{variant}_w{weight:g}"
            opts = dict(he_settings=he,ground_weight=weight,sigma_normal_deg=args.sigma_normal_deg,
                        sigma_offset_m=args.sigma_offset_m,refit_scales=refit,
                        normal_enabled=normal_enabled,
                        max_log_scale_delta=args.max_log_scale_delta,max_rotation_deg=args.max_rotation_deg,
                        max_translation_m=args.max_translation_m)
            LOG.info("Optimizing %s",name)
            result = fit_ground_handeye(pairs,partition,mask,X,s,matches,**opts)
            if args.gt_calib and "final" in result:
                result["gt_evaluation"]=evaluate_camera_lidar(np.array(result["final"]["T_camera_lidar"]),gt)
            summary["experiments"][name]=result
            if args.oracle:
                oracle = fit_ground_handeye(pairs,partition,mask,X,s,matches,initial_X=gt,**opts)
                if "final" in oracle:
                    oracle["gt_evaluation"]=evaluate_camera_lidar(np.array(oracle["final"]["T_camera_lidar"]),gt)
                summary["oracle_experiments"][name]=oracle
            write_json(args.out/"summary.json",summary)
    if args.image_dir is not None:
        from spade_calib.evaluation.segment_projection import export_projections
        from spade_calib.evaluation.kitti import parse_kitti_odom_calib
        calib = args.intrinsics_calib or args.gt_calib or Path(args.sequence or config["sequence"])/"calib.txt"
        K = parse_kitti_odom_calib(calib,require_tr=False)[f"P{camera_index}"][:,:3]
        transforms = {"HE":X}
        for name,result in summary["experiments"].items():
            if result.get("success") and "final" in result:
                transforms[name] = np.asarray(result["final"]["T_camera_lidar"])
        # Only the explicit evaluation GT is shown; never visualize oracle fits as deployable.
        if args.gt_calib:
            transforms["GT_reference"] = gt
        scans = args.projection_lidar_dir or Path(config["lidar_dir"])
        anchors = {int(e["anchor_frame"]):int(e["segment_id"])
                   for e in old_summary["segments"] if e["arms"].get(args.arm,{}).get("status")=="ok"}
        extra = set(args.projection_frames or [])
        if any(f<0 for f in extra):
            raise ValueError("Negative extra frame ID")
        summary["projection"] = export_projections(
            image_dir=args.image_dir, scan_dir=scans, run_dir=run, out_dir=args.out/"projection",
            anchors=anchors, extra_frames=extra, arm=args.arm, cloud_mode=args.projection_cloud,
            K=K, transforms=transforms, read_xyz_ply=read_xyz_ply,
            max_depth_m=args.projection_max_depth)
        summary["projection"]["camera_index"] = camera_index
        summary["projection"]["intrinsics_source"] = str(calib)
        write_json(args.out/"summary.json",summary)
    LOG.info("Completed B4: %s",args.out)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
