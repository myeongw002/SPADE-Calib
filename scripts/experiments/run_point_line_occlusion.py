#!/usr/bin/env python3
"""P2L-0: fixed-camera-line, Camera-view occlusion-point correspondence pilot."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from run_segment_ground import read_xyz_ply
from spade_calib.calibration.point_line_occlusion import (
    project_grid, occlusion_edges, match_points, evaluate_fixed,
)
from spade_calib.evaluation.kitti import kitti_gt_camera_lidar, parse_kitti_odom_calib
from spade_calib.selection.scale_segmentation import validate_extrinsic


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")


def image(path):
    im=cv2.imread(str(path))
    if im is None:
        raise ValueError(f"Cannot read {path}")
    return im


def save(path,im):
    if not cv2.imwrite(str(path),im):
        raise OSError(path)


def draw_lines(im,segments):
    for s in segments:
        cv2.line(im,*[tuple(x) for x in np.rint(s).astype(int)],(0,255,0),1,cv2.LINE_AA)


def draw_points(im,uv,color,radius=1):
    for v in uv:
        cv2.circle(im,tuple(np.rint(v).astype(int)),radius,color,-1)


def finite(x):
    return float(x) if np.isfinite(x) else None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=Path("outputs/segment07_trial1"))
    p.add_argument("--ground",type=Path,default=Path("outputs/ground07_B4c"))
    p.add_argument("--camera",type=Path,default=Path("outputs/edge07_E1_depth"))
    p.add_argument("--cloud-data",type=Path,help="Optional M0 local cloud manifest root; no 3D line files read")
    p.add_argument("--gt-calib",type=Path,help="Optional diagnostic-only KITTI calibration")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--frames",type=int,nargs="+")
    p.add_argument("--jump-threshold",type=float,default=.4)
    p.add_argument("--gates",type=float,nargs="+",default=[20.,40.,70.])
    p.add_argument("--cell",type=int,default=4)
    a=p.parse_args(argv)
    gates=sorted(set(a.gates))
    if (a.cell<1 or not np.isfinite([a.jump_threshold,*gates]).all()
            or a.jump_threshold<0 or min(gates)<=0):
        p.error("Invalid cell, jump or gates")
    if a.out.exists() and (not a.out.is_dir() or any(a.out.iterdir())):
        raise FileExistsError("Use fresh output")
    run=read(a.run/"summary.json");cfg=read(a.run/"config.json")
    gd=read(a.ground/"summary.json");cam=read(a.camera/"summary.json")
    fit=gd["experiments"]["B4c_w0.1"]
    if not fit["success"] or fit["refit_scales"] or fit["normal_enabled"]:
        raise ValueError("Expected successful offset-only fixed-scale B4-c")
    he=read(a.run/"segmented_handeye.json")["fit"]["T_camera_lidar"]
    if not np.allclose(he,gd["baseline"]["T_camera_lidar"],atol=1e-9,rtol=0):
        raise ValueError("B4-c and run HE mismatch")
    X=validate_extrinsic(np.asarray(fit["final"]["T_camera_lidar"],float))
    K=np.asarray(cam["K"],float);camera=int(cam["camera_index"])
    if camera!=int(cfg["camera_index"]) or K.shape!=(3,3) or not np.isfinite(K).all():
        raise ValueError("Camera mismatch")
    if cam["depth_mode"]=="disabled":
        raise ValueError("E1 depth outputs required")
    if a.gt_calib:
        cal=parse_kitti_odom_calib(a.gt_calib)
        if not np.allclose(cal[f"P{camera}"][:,:3],K):
            raise ValueError("GT diagnostic calibration K mismatch")
        Xgt=kitti_gt_camera_lidar(a.gt_calib,camera)
    else:
        Xgt=None
    poses={"B4c":X}
    if Xgt is not None:poses["GT_oracle"]=Xgt
    frames={int(e["anchor_frame"]):e for e in run["segments"]}
    selected=sorted(set(a.frames or (int(f) for f in cam["frames"])))
    if not selected or any(f not in frames or str(f) not in cam["frames"] for f in selected):
        raise ValueError("P2L-0 requires E1 frames that are archived segment anchors")
    planes={int(e["segment_id"]):e for e in gd["ground_diagnostics"] if e["accepted"]}
    if any(int(frames[f]["segment_id"]) not in planes for f in selected):
        raise ValueError("Missing B4-c LiDAR ground plane for anchor")
    rows=defaultdict(list)
    with (a.camera/"features.csv").open(newline="") as fp:
        for r in csv.DictReader(fp):
            rows[int(r["frame"])].append(r)
    manifest=None; clouds={}
    if a.cloud_data:
        manifest=read(a.cloud_data/"manifest.json")
        if int(manifest["camera_index"])!=camera or not np.allclose(manifest["K"],K):
            raise ValueError("M0 camera mismatch")
        clouds={int(e["frame"]):e for e in manifest["frames"]}
        if any(f not in clouds for f in selected):
            raise ValueError("M0 cloud not available for E1 anchor")
        if Path(manifest["run"]).name!=a.run.name:
            raise ValueError("M0 prepared from a different named run")
    a.out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    summary=dict(warning="Galibr-inspired custom pilot, NOT exact replication. "
                 "No optimization or verified physical identity. GT oracle is diagnostic only. "
                 "GT fixed-pair inlier fractions are geometric consistency, not precision. "
                 "A0 long segment or M0 local accumulated cloud: no deskew, moving objects and occlusion errors remain. "
                 "Point IDs are indices in frozen cloud, not reconstructed 3D lines.",
                 source=dict(run=str(a.run),camera=str(a.camera),ground=str(a.ground),
                             cloud_data=str(a.cloud_data) if a.cloud_data else None,
                             cloud_mode="M0_local" if a.cloud_data else gd["arm"],
                             run_sha256=sha(a.run/"summary.json"),ground_sha256=sha(a.ground/"summary.json"),
                             camera_sha256=sha(a.camera/"summary.json"),features_sha256=sha(a.camera/"features.csv"),
                             manifest_sha256=sha(a.cloud_data/"manifest.json") if manifest else None),
                 settings=dict(cell_px=a.cell,max_depth_m=40.,ground_margin_m=.25,
                               jump_abs_m=1.,jump_rel=.1,neighbor_gap_cells=2,
                               foreground_same_abs_m=.4,foreground_same_rel=.05,
                               camera_jump_threshold=a.jump_threshold,camera_jump_dominant=True,
                               camera_valid_fraction=.5,gates_px=gates,ambiguity_px=4.,
                               angle_deg=45.,bin_px=8.,min_bins=3,min_coverage=.25,
                               gt_consistency_px=10.),
                 K=K.tolist(),X_B4c=X.tolist(),X_GT_diagnostic=Xgt.tolist() if Xgt is not None else None,
                 frames={})
    stats_rows=[];review=[]
    for f in selected:
        e=frames[f];k=int(e["segment_id"]);row=cam["frames"][str(f)]
        src=a.camera/f"frame_{f:06d}";im=image(src/"original.png")
        if list(im.shape[:2])!=row["shape"] or sha(src/"elsed_segments.npz")!=row["elsed_sha256"]:
            raise ValueError("E1 image shape or ELSED archive mismatch")
        with np.load(src/"elsed_segments.npz",allow_pickle=False) as z:
            raw=np.asarray(z["segments"],float).reshape(-1,2,2)
        rec=rows[f]
        if len(rec)!=row["eligible"] or len({int(r["image_id"]) for r in rec})!=len(rec):
            raise ValueError("E1 feature count/ID mismatch")
        camera_rows=[]
        for r in rec:
            idx=int(r["image_id"])
            uv=np.array([r[x] for x in ("u0","v0","u1","v1")],float).reshape(2,2)
            if idx<0 or idx>=len(raw) or not np.allclose(uv,raw[idx],rtol=0,atol=1e-7):
                raise ValueError("E1 feature coordinates mismatch")
            if not r["jump"] or not r["crease"] or not r["valid_fraction"]:continue
            j,c,v=map(float,(r["jump"],r["crease"],r["valid_fraction"]))
            if np.isfinite([j,c,v]).all() and j>=a.jump_threshold and j>=c and v>=.5:
                camera_rows.append((idx,uv))
        ids=np.asarray([x[0] for x in camera_rows],int)
        segments=np.asarray([x[1] for x in camera_rows],float).reshape(-1,2,2)
        if a.cloud_data:
            base=a.cloud_data/f"frame_{f:06d}";cr=clouds[f]
            cp=base/"cloud.npz"
            if sha(cp)!=cr["cloud_sha256"] or sha(base/"image.png")!=cr["image_sha256"]:
                raise ValueError("M0 archive changed")
            if not np.array_equal(image(base/"image.png"),im):
                raise ValueError("M0/E1 image mismatch")
            with np.load(cp,allow_pickle=False) as z:
                xyz=np.asarray(z["xyz"],float)
            if len(xyz)!=cr["points"]:
                raise ValueError("M0 point count mismatch")
        else:
            cp=a.run/f"segment_{k:02d}"/gd["arm"]/"cloud_voxel.ply"
            xyz=read_xyz_ply(cp)
        pl=planes[k]["lidar"];n=np.asarray(pl["normal"],float);d=float(pl["offset_m"])
        if not np.isclose(np.linalg.norm(n),1.) or n[2]<=0 or d<=0:
            raise ValueError("Expected unit positive-up ground plane")
        ng=xyz@n+d>.25
        folder=a.out/f"frame_{f:06d}";folder.mkdir()
        info=dict(segment=k,cloud_sha256=sha(cp),points=len(xyz),
                  non_ground=int(ng.sum()),camera_lines=len(segments),camera_line_ids=ids.tolist(),arms={})
        np.savez_compressed(folder/"camera_lines.npz",segments=segments,image_ids=ids)
        for name,T in poses.items():
            projected=project_grid(xyz,T,K,im.shape[:2],ng,cell=a.cell)
            edge=occlusion_edges(projected)
            overlay=im.copy();draw_lines(overlay,segments);draw_points(overlay,projected["uv"][edge.any(1)],(0,165,255))
            save(folder/f"{name}_edge_candidates.png",overlay)
            for gate in gates:
                for arm in ("all","edge"):
                    label=f"{name}_{arm}_{gate:g}px"
                    groups,stats=match_points(projected,edge,segments,gate=gate,use_edges=arm=="edge")
                    st=dict(frame=f,init=name,arm=arm,gate=gate,camera_lines=len(segments),
                            visible=len(projected["ids"]),edge_points=int(edge.any(1).sum()),**stats)
                    fixed=evaluate_fixed(xyz,groups,segments,Xgt,K,im.shape[:2]) if Xgt is not None else None
                    if fixed is not None:
                        err=np.concatenate([r["signed_px"][r["inside"]] for r in fixed]) if fixed else np.empty(0)
                        st.update(gt_inside=len(err),gt_within10=int((abs(err)<=10).sum()),
                                  gt_abs_median=float(np.median(abs(err))) if len(err) else None)
                    im2=im.copy();draw_lines(im2,segments)
                    pts=[];line_ids=[];uvs=[];res=[];gt_res=[];gt_inside=[]
                    for i,gr in enumerate(groups):
                        line=gr["line"];pp=gr["point_indices"];count=len(pp)
                        pts.extend(pp);line_ids.extend([ids[line]]*count);uvs.extend(gr["uv"]);res.extend(gr["signed_px"])
                        draw_points(im2,gr["uv"],(0,0,255),2)
                        r=dict(frame=f,init=name,arm=arm,gate=gate,image_id=int(ids[line]),
                               points=count,coverage=gr["coverage"],raw_count=gr["raw_count"],
                               label="",notes="")
                        if fixed is not None:
                            ev=fixed[i];gt_res.extend(ev["signed_px"]);gt_inside.extend(ev["inside"])
                            r.update(gt_inside=int(ev["inside"].sum()),
                                     gt_within10=int((ev["inside"]&(abs(ev["signed_px"])<=10)).sum()))
                        review.append(r)
                    pi=np.asarray(pts,int)
                    np.savez_compressed(folder/f"{label}_pairs.npz",point_indices=pi,xyz_lidar=xyz[pi],
                                        image_ids=np.asarray(line_ids,int),uv=np.asarray(uvs,float).reshape(-1,2),
                                        signed_px=np.asarray(res,float),gt_signed_px=np.asarray(gt_res,float),
                                        gt_inside=np.asarray(gt_inside,bool))
                    save(folder/f"{label}.png",im2)
                    stats_rows.append(st);info["arms"][label]=st
        summary["frames"][str(f)]=info
        write(folder/"summary.json",info)
        print(f"{f:06d}: {len(xyz)} points, {len(segments)} camera lines",flush=True)
    for name,data in (("metrics.csv",stats_rows),("review.csv",review)):
        keys=list(dict.fromkeys(k for r in data for k in r))
        with (a.out/name).open("w",newline="") as fp:
            w=csv.DictWriter(fp,fieldnames=keys);w.writeheader();w.writerows(data)
    write(a.out/"summary.json",summary)
    return 0


if __name__=="__main__":
    with threadpool_limits(limits=1):
        raise SystemExit(main())
