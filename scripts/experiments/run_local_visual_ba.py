#!/usr/bin/env python3
"""H: local visual bundle adjustment + fixed-GLIM trajectory/scene constraints."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from run_scene_consistency import load_ground
from run_segment_ground import read_xyz_ply,write_json
from spade_calib.calibration.local_visual_ba import LocalBA,solve_local_ba
from spade_calib.io.colmap import resolve_colmap_images,load_registered_images
from spade_calib.io.colmap_observations import load_cameras,load_observations,camera_parameters,project
from spade_calib.io.kitti import load_kitti_trajectory_data,frame_id_from_name
from spade_calib.handeye.solver import HandEyeSettings
from spade_calib.selection.scale_segmentation import validate_extrinsic
from spade_calib.evaluation.kitti import kitti_gt_camera_lidar,evaluate_camera_lidar

LOG=logging.getLogger(__name__)


def build_local(folder,ground_folder,match,frames,anchor,R,t,intr,observations,
                factors,Ra,ta,scale,max_per_view,ground_cap,min_parallax,seed):
    """Use archived in-segment landmarks and real feature pixels, not projections as labels."""
    with np.load(folder/"sfm_camera"/"points.npz",allow_pickle=False) as z:
        P=np.asarray(z["xyz_m"],float);ids=z["point_ids"].astype(np.int64)
    if len(ids)!=len(P) or len(np.unique(ids))!=len(ids):
        raise ValueError("Invalid archived SfM points")
    order=np.argsort(ids);ids=ids[order];P=P[order]
    obs=[]
    for i,(pids,uv) in enumerate(observations):
        j=np.searchsorted(ids,pids);ok=j<len(ids)
        j=j[ok];uv=uv[ok];pids=pids[ok]
        ok=ids[j]==pids;j=j[ok];uv=uv[ok]
        q=(P[j]-t[i])@R[i]
        error=np.linalg.norm(project(q,np.broadcast_to(intr[i],(len(q),8)))-uv,axis=1)
        ok=(q[:,2]>.05)&(error<=3.)
        j=j[ok];uv=uv[ok]
        if len(np.unique(j))!=len(j):
            raise ValueError("Duplicate landmark observation within image")
        obs.extend((i,int(p),u) for p,u in zip(j,uv))
    if not obs: raise ValueError("No actual feature observations survive archive/reprojection gate")
    oc=np.array([v[0] for v in obs],int);op=np.array([v[1] for v in obs],int)
    uv=np.stack([v[2] for v in obs])
    # At least two selected views and actual selected-view parallax.
    idx=np.argsort(op,kind="stable");sorted_p=op[idx]
    starts=np.r_[0,np.flatnonzero(np.diff(sorted_p))+1,len(idx)]
    eligible=np.zeros(len(P),bool)
    for a,b in zip(starts[:-1],starts[1:]):
        if b-a<2: continue
        j=sorted_p[a];r=P[j]-t[oc[idx[a:b]]]
        r/=np.linalg.norm(r,axis=1,keepdims=True)
        if np.min(r@r[0])<=np.cos(np.deg2rad(min_parallax)):
            eligible[j]=True
    # Re-identify archived B4 ground support by exact XYZ (PLY float32
    # roundoff only). Keep actual track IDs, not arbitrary nearest surfaces.
    gp=read_xyz_ply(ground_folder/"ground_camera.ply").astype(float)
    dist,j=cKDTree(P).query(gp,k=1)
    if not len(gp) or np.max(dist)>1e-4:
        raise ValueError("Archived B4 ground points cannot be identified in SfM archive")
    gj=np.unique(j);gj=gj[eligible[gj]]
    if len(gj)<6:
        raise ValueError("Too few ground tracks in selected views; reduce --center-stride")
    rng=np.random.default_rng(seed)
    chosen_ground=np.sort(rng.choice(gj,min(len(gj),ground_cap),replace=False))
    chosen=set(chosen_ground.tolist())
    # Per-view proposals, union tracks, keep ALL their valid observations.
    # A track can be proposed by several views, hence not a hard obs cap.
    for i in range(len(frames)):
        cand=np.unique(op[(oc==i)&eligible[op]])
        cand=np.setdiff1d(cand,np.array(sorted(chosen),dtype=int))
        have=len(np.unique(op[(oc==i)&np.isin(op,list(chosen))]))
        n=min(len(cand),max(0,max_per_view-have))
        if n: chosen.update(rng.choice(cand,n,replace=False).tolist())
    keep=np.array(sorted(chosen),int)
    mask=np.isin(op,keep)
    ci=oc[mask];pi=np.searchsorted(keep,op[mask]);pixels=uv[mask]
    centers=np.flatnonzero(np.arange(len(frames))!=anchor)
    local=LocalBA(frames,anchor,R,t,P[keep],ids[keep],ci,pi,pixels,intr,
                 centers,Ra,ta,factors,np.searchsorted(keep,chosen_ground),
                 match.camera.normal,match.camera.offset_m,
                 match.lidar.normal,match.lidar.offset_m)
    report=dict(selected_frames=frames.tolist(),anchor_frame=int(frames[anchor]),
                gauge_frame=int(frames[local.gauge[0]]),gauge_translation_axis=local.gauge[1],
                selected_landmarks=len(keep),observations=len(ci),
                min_observations_per_view=int(np.bincount(ci,minlength=len(frames)).min()),
                eligible_ground_tracks=len(gj),selected_ground_point_ids=ids[chosen_ground].tolist(),
                original_scale=float(scale))
    return local,report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--ground-run",type=Path,required=True)
    p.add_argument("--scene-factors",type=Path,required=True,help="G --prepare-only output")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--window",type=int,default=5)
    p.add_argument("--scene-kind",choices=["point","plane"],default="point")
    p.add_argument("--max-tracks-per-view",type=int,default=40)
    p.add_argument("--ground-cap",type=int,default=12)
    p.add_argument("--min-parallax-deg",type=float,default=1.)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--max-nfev",type=int,default=80)
    p.add_argument("--visual-weight",type=float,default=1.)
    p.add_argument("--motion-weight",type=float,default=1.)
    p.add_argument("--ground-weight",type=float,default=.1)
    p.add_argument("--sigma-px",type=float,default=1.)
    p.add_argument("--sigma-scene",type=float,default=.15)
    p.add_argument("--sigma-ground",type=float,default=.15)
    p.add_argument("--pose-rotation-deg",type=float,default=5.)
    p.add_argument("--pose-translation-m",type=float,default=2.)
    p.add_argument("--max-log-scale",type=float,default=.15)
    p.add_argument("--sequence",type=Path)
    p.add_argument("--glim",type=Path)
    p.add_argument("--sfm",type=Path)
    p.add_argument("--gt-calib",type=Path)
    p.add_argument("--image-dir",type=Path)
    args=p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    if (args.window<1 or args.window%2==0 or args.max_tracks_per_view<12 or args.ground_cap<6
            or not 0<args.min_parallax_deg<90 or args.seed<0 or args.max_nfev<10):
        raise ValueError("Invalid BA sampling/iteration settings")
    if args.out.exists() and any(args.out.iterdir()): raise FileExistsError("Use fresh output")
    cfg=json.loads((args.run/"config.json").read_text())
    saved=json.loads((args.run/"segmented_handeye.json").read_text())["fit"]
    archive=json.loads((args.run/"summary.json").read_text())
    prep=json.loads((args.scene_factors/"summary.json").read_text())
    X=validate_extrinsic(np.array(saved["T_camera_lidar"]))
    rows=saved["segments"];scales=np.array([r["sfm_scale_to_metric"] for r in rows])
    K=len(rows);he=HandEyeSettings(**saved["settings"])
    cost=saved["optimization"]["cost_whitened_robust"]
    if (not prep["settings"].get("prepare_only") or Path(prep["run"]).name!=args.run.name
            or not np.allclose(prep["baseline"]["T_camera_lidar"],X,atol=1e-9)
            or not np.allclose(prep["baseline"]["scales"],scales,rtol=1e-10)
            or not np.isclose(prep["recomputed_cost"],cost,rtol=1e-9)
            or prep["support_mode"]!="voxel"):
        raise ValueError("Scene factor archive must match original voxel GLIM experiment")
    matches,ground_doc=load_ground(args.ground_run,args.run,X,scales,cost)
    by_k={m.segment_id:m for m in matches}
    if sorted(by_k)!=list(range(K)): raise ValueError("H requires all segments with accepted B4 ground support")
    samples=load_kitti_trajectory_data(args.glim or cfg["glim"],args.sfm or cfg["sfm"],
                    Path(args.sequence or cfg["sequence"])/"times.txt",
                    glim_first_frame=int(cfg["glim_first_frame"]))
    if (list(samples.image_names)!=[n for r in rows for n in r["image_names"]]
            or samples.timestamps_ns.tolist()!=[n for r in rows for n in r["timestamps_ns"]]):
        raise ValueError("Original trajectory frame identities changed")
    images=resolve_colmap_images(Path(args.sfm or cfg["sfm"]))
    if images.suffix not in (".txt",".bin"): raise ValueError("Full COLMAP model required")
    cams=load_cameras(images.parent/("cameras"+images.suffix))
    records=load_registered_images(images)
    frame_ids=np.array([frame_id_from_name(n) for n in samples.image_names])
    selected=[];anchors={};entries={e["segment_id"]:e for e in archive["segments"]}
    for k,r in enumerate(rows):
        a,b=int(r["start_index"]),int(r["stop_index_exclusive"])
        ai=(a+b-1)//2;anchor=int(frame_ids[ai])
        if r["segment_id"]!=k or entries[k]["anchor_frame"]!=anchor:
            raise ValueError("Segment anchor mismatch")
        anchors[anchor]=k
        with np.load(args.scene_factors/f"factors_w{args.window}"/f"segment_{k:02d}.npz",allow_pickle=False) as z:
            d={name:z[name] for name in z.files}
        fi=d["frames"];ci=np.searchsorted(frame_ids,fi)
        if (np.any(ci<a) or np.any(ci>=b) or not np.array_equal(frame_ids[ci],fi)
                or ai in ci or len(np.unique(ci))!=len(ci)):
            raise ValueError("Factor frame mismatch")
        L=samples.lidar_R_world[ai];l=samples.lidar_t_world_m[ai]
        C=samples.camera_R_world_sfm[ai];c=samples.camera_t_world_sfm[ai]
        Ra=L.T@samples.lidar_R_world[ci];ta=(samples.lidar_t_world_m[ci]-l)@L
        Rb=C.T@samples.camera_R_world_sfm[ci];tb=(samples.camera_t_world_sfm[ci]-c)@C
        for name,v in (("Ra",Ra),("ta",ta),("Rb",Rb),("tb",tb)):
            if not np.allclose(d[name],v,rtol=1e-9,atol=1e-8):
                raise ValueError(f"Factor pose mismatch: {name}")
        # Sorted camera order, factor rows remain in ascending frame order.
        ii=np.sort(np.r_[ci,ai]);anchor_index=int(np.flatnonzero(ii==ai)[0])
        selected.append((ii,anchor_index,C.T@samples.camera_R_world_sfm[ii],
                         scales[k]*((samples.camera_t_world_sfm[ii]-c)@C),Ra,ta,d))
    wanted=[records[samples.image_names[i]].image_id for ii,*_ in selected for i in ii]
    obs=load_observations(images,wanted)
    segments=[];supports=[]
    for k,(ii,anchor,R,t,Ra,ta,d) in enumerate(selected):
        rec=[records[samples.image_names[i]] for i in ii]
        intr=[];o=[]
        for v in rec:
            cid,pids,uv=obs[v.image_id]
            if cid!=v.camera_id: raise ValueError("Camera ID mismatch")
            intr.append(camera_parameters(cams[cid]));o.append((pids,uv))
        seg,report=build_local(args.run/f"segment_{k:02d}",args.ground_run/f"segment_{k:02d}",
                    by_k[k],frame_ids[ii],anchor,R,t,np.stack(intr),o,
                    d[args.scene_kind+"_factors"],Ra,ta,scales[k],
                    args.max_tracks_per_view,args.ground_cap,args.min_parallax_deg,[args.seed,k])
        segments.append(seg);supports.append(report)
        LOG.info("Segment %d: %d cameras, %d landmarks, %d observations",k,len(ii),len(seg.points),len(seg.uv))
    args.out.mkdir(parents=True,exist_ok=True)
    gt=kitti_gt_camera_lidar(args.gt_calib,int(cfg["camera_index"])) if args.gt_calib else None
    summary=dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                 supports=supports,baseline=dict(T_camera_lidar=X.tolist(),scales=scales.tolist()),
                 scene_support=dict(voxel_m=prep["local_voxel_m"],window=args.window,
                                    kind=args.scene_kind,center_stride=prep["settings"]["center_stride"]),
                 experiments={},warning="Local subsampled BA, fixed intrinsics/GLIM, no held-out visual evaluation. Ground offset updated with fixed-normal landmark displacement; not a full plane re-fit.")
    if gt is not None: summary["baseline"]["gt_evaluation"]=evaluate_camera_lidar(X,gt)
    b4=ground_doc["experiments"].get("B4c_w0.1")
    if b4 and b4.get("success"): summary["B4c_reference"]=b4
    opts={k:getattr(args,k) for k in ("visual_weight","motion_weight","ground_weight","sigma_px","sigma_scene","sigma_ground","max_nfev","pose_rotation_deg","pose_translation_m","max_log_scale")}
    def fit(name,kind,fixed=None):
        LOG.info("Optimizing %s",name);start=perf_counter()
        result,local=solve_local_ba(segments,X,scales,he,kind=kind,fixed_local=fixed,**opts)
        result["elapsed_s"]=perf_counter()-start
        if gt is not None and kind!="visual":
            result["gt_evaluation"]=evaluate_camera_lidar(np.array(result["final"]["T_camera_lidar"]),gt)
        summary["experiments"][name]=result
        target=args.out/name;target.mkdir()
        for k,(seg,x) in enumerate(zip(segments,local)):
            R,t,P=seg.state(x)
            np.savez_compressed(target/f"segment_{k:02d}.npz",frames=seg.frames,
                    camera_R_anchor=R,camera_t_anchor_ref_metric=t,landmark_xyz_ref_metric=P,
                    point_ids=seg.point_ids,scale_ref=scales[k],
                    scale_est=result["final"]["scales"][k],local_parameters=x,
                    obs_camera_index=seg.obs_cam,obs_landmark_index=seg.obs_pt,obs_uv=seg.uv)
        write_json(args.out/"summary.json",summary)
        LOG.info("%s: success=%s, nfev=%d, %.1fs",name,result["success"],result["nfev"],result["elapsed_s"])
        return local
    zeros=[np.zeros(s.size) for s in segments]
    fit("Hfixed_HE","he",zeros)
    fit("Hfixed_scene","scene",zeros)
    local=fit("H0_visual","visual")
    # H0 result as a diagnostic two-stage pipeline; keep convergence status.
    fit("H0_then_HE","he",local)
    fit("H0_then_scene","scene",local)
    # Both joint arms start at identical archived poses/landmarks, NOT H0.
    fit("H1_joint_HE","he")
    fit("H2_joint_scene","scene")
    if args.image_dir:
        from spade_calib.evaluation.segment_projection import export_projections
        from spade_calib.evaluation.kitti import parse_kitti_odom_calib
        calib=args.gt_calib or Path(args.sequence or cfg["sequence"])/"calib.txt"
        Kcam=parse_kitti_odom_calib(calib,require_tr=False)[f"P{int(cfg['camera_index'])}"][:,:3]
        transforms={"HE":X}
        if b4 and b4.get("success"): transforms["B4c"]=np.array(b4["final"]["T_camera_lidar"])
        for name,r in summary["experiments"].items():
            if name!="H0_visual" and r["success"]:
                transforms[name]=np.array(r["final"]["T_camera_lidar"])
        if gt is not None: transforms["GT_reference"]=gt
        summary["projection"]=export_projections(image_dir=args.image_dir,scan_dir=Path(cfg["lidar_dir"]),
                    run_dir=args.run,out_dir=args.out/"projection",anchors=anchors,extra_frames=set(),
                    arm="A0",cloud_mode="both",K=Kcam,transforms=transforms,read_xyz_ply=read_xyz_ply,
                    max_depth_m=float(cfg["max_range"]))
        write_json(args.out/"summary.json",summary)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
