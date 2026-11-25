#!/usr/bin/env python3
"""H: local visual bundle adjustment + fixed-GLIM trajectory/scene constraints."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from dataclasses import replace
import hashlib
from scipy.spatial.transform import Rotation
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from run_scene_consistency import load_ground
from run_segment_ground import read_xyz_ply,write_json
from spade_calib.calibration.local_visual_ba import LocalBA,solve_local_ba
from spade_calib.calibration.ba_diagnostics import virtual_moments,moment_factors
from spade_calib.calibration.crossmodal_lines import fit_lines,metrics as line_metrics,image_line,sample_line,project_line
from spade_calib.io.colmap import resolve_colmap_images,load_registered_images
from spade_calib.io.colmap_observations import load_cameras,load_observations,camera_parameters,project,unique_view_observations
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
    duplicate_counts=[]
    for i,(pids,uv) in enumerate(observations):
        j=np.searchsorted(ids,pids);ok=j<len(ids)
        j=j[ok];uv=uv[ok];pids=pids[ok]
        ok=ids[j]==pids;j=j[ok];uv=uv[ok]
        q=(P[j]-t[i])@R[i]
        error=np.linalg.norm(project(q,np.broadcast_to(intr[i],(len(q),8)))-uv,axis=1)
        ok=(q[:,2]>.05)&(error<=3.)
        j,uv,removed=unique_view_observations(j[ok],uv[ok],error[ok])
        duplicate_counts.append(removed)
        if removed:
            LOG.debug("Frame %d: removed %d duplicate landmark observations",int(frames[i]),removed)
        obs.extend((i,int(p),u) for p,u in zip(j,uv))
    if sum(duplicate_counts):
        LOG.info("Anchor %d: removed %d duplicate observations across %d views",
                 int(frames[anchor]),sum(duplicate_counts),sum(c>0 for c in duplicate_counts))
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
                duplicate_observations_removed=int(sum(duplicate_counts)),
                duplicate_counts_by_frame={str(int(f)):int(c) for f,c in zip(frames,duplicate_counts) if c},
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
    p.add_argument("--ablation-windows",type=int,nargs="+",help="Matched joint BA window ablation; replaces normal seven arms")
    p.add_argument("--extended-ablation",action="store_true",help="Also run ground-off and rotation attenuation controls")
    p.add_argument("--ij",action="store_true",help="Run I observability and J virtual moment experiments instead of H")
    p.add_argument("--s1",action="store_true",help="S1: diagnose scale coupling at archived H1/H2 states without refitting")
    p.add_argument("--line-data",type=Path,help="M1 mode: prepared M0 manifest and annotations")
    p.add_argument("--line-weights",type=float,nargs="+",default=[.01,.1,1.])
    p.add_argument("--line-sigma-px",type=float,default=2.)
    p.add_argument("--line-holdout-frame",type=int,help="Optional exploratory frame leave-one-out; never use held-out line in fit")
    p.add_argument("--line-initial-summary",type=Path,help="Prior H summary for M1a/b (default --reference-ba/summary.json)")
    p.add_argument("--ij-mode",choices=["both","I","J"],default="both")
    p.add_argument("--shuffle-repeats",type=int,default=3)
    p.add_argument("--information-maxiter",type=int,default=10000)
    p.add_argument("--rotation-multiplier",type=float,default=.5,help="Rotation attenuation in extended controls only")
    p.add_argument("--visual-max-nfev",type=int,default=300,help="H0 budget in ablation mode")
    p.add_argument("--reference-ba",type=Path,help="Validate selected visual support and original state against prior Hfixed_HE NPZs")
    p.add_argument("--gt-poses",type=Path,help="Optional KITTI camera0 GT poses, evaluation only")
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
    if args.s1 and (not args.reference_ba or args.ij or args.line_data or args.ablation_windows
                     or args.extended_ablation or args.image_dir or args.window!=5 or args.scene_kind!="point"
                     or args.ground_weight<=0 or args.information_maxiter<1):
        raise ValueError("S1 requires original --reference-ba H, W5 point, positive ground; no other modes")
    if args.line_data:
        if args.ij or args.ablation_windows or args.extended_ablation or args.image_dir or not args.reference_ba or args.window!=5 or args.scene_kind!="point":
            raise ValueError("M1 requires --reference-ba, W5 point factors; no H/IJ/projection flags")
        if (not np.isfinite(args.line_weights).all() or min(args.line_weights)<=0
                or len(set(args.line_weights))!=len(args.line_weights)
                or not np.isfinite(args.line_sigma_px) or args.line_sigma_px<=0 or args.ground_weight<=0):
            raise ValueError("Invalid M1 settings")
    elif args.line_holdout_frame is not None or args.line_initial_summary:
        raise ValueError("Line holdout/initializer options require --line-data")
    windows=sorted(set([1,5] if args.ij else (args.ablation_windows or [args.window])))
    if args.ij and (args.window!=5 or args.scene_kind!="point" or args.ablation_windows or args.extended_ablation or args.ground_weight<=0 or args.shuffle_repeats<1 or args.information_maxiter<1):
        raise ValueError("IJ requires W1/W5 point factors, --window 5, positive ground weight; no H ablation flags")
    if (any(w<1 or w%2==0 for w in windows) or args.window not in windows
            or args.visual_max_nfev<10 or not 0<args.rotation_multiplier<=1
            or (args.extended_ablation and not args.ablation_windows)
            or (args.gt_poses and not args.gt_calib)):
        raise ValueError("Invalid ablation settings")
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
    selected=[];anchors={};alternate={w:[] for w in windows};entries={e["segment_id"]:e for e in archive["segments"]}
    for k,r in enumerate(rows):
        a,b=int(r["start_index"]),int(r["stop_index_exclusive"])
        ai=(a+b-1)//2;anchor=int(frame_ids[ai])
        if r["segment_id"]!=k or entries[k]["anchor_frame"]!=anchor:
            raise ValueError("Segment anchor mismatch")
        anchors[anchor]=k
        with np.load(args.scene_factors/f"factors_w{args.window}"/f"segment_{k:02d}.npz",allow_pickle=False) as z:
            d={name:z[name] for name in z.files}
        fi=d["frames"];ci=np.searchsorted(frame_ids,fi)
        for w in windows:
            with np.load(args.scene_factors/f"factors_w{w}"/f"segment_{k:02d}.npz",allow_pickle=False) as z:
                for key in ("frames","Ra","ta","Rb","tb"):
                    if not np.array_equal(z[key],d[key]):
                        raise ValueError(f"Window {w}, segment {k}: factor support/pose mismatch: {key}")
                F=z[args.scene_kind+"_factors"]
                if F.shape!=(len(fi),12,12) or not np.isfinite(F).all() or np.any(z["point_counts"]<3):
                    raise ValueError("Invalid moment factor archive")
                alternate[w].append((F,z["point_counts"]))
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
        if args.reference_ba:
            with np.load(args.reference_ba/"Hfixed_HE"/f"segment_{k:02d}.npz",allow_pickle=False) as z:
                for key,value in (("frames",seg.frames),("point_ids",seg.point_ids),
                                  ("obs_camera_index",seg.obs_cam),("obs_landmark_index",seg.obs_pt),("obs_uv",seg.uv)):
                    if not np.array_equal(z[key],value):
                        raise ValueError(f"Prior H support mismatch: segment {k}, {key}")
                for key,value in (("camera_R_anchor",seg.R),("camera_t_anchor_ref_metric",seg.t),
                                  ("landmark_xyz_ref_metric",seg.points),("scale_ref",scales[k])):
                    if not np.allclose(z[key],value,rtol=1e-10,atol=1e-9):
                        raise ValueError(f"Prior H initial state mismatch: segment {k}, {key}")
        h=hashlib.sha256()
        for v in (seg.frames,seg.point_ids,seg.obs_cam,seg.obs_pt,seg.uv,seg.intr,seg.ground_ids):
            a=np.ascontiguousarray(v)
            h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
        report["support_sha256"]=h.hexdigest()
        segments.append(seg);supports.append(report)
        LOG.info("Segment %d: %d cameras, %d landmarks, %d observations",k,len(ii),len(seg.points),len(seg.uv))
    by_window={w:[replace(s,factors=alternate[w][k][0]) for k,s in enumerate(segments)] for w in windows}
    args.out.mkdir(parents=True,exist_ok=True)
    gt=kitti_gt_camera_lidar(args.gt_calib,int(cfg["camera_index"])) if args.gt_calib else None
    summary=dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                 supports=supports,baseline=dict(T_camera_lidar=X.tolist(),scales=scales.tolist()),
                 scene_support=dict(voxel_m=prep["local_voxel_m"],window=args.window,
                                    kind=args.scene_kind,center_stride=prep["settings"]["center_stride"]),
                 experiments={},warning="Local subsampled BA, fixed intrinsics/GLIM, no held-out visual evaluation. Ground offset updated with fixed-normal landmark displacement; not a full plane re-fit.")
    summary["window_point_counts"]={str(w):[dict(total=int(c.sum()),mean=float(c.mean())) for _,c in alternate[w]] for w in windows}
    summary["reference_ba_verified"]=bool(args.reference_ba)
    if gt is not None: summary["baseline"]["gt_evaluation"]=evaluate_camera_lidar(X,gt)
    gt_world=None
    if args.gt_poses:
        from spade_calib.evaluation.kitti import parse_kitti_odom_calib
        cal=parse_kitti_odom_calib(args.gt_calib)
        raw=np.loadtxt(args.gt_poses).reshape(-1,3,4)
        Rgt=Rotation.from_matrix(raw[:,:,:3]).as_matrix()
        p0,pk=cal["P0"],cal[f"P{int(cfg['camera_index'])}"]
        baseline=np.linalg.solve(pk[:,:3],pk[:,3])-np.linalg.solve(p0[:,:3],p0[:,3])
        tgt=raw[:,:,3]-Rgt@baseline
        gt_world=(Rgt,tgt)
    def trajectory_metrics(states,ratios):
        if gt_world is None: return None
        trans=[];rot=[];length=[];anchor_errors=[];negative=[];counts=[]
        GR,GT=gt_world
        for seg,(R,t,P),ratio in zip(segments,states,ratios):
            f=seg.frames; a=seg.anchor
            if f.min()<0 or f.max()>=len(GR): raise ValueError("GT frame out of range")
            Rg=GR[f];tg=GT[f]
            t=ratio*t
            gt_anchor=(tg-tg[a])@Rg[a]
            mask=np.arange(len(f))!=a
            anchor_errors.extend(np.linalg.norm(t[mask]-gt_anchor[mask],axis=1))
            lookup={int(v):i for i,v in enumerate(f)}
            ij=[(i,lookup[int(v)+10]) for i,v in enumerate(f) if int(v)+10 in lookup]
            counts.append(len(ij))
            if not ij: continue
            i,j=np.array(ij).T
            br=R[i].transpose(0,2,1)@R[j]; gr=Rg[i].transpose(0,2,1)@Rg[j]
            bt=np.einsum("ni,nij->nj",t[j]-t[i],R[i]);gt=np.einsum("ni,nij->nj",tg[j]-tg[i],Rg[i])
            trans.extend(np.linalg.norm(bt-gt,axis=1))
            rot.extend(np.rad2deg(Rotation.from_matrix(gr.transpose(0,2,1)@br).magnitude()))
            le=np.linalg.norm(bt,axis=1)-np.linalg.norm(gt,axis=1)
            length.extend(le)
            neg=np.rad2deg(Rotation.from_matrix(gr).as_rotvec()[:,1])<-2
            negative.extend(le[neg])
        rms=lambda v:float(np.sqrt(np.mean(np.square(v)))) if len(v) else None
        return dict(pair_count=len(trans),pairs_per_segment=counts,lag_frames=10,
                    translation_rmse_m=rms(trans),rotation_rmse_deg=rms(rot),
                    motion_length_mae_m=float(np.mean(abs(np.asarray(length)))) if length else None,
                    negative_yaw_count=len(negative),negative_yaw_length_bias_m=float(np.mean(negative)) if negative else None,
                    anchor_translation_rmse_m=rms(anchor_errors))
    if gt_world is not None:
        summary["baseline"]["trajectory_gt"]=trajectory_metrics([(s.R,s.t,s.points) for s in segments],np.ones(K))
    b4=ground_doc["experiments"].get("B4c_w0.1")
    if b4 and b4.get("success"): summary["B4c_reference"]=b4
    opts={k:getattr(args,k) for k in ("visual_weight","motion_weight","ground_weight","sigma_px","sigma_scene","sigma_ground","max_nfev","pose_rotation_deg","pose_translation_m","max_log_scale")}
    line_train=[];line_test=[];line_K=None;line_X=None
    if args.line_data:
        mp=args.line_data/"manifest.json";ap=args.line_data/"annotations.json"
        md=json.loads(mp.read_text());ad=json.loads(ap.read_text())
        if ad["manifest_sha256"]!=hashlib.sha256(mp.read_bytes()).hexdigest():
            raise ValueError("Annotation manifest mismatch")
        if Path(md["run"]).name!=args.run.name or md["camera_index"]!=int(cfg["camera_index"]):
            raise ValueError("M0/M1 sequence or camera mismatch")
        line_K=np.asarray(md["K"],float);line_X=validate_extrinsic(np.asarray(md["T_camera_lidar"],float))
        prior=json.loads((args.line_initial_summary or (args.reference_ba/"summary.json")).read_text())
        if not np.allclose(prior["baseline"]["T_camera_lidar"],X,atol=1e-9):
            raise ValueError("H baseline mismatch")
        arm=prior["experiments"][md["initial_arm"]]
        if not arm["success"] or not np.allclose(line_X,arm["final"]["T_camera_lidar"],atol=1e-9):
            raise ValueError("M0 initialization is not the specified successful H arm")
        from spade_calib.evaluation.kitti import parse_kitti_odom_calib
        kp=parse_kitti_odom_calib(Path(args.sequence or cfg["sequence"])/"calib.txt",require_tr=False)[f"P{int(cfg['camera_index'])}"][:,:3]
        if not np.allclose(line_K,kp,rtol=1e-8,atol=1e-8): raise ValueError("M0 rectified K mismatch")
        frames={r["frame"]:r for r in md["frames"]};seen=set();splits={};features={}
        for r in ad["records"]:
            if r["id"] in seen or r["frame"] not in frames or not r["manual_identity_confirmed"] or r["split"] not in ("train","test"):
                raise ValueError("Invalid M0 manual record")
            seen.add(r["id"])
            for store,key in ((splits,r["frame"]),(features,r["feature"])):
                if key in store and store[key]!=r["split"]: raise ValueError("Original annotation train/test leakage")
                store[key]=r["split"]
            sample_line(r["xyz_lidar"]);image_line(r["uv_image"])
            uv=np.asarray(r["uv_image"]);hh,ww=frames[r["frame"]]["image_shape"]
            if np.any(uv<0) or np.any(uv>=np.array([ww,hh])):
                raise ValueError("Image annotation outside original image")
            if r["split"]=="test" or r["frame"]==args.line_holdout_frame: line_test.append(r)
            else: line_train.append(r)
        if args.line_holdout_frame is not None and args.line_holdout_frame not in {r["frame"] for r in line_test}:
            raise ValueError("Requested holdout frame has no annotation")
        if len(line_train)<2 or len({r["frame"] for r in line_train})<2:
            raise ValueError("M1 requires at least two line records from two frames")
        if line_metrics(line_X,line_train,line_K)["nonpositive_or_near_samples"]:
            raise ValueError("Invalid line depth at H2 initializer")
        summary["line_data"]=dict(manifest_sha256=ad["manifest_sha256"],annotations_sha256=hashlib.sha256(ap.read_bytes()).hexdigest(),
            train_ids=[r["id"] for r in line_train],test_ids=[r["id"] for r in line_test],
            feature_overlap=sorted(set(r["feature"] for r in line_train)&set(r["feature"] for r in line_test)),
            warning="Exploratory frame holdout may share a physical feature; weights not selected by GT/test. Synchronous target LiDAR coordinates, fixed GLIM, line residual depends only on X.")
        if gt is not None: summary["line_data"]["GT_diagnostic"]=dict(train=line_metrics(gt,line_train,line_K),test=line_metrics(gt,line_test,line_K))
    def fit(name,kind,fixed=None,window=None,custom_support=None,**overrides):
        w=args.window if window is None else window
        support=by_window[w] if custom_support is None else custom_support
        settings={**opts,**overrides}
        LOG.info("Optimizing %s",name);start=perf_counter()
        result,local=solve_local_ba(support,X,scales,he,kind=kind,fixed_local=fixed,
            line_records=line_train or None,line_K=line_K,**settings)
        result["window"]=w;result["fit_settings"]=settings
        states=[s.state(x) for s,x in zip(support,local)]
        ratio=np.asarray(result["final"]["scales"])/scales
        Y=np.array(result["final"]["T_camera_lidar"])
        if args.line_data:
            result["line_train"]=line_metrics(Y,line_train,line_K)
            result["line_test"]=line_metrics(Y,line_test,line_K)
        if gt_world is not None:
            result["trajectory_gt"]=trajectory_metrics(states,ratio)
        # Evaluate unattenuated physical scene RMS on BOTH frozen geometries.
        result["cross_window_scene_rmse_m"]={}
        for ww,ss in by_window.items():
            errs=[np.mean(np.sum(s.motion(R,t,Y,ratio[k],"scene",he)**2,axis=1))
                  for k,(s,(R,t,P)) in enumerate(zip(ss,states))]
            result["cross_window_scene_rmse_m"][str(ww)]=float(np.sqrt(np.mean(errs)))
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
    if args.s1:
        ref=args.reference_ba
        prior=json.loads((ref/"summary.json").read_text())
        if (not np.allclose(prior["baseline"]["T_camera_lidar"],X,atol=1e-9)
                or not np.allclose(prior["baseline"]["scales"],scales,rtol=1e-10)
                or Path(prior["settings"]["run"]).name!=args.run.name
                or prior["settings"]["window"]!=5 or prior["settings"]["scene_kind"]!="point"):
            raise ValueError("S1 reference baseline/config mismatch")
        for key in opts:
            if key!="max_nfev" and not np.isclose(prior["settings"][key],opts[key]):
                raise ValueError(f"S1 reference setting mismatch: {key}")
        # Original H archive (or H ablation W5 with same settings).
        arms=[("H1_joint_HE","he"),("H2_joint_scene" if "H2_joint_scene" in prior["experiments"] else "H2_joint_scene_w5","scene")]
        for name,kind in arms:
            r=prior["experiments"][name]
            if not r["success"] or r.get("he_block_robust",False) or r.get("rotation_multiplier",1.)!=1.:
                raise ValueError(f"S1 needs successful unattenuated original H arm: {name}")
            for key,val in r.get("fit_settings",{}).items():
                if key in opts and key!="max_nfev" and not np.isclose(val,opts[key]):
                    raise ValueError(f"S1 arm setting mismatch: {name}, {key}")
            if r.get("window",5)!=5 or r.get("ground_weight",opts["ground_weight"])!=opts["ground_weight"]:
                raise ValueError("S1 arm window/ground mismatch")
            Y=validate_extrinsic(np.asarray(r["final"]["T_camera_lidar"],float))
            est=np.asarray(r["final"]["scales"],float)
            x=[Rotation.from_matrix(Y[:3,:3]@X[:3,:3].T).as_rotvec(),Y[:3,3]-X[:3,3],np.log(est/scales)]
            for k,s in enumerate(segments):
                with np.load(ref/name/f"segment_{k:02d}.npz",allow_pickle=False) as z:
                    for key,value in (("frames",s.frames),("point_ids",s.point_ids),
                                      ("obs_camera_index",s.obs_cam),("obs_landmark_index",s.obs_pt),("obs_uv",s.uv)):
                        if not np.array_equal(z[key],value):
                            raise ValueError(f"S1 archived support mismatch: {name}, {k}, {key}")
                    local=np.asarray(z["local_parameters"],float)
                    if local.shape!=(s.size,) or not np.isfinite(local).all(): raise ValueError("Invalid archived local state")
                    R,t,P=s.state(local)
                    for key,value in (("camera_R_anchor",R),("camera_t_anchor_ref_metric",t),
                                      ("landmark_xyz_ref_metric",P),("scale_ref",scales[k]),("scale_est",est[k])):
                        if not np.allclose(z[key],value,rtol=1e-9,atol=1e-8):
                            raise ValueError(f"S1 archived state mismatch: {name}, {k}, {key}")
                    x.append(local)
            result,_=solve_local_ba(segments,X,scales,he,kind=kind,**opts,
                      scale_diagnostic_state=np.concatenate(x),information_maxiter=args.information_maxiter)
            # Exact objective reproduction gate: do not interpret information
            # if the current code/support differs from archived H.
            for state in ("initial","final"):
                if not np.isclose(result[state]["cost"],r[state]["cost"],rtol=1e-7,atol=1e-9):
                    raise ValueError(f"S1 objective mismatch: {name}, {state}")
            result["reference_arm"]=name
            result["reference_cost_verified"]=True
            result["source_active_bounds_reported"]=bool(r.get("bound_hit"))
            if gt is not None: result["gt_evaluation"]=evaluate_camera_lidar(Y,gt)
            summary["experiments"][name]=result
            write_json(args.out/"summary.json",summary)
        return 0
    if args.line_data:
        # Sequential H2 baseline (the exact M0 initializer), not a refit.
        for name,Y,result in (
                ("M1a_H2_reference",line_X,{"success":True,"reference_only":True}),
                ("M1b_H2_xz",*fit_lines(line_train,line_X,line_K,mode="xz",sigma_px=args.line_sigma_px))):
            result["final"]={"T_camera_lidar":Y.tolist(),"scales":arm["final"]["scales"]}
            result["line_train"]=line_metrics(Y,line_train,line_K)
            result["line_test"]=line_metrics(Y,line_test,line_K)
            if gt is not None: result["gt_evaluation"]=evaluate_camera_lidar(Y,gt)
            summary["experiments"][name]=result
        write_json(args.out/"summary.json",summary)
        # Matched baselines for both fixed and free visual states.
        fit("M1c_fixed_HE_w0","he",zeros)
        fit("M1d_joint_HE_w0","he")
        fit("M1d_joint_scene_w0","scene")
        for w in args.line_weights:
            lw=dict(line_weight=w,line_sigma_px=args.line_sigma_px)
            fit(f"M1c_fixed_HE_w{w:g}","he",zeros,**lw)
            fit(f"M1d_joint_HE_w{w:g}","he",**lw)
            fit(f"M1d_joint_scene_w{w:g}","scene",**lw)
    elif args.ij:
        if args.ij_mode in ("both","I"):
            info=dict(information=True,information_maxiter=args.information_maxiter)
            fit("I1_HE_noground","he",ground_weight=0.,**info)
            fit("I2_HE_ground","he",**info)
            fit("I3_scene_noground","scene",ground_weight=0.,**info)
            fit("I4_scene_ground","scene",**info)
        if args.ij_mode in ("both","J"):
            controls,meta=virtual_moments([s.factors for s in by_window[5]],args.seed,args.shuffle_repeats)
            summary["virtual_moments"]=meta
            fit("J0_HE_block","he",he_block_robust=True)
            fit("J1_real_w1","scene",window=1)
            if args.ij_mode=="both":
                # Identical objective/state/settings; alias the saved I4 result,
                # do not run again or count it as independent evidence.
                import shutil
                shutil.copytree(args.out/"I4_scene_ground",args.out/"J2_real_w5")
                summary["experiments"]["J2_real_w5"]={**summary["experiments"]["I4_scene_ground"],"alias_of":"I4_scene_ground"}
                write_json(args.out/"summary.json",summary)
            else: fit("J2_real_w5","scene",window=5)
            for name,fs in controls.items():
                fit(name,"scene",custom_support=[replace(s,factors=f) for s,f in zip(segments,fs)])
            # Small-angle equivalent to HE block whitening: translation sigma
            # sigma_t, isotropic alpha=(sigma_t/sigma_R)^2/2, zero centroid.
            # Finite rotations use chordal rather than log residual.
            alpha=(he.sigma_relative_translation_m/np.deg2rad(he.sigma_relative_rotation_deg))**2/2
            fs=[moment_factors(np.zeros((len(s.centers),3)),np.broadcast_to(np.eye(3)*alpha,(len(s.centers),3,3))) for s in segments]
            summary["virtual_moments"]["HE_matched_isotropic_alpha_m2"]=float(alpha)
            fit("J0b_HE_matched_chordal","scene",custom_support=[replace(s,factors=f) for s,f in zip(segments,fs)],sigma_scene=he.sigma_relative_translation_m)
    elif args.ablation_windows:
        fit("Hfixed_HE","he",zeros)
        for w in windows: fit(f"Hfixed_scene_w{w}","scene",zeros,window=w)
        local=fit("H0_visual","visual",max_nfev=args.visual_max_nfev)
        fit("H0_then_HE","he",local)
        for w in windows: fit(f"H0_then_scene_w{w}","scene",local,window=w)
        fit("H1_joint_HE","he")
        for w in windows: fit(f"H2_joint_scene_w{w}","scene",window=w)
        if args.extended_ablation:
            fit("H1_HE_block","he",he_block_robust=True)
            fit("H1_HE_noground","he",ground_weight=0.)
            fit("H1_HE_rotlow","he",rotation_multiplier=args.rotation_multiplier)
            for w in windows:
                fit(f"H2_scene_w{w}_noground","scene",window=w,ground_weight=0.)
                fit(f"H2_scene_w{w}_rotlow","scene",window=w,rotation_multiplier=args.rotation_multiplier)
    else:
        fit("Hfixed_HE","he",zeros)
        fit("Hfixed_scene","scene",zeros)
        local=fit("H0_visual","visual")
        fit("H0_then_HE","he",local)
        fit("H0_then_scene","scene",local)
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
