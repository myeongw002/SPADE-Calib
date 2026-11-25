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
    windows=sorted(set(args.ablation_windows or [args.window]))
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
    def fit(name,kind,fixed=None,window=None,**overrides):
        w=args.window if window is None else window
        support=by_window[w]
        settings={**opts,**overrides}
        LOG.info("Optimizing %s",name);start=perf_counter()
        result,local=solve_local_ba(support,X,scales,he,kind=kind,fixed_local=fixed,**settings)
        result["window"]=w;result["fit_settings"]=settings
        states=[s.state(x) for s,x in zip(support,local)]
        ratio=np.asarray(result["final"]["scales"])/scales
        Y=np.array(result["final"]["T_camera_lidar"])
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
    if args.ablation_windows:
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
