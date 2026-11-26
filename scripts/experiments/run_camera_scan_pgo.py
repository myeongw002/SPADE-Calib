#!/usr/bin/env python3
"""PGO pilot: camera-initialized real scan edges and matched GLIM control."""
from __future__ import annotations
import argparse,json,hashlib
from dataclasses import asdict
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
from spade_calib.io.kitti import load_kitti_trajectory_data,frame_id_from_name
from spade_calib.io.kitti_bin import load_kitti_bin_scan
from spade_calib.selection.scale_segmentation import ScalePartition,build_frame_lag_pairs,validate_extrinsic
from spade_calib.selection.trajectory import MotionPairs
from spade_calib.handeye.residuals import residuals
from spade_calib.handeye.solver import HandEyeSettings
from spade_calib.calibration.segmented_ground import _robust
from spade_calib.calibration.camera_scan_pgo import (pose,inverse,relative,edge_error,Graph,solve,
    RegistrationSettings,prepare_scan,register)
from spade_calib.handeye.trajectory_oracle import kitti_oracle_pairs
from spade_calib.evaluation.kitti import parse_kitti_odom_calib,evaluate_camera_lidar


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("run","out"):p.add_argument("--"+arg,type=Path,required=True)
    for arg in ("sequence","glim","sfm","lidar-dir","gt-calib","gt-poses"):p.add_argument("--"+arg,type=Path)
    p.add_argument("--stride",type=int,default=5,help="Matched pose stride, not raw frame lag")
    p.add_argument("--weight",type=float,default=1.)
    p.add_argument("--max-nfev",type=int,default=100)
    p.add_argument("--full-pose",action="store_true")
    args=p.parse_args(argv)
    if bool(args.gt_calib)!=bool(args.gt_poses) or args.stride<1 or args.max_nfev<1 or not np.isfinite(args.weight) or args.weight<=0:raise ValueError("Invalid options")
    if args.out.exists() and any(args.out.iterdir()):raise FileExistsError("Fresh output required")
    read=lambda p:json.loads(p.read_text())
    cfg=read(args.run/"config.json");saved=read(args.run/"segmented_handeye.json")["fit"]
    rows=saved["segments"];X=validate_extrinsic(np.array(saved["T_camera_lidar"]))
    partition=ScalePartition(tuple([0]+[int(r["stop_index_exclusive"]) for r in rows]))
    samples=load_kitti_trajectory_data(args.glim or cfg["glim"],args.sfm or cfg["sfm"],
                Path(args.sequence or cfg["sequence"])/"times.txt",glim_first_frame=int(cfg["glim_first_frame"]))
    if list(samples.image_names)!=[n for r in rows for n in r["image_names"]] or samples.timestamps_ns.tolist()!=[n for r in rows for n in r["timestamps_ns"]]:
        raise ValueError("Archived matched inputs differ")
    he=HandEyeSettings(**saved["settings"]);pairs=build_frame_lag_pairs(samples,int(cfg["lag"]))
    selected=np.asarray(saved["selected_pair_rows"]);groups=partition.groups(pairs)
    if selected.ndim!=1 or selected.dtype.kind not in "iu" or len(np.unique(selected))!=len(selected) or np.any(selected<0) or np.any(selected>=pairs.count) or np.any(groups[selected]<0):
        raise ValueError("Invalid archive selection")
    cost=0.
    sigma=np.r_[np.full(3,np.deg2rad(he.sigma_relative_rotation_deg)),np.full(3,he.sigma_relative_translation_m)]
    for k,(a,b) in enumerate(partition.intervals):
        row=rows[k];idx=selected[groups[selected]==k]
        if row["segment_id"]!=k or row["start_index"]!=a or len(idx)!=row["motion_pair_count"]:raise ValueError("Archive partition mismatch")
        q=MotionPairs(pairs.Ra[idx],pairs.ta[idx],pairs.Rb[idx],pairs.tb[idx],pairs.indices[idx],pairs.starts_ns[idx])
        r=residuals(q,X[:3,:3],X[:3,3],float(row["sfm_scale_to_metric"]))/sigma
        cost+=.5*float(np.sum(_robust(r,he.robust_loss)**2))
    if not np.isclose(cost,saved["optimization"]["cost_whitened_robust"],rtol=1e-5,atol=1e-7):raise ValueError("Archived HE cost mismatch")
    args.out.mkdir(parents=True,exist_ok=True)
    reg=RegistrationSettings()
    summary=dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},registration=asdict(reg),
        source_sha256={k:hashlib.sha256((args.run/k).read_bytes()).hexdigest() for k in ("config.json","segmented_handeye.json")},
        archived_cost_verified=True,segments=[],experiments={},
        warning="Rigid synthetic16 scans, no deskew or dynamic filtering. Registration initialized from estimated camera and X; GLIM seed diagnostic not independent. Camera tether from original SfM, not new visual evidence. Fixed registration Hessian is local, not covariance.")
    def write(): (args.out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    node_ids=[];Cs=[];Ls=[];segments=[];anchors=[];ce=[];le=[];As=[];Fs=[];Ag=[];records=[]
    lidar=Path(args.lidar_dir or cfg["lidar_dir"])
    for k,(a,b) in enumerate(partition.intervals):
        anchor=(a+b-1)//2
        ids=np.array(sorted(set(range(a,b,args.stride))|{anchor,b-1}))
        frames=[frame_id_from_name(samples.image_names[i]) for i in ids]
        n=len(ids);offset=len(node_ids);anchors.append(offset+int(np.where(ids==anchor)[0][0]))
        R=samples.camera_R_world_sfm[anchor]
        C=pose(R.T@samples.camera_R_world_sfm[ids],float(rows[k]["sfm_scale_to_metric"])*(samples.camera_t_world_sfm[ids]-samples.camera_t_world_sfm[anchor])@R)
        L=pose(samples.lidar_R_world[ids],samples.lidar_t_world_m[ids])
        node_ids.extend(ids.tolist());Cs.extend(C);Ls.extend(L);segments.extend([k]*n)
        scans=[]
        for f,i in zip(frames,ids):
            scan=load_kitti_bin_scan(lidar/f"{f:06d}.bin",scan_id=0,timestamp_ns=int(samples.timestamps_ns[i]),max_range_m=float(cfg["max_range"]))
            scans.append(prepare_scan(scan.xyz,reg,17+f))
        ij=np.array([(i,i+d) for d in (1,2) for i in range(n-d)],int)
        B=relative(C,ij);G=relative(L,ij);initial=inverse(X)@B@X
        ce.extend((ij+offset).tolist())
        counts=Counter()
        for e,(i,j) in enumerate(ij):
            r=register(scans[i],scans[j],initial[e],reg)
            counts[r["status"]]+=1
            # GLIM-seeded repeat is diagnostic only; never changes primary mask.
            rg=register(scans[i],scans[j],G[e],reg)
            record=dict(segment=k,frames=[frames[i],frames[j]],nodes=[int(i+offset),int(j+offset)],camera_seed=r,glim_seed=rg,
                        initial_camera=initial[e].tolist(),initial_glim=G[e].tolist())
            if r["status"]=="ok":
                le.append([int(i+offset),int(j+offset)]);As.append(r["T"]);Fs.append(r["factor"]);Ag.append(G[e])
                if rg["status"]=="ok":
                    err=edge_error(np.asarray(r["T"])[None],np.asarray(rg["T"])[None])[0]
                    record["seed_disagreement_m"]=float(np.linalg.norm(err[3:]))
                    record["seed_disagreement_deg"]=float(np.rad2deg(np.linalg.norm(err[:3])))
            records.append(record)
        summary["segments"].append(dict(segment=k,frames=frames,anchor_frame=frame_id_from_name(samples.image_names[anchor]),registration_status=dict(counts)))
        (args.out/"registration_edges.json").write_text(json.dumps(records,indent=2,allow_nan=False)+"\n")
        write();print("Segment",k,dict(counts),flush=True)
    C=np.array(Cs);node_ids=np.array(node_ids);segments=np.array(segments);ce=np.array(ce);le=np.array(le).reshape(-1,2)
    if len(le)==0 or set(segments[le[:,0]])!=set(segments):
        summary["failure"]="At least one segment has no accepted scan registration edges; no joint fit performed";write();return 2
    A=np.array(As);F=np.array(Fs);Ag=np.array(Ag);anchors=np.array(anchors)
    graph=Graph(C,segments,anchors,ce,le,A,F);control=Graph(C,segments,anchors,ce,le,Ag,F)
    arms=[("initial",graph,"fixed",False),("reg_X",graph,"fixed",True),("reg_pose",graph,"translation",False),
          ("reg_joint",graph,"translation",True),("glim_X",control,"fixed",True),("glim_joint",control,"translation",True)]
    if args.full_pose:arms.extend([("reg_joint_full",graph,"full",True),("glim_joint_full",control,"full",True)])
    fitted={}
    for name,g,mode,free in arms:
        print("Fitting",name,flush=True)
        result,cp=solve(g,X,pose_mode=mode,free_X=free,weight=args.weight,max_nfev=args.max_nfev,
                       sigma_t=he.sigma_relative_translation_m,sigma_r_deg=he.sigma_relative_rotation_deg)
        result.update(pose_mode=mode,free_X=free,edge_source="GLIM" if g is control else "registration")
        fitted[name]=(result,cp);summary["experiments"][name]=result;write()
    # Evaluation only: never used by ICP or graph optimization.
    if args.gt_calib:
        allframes=np.array([frame_id_from_name(n) for n in samples.image_names])
        raw=np.loadtxt(args.gt_poses);cal=parse_kitti_odom_calib(args.gt_calib)
        def oracle(edges):
            ids=node_ids[edges];n=len(ids);I=np.tile(np.eye(3),(n,1,1));z=np.zeros((n,3))
            return kitti_oracle_pairs(raw,cal,int(cfg["camera_index"]),allframes,
                     MotionPairs(I,z,I,z,ids,samples.timestamps_ns[ids[:,0]]))
        gt,Xgt=oracle(ce);gtL,_=oracle(le);Bgt=pose(gt.Rb,gt.tb);Agt=pose(gtL.Ra,gtL.ta)
        def errors(p,q):
            e=edge_error(p,q);return dict(t_rmse_m=float(np.sqrt(np.mean(np.sum(e[:,3:]**2,axis=1)))),
                        r_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(np.sum(e[:,:3]**2,axis=1))))))
        summary["registration_GT"]=dict(camera_initial=errors(relative(C@X,le),Agt),registration=errors(A,Agt),GLIM=errors(Ag,Agt))
        for name,(result,cp) in fitted.items():
            xp=np.array(result["T_camera_lidar"])
            result["gt_extrinsic"]=evaluate_camera_lidar(xp,Xgt)
            result["camera_GT_RPE"]=errors(relative(cp,ce),Bgt)
            result["lidar_GT_RPE"]=errors(relative(cp@xp,le),Agt)
            print(name,result["gt_extrinsic"]["translation_error_total_m"],result["camera_GT_RPE"],flush=True)
    np.savez_compressed(args.out/"graph.npz",matched_indices=node_ids,frames=np.array([frame_id_from_name(samples.image_names[i]) for i in node_ids]),
         C0=C,segments=segments,anchors=anchors,camera_edges=ce,lidar_edges=le,registration_A=A,registration_F=F,glim_A=Ag,
         **{"C_"+name:cp for name,(_,cp) in fitted.items()})
    write();return 0


if __name__=="__main__":raise SystemExit(main())
