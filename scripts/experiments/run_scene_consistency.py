#!/usr/bin/env python3
"""G1-G3: fixed-pose segment sliding-window scene consistency ablation.

Reuses archived reconstruction, A0 map normals and B4-c ground support.
No new camera-LiDAR correspondences. GT evaluation only (optional).
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from run_segment_ground import read_xyz_ply,write_ply,write_json
from spade_calib.calibration.ground import Plane
from spade_calib.calibration.segmented_ground import SegmentGround,_robust
from spade_calib.calibration.scene_consistency import SceneSegment,fit_scene,scene_factors,voxel_return_indices
from spade_calib.handeye.residuals import residuals
from spade_calib.handeye.solver import HandEyeSettings
from spade_calib.io.kitti import load_kitti_trajectory_data,frame_id_from_name,read_kitti_times_ns
from spade_calib.io.kitti_bin import load_kitti_bin_scan
from spade_calib.lidar.trajectory import GlimTrajectory
from spade_calib.selection.scale_segmentation import ScalePartition,build_frame_lag_pairs,validate_extrinsic
from spade_calib.evaluation.kitti import evaluate_camera_lidar,kitti_gt_camera_lidar
from spade_calib.selection.trajectory import MotionPairs

LOG=logging.getLogger(__name__)


def normal_support(tree,p,neighbors=16):
    """Fixed A0 neighborhood PCA, normals assigned to exact raw return target."""
    dist,ids=tree.query(p,k=neighbors,workers=1)
    xyz=tree.data[ids]; centered=xyz-xyz.mean(axis=1,keepdims=True)
    cov=np.swapaxes(centered,1,2)@centered/neighbors
    e,V=np.linalg.eigh(cov)
    valid=((dist[:,0]<=.35)&(dist[:,-1]<=1.5)&(e[:,1]>=.0025)
           &(e[:,0]<=.2*e[:,1])&(e[:,1]>=.05*e[:,2]))
    return valid,V[:,:,0]


def full_normal_support(tree,p,chunk=4096):
    """Bound temporary PCA allocations; no filtering or sampling here."""
    valid=np.empty(len(p),dtype=bool)
    normals=np.empty((len(p),3),dtype=np.float32)
    for start in range(0,len(p),chunk):
        end=min(start+chunk,len(p))
        m,n=normal_support(tree,p[start:end])
        valid[start:end]=m
        normals[start:end]=n
    return valid,normals


def voxel_support(cache,window,Ra,ta,voxel):
    """Accumulate ALL range-valid returns, voxelize once, then normal gate.

    Cached XYZ are in anchor LiDAR axes. Normal validity is computed once
    per original return but is NOT applied before voxel selection.
    """
    xyz=np.concatenate([cache[f][0] for f in window])
    q=(xyz-ta)@Ra
    ids=voxel_return_indices(q,voxel)
    v=np.concatenate([cache[f][3] for f in window])
    keep=ids[v[ids]]
    n=np.concatenate([cache[f][1] for f in window])
    src=np.concatenate([np.column_stack((np.full(len(cache[f][2]),f,dtype=np.int64),cache[f][2]))
                        for f in window])
    return q[keep],n[keep].astype(float),src[keep],len(q),len(ids)


def choose_support(cache,window,cap,seed):
    """Balance source scans, cap equal total budget per camera center.

    cache values are (anchor XYZ, anchor normals, original return IDs).
    Every selected return is unique within this center; may reappear in
    neighboring centers. Deterministic per-center/window selection.
    """
    active=[f for f in window if len(cache[f][0])]
    if not active:
        raise ValueError("No valid normal support in camera window")
    rng=np.random.default_rng(seed)
    # Round-robin quotas, then redistribute unused quota to available scans.
    perm={f:rng.permutation(len(cache[f][0])) for f in active}
    counts={f:0 for f in active}
    remaining=min(cap,sum(len(perm[f]) for f in active))
    while remaining:
        for f in active:
            if counts[f]<len(perm[f]) and remaining:
                counts[f]+=1;remaining-=1
    p=[];n=[];src=[]
    for f in active:
        ii=perm[f][:counts[f]]
        x,v,ret=cache[f]
        p.append(x[ii]);n.append(v[ii]);src.append(np.column_stack((np.full(len(ii),f),ret[ii])))
    return np.concatenate(p),np.concatenate(n),np.concatenate(src)


def load_ground(path,run,X,s,cost):
    d=json.loads((path/"summary.json").read_text())
    if (d["arm"]!="A0" or not np.allclose(d["baseline"]["T_camera_lidar"],X,atol=1e-9)
            or not np.allclose(d["baseline"]["scales"],s,rtol=1e-10)
            or not np.isclose(d["archived_HE_cost"],cost,rtol=1e-9)
            or Path(d["run"]).name!=run.name):
        raise ValueError("Ground support does not match archived A0 HE")
    matches=[]
    for r in d["ground_diagnostics"]:
        if r["accepted"]:
            matches.append(SegmentGround(int(r["segment_id"]),
                Plane(r["lidar"]["normal"],r["lidar"]["offset_m"]),
                Plane(r["camera"]["normal"],r["camera"]["offset_m"])))
    if sorted(m.segment_id for m in matches)!=sorted(d["accepted_ground_segments"]) or not matches:
        raise ValueError("Ground match identity mismatch")
    return matches,d


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--ground-run",type=Path,help="Existing B4/B4c summary; enables G3")
    for name in ("sequence","glim","sfm","lidar-dir","gt-calib","image-dir"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--windows",type=int,nargs="+",default=[1,3,5,11])
    p.add_argument("--center-stride",type=int,default=1,help="Stride in matched camera frames")
    p.add_argument("--support-mode",choices=["voxel","sampled"],default="voxel",
                   help="voxel: all raw scans then one local voxel grid; sampled: legacy G")
    p.add_argument("--local-voxel",type=float,default=None,help="Local LiDAR voxel side in m; default archived A0 voxel")
    p.add_argument("--source-cap",type=int,default=512,help="Legacy sampled mode only")
    p.add_argument("--points-per-center",type=int,default=128)
    p.add_argument("--min-points",type=int,default=32)
    p.add_argument("--sigma-scene",type=float,default=.15)
    p.add_argument("--ground-weight",type=float,default=.1)
    p.add_argument("--sigma-ground",type=float,default=.15)
    p.add_argument("--max-rotation-deg",type=float,default=5.)
    p.add_argument("--max-translation-m",type=float,default=1.)
    p.add_argument("--max-log-scale",type=float,default=.15)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--linear",action="store_true",help="Unrobustified control")
    p.add_argument("--oracle",action="store_true",help="GT X start diagnostic; fixed supports")
    p.add_argument("--export-maps",action="store_true",help="Sampled reference and camera-driven map PLYs")
    args=p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format="%(levelname)s %(message)s")
    if (not args.windows or len(set(args.windows))!=len(args.windows) or any(w<1 or w%2==0 for w in args.windows)
            or args.center_stride<1 or args.min_points<3
            or (args.support_mode=="sampled" and not args.min_points<=args.points_per_center<=args.source_cap)
            or (args.local_voxel is not None and (not np.isfinite(args.local_voxel) or args.local_voxel<=0))
            or args.seed<0 or (args.oracle and args.gt_calib is None)):
        raise ValueError("Invalid window/sample settings")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    run=args.run
    cfg=json.loads((run/"config.json").read_text())
    voxel=float(cfg["voxel"]) if args.local_voxel is None else args.local_voxel
    if not np.isfinite(voxel) or voxel<=0:
        raise ValueError("Invalid local voxel")
    if args.support_mode=="voxel" and args.export_maps:
        LOG.warning("Voxel mode exports all retained returns for all arms; memory and disk usage may be large")
    saved=json.loads((run/"segmented_handeye.json").read_text())["fit"]
    archived=json.loads((run/"summary.json").read_text())
    rows=saved["segments"]; K=len(rows)
    partition=ScalePartition(tuple([0]+[int(r["stop_index_exclusive"]) for r in rows]))
    X=validate_extrinsic(np.array(saved["T_camera_lidar"]))
    s=np.array([r["sfm_scale_to_metric"] for r in rows])
    he=HandEyeSettings(**saved["settings"])
    glim_path=args.glim or Path(cfg["glim"])
    times_path=Path(args.sequence or cfg["sequence"])/"times.txt"
    scan_dir=args.lidar_dir or Path(cfg["lidar_dir"])
    first=int(cfg["glim_first_frame"])
    samples=load_kitti_trajectory_data(glim_path,args.sfm or cfg["sfm"],times_path,glim_first_frame=first)
    if (list(samples.image_names)!=[n for r in rows for n in r["image_names"]]
            or samples.timestamps_ns.tolist()!=[n for r in rows for n in r["timestamps_ns"]]):
        raise ValueError("Archived matched frame identities mismatch")
    frames=np.array([frame_id_from_name(n) for n in samples.image_names])
    pairs=build_frame_lag_pairs(samples,int(cfg["lag"]))
    selected=np.asarray(saved["selected_pair_rows"])
    if (selected.ndim!=1 or selected.dtype.kind not in "iu" or len(np.unique(selected))!=len(selected)
            or np.any(selected<0) or np.any(selected>=pairs.count)):
        raise ValueError("Invalid archived pair IDs")
    mask=np.zeros(pairs.count,dtype=bool);mask[selected]=True
    groups=partition.groups(pairs)
    if np.any(mask & (groups<0)): raise ValueError("Archived cross-segment pair")
    cost=0.
    for k in range(K):
        m=mask&(groups==k)
        if m.sum()!=rows[k]["motion_pair_count"]: raise ValueError("Archived pair count mismatch")
        part=MotionPairs(pairs.Ra[m],pairs.ta[m],pairs.Rb[m],pairs.tb[m],pairs.indices[m],pairs.starts_ns[m])
        r=residuals(part,X[:3,:3],X[:3,3],float(s[k]))
        r[:,:3]/=np.deg2rad(he.sigma_relative_rotation_deg);r[:,3:]/=he.sigma_relative_translation_m
        cost+=.5*np.sum(_robust(r,he.robust_loss)**2)
    expected=saved["optimization"]["cost_whitened_robust"]
    if not np.isclose(cost,expected,rtol=1e-5,atol=1e-7):
        raise ValueError("Archived HE cost mismatch: check GLIM/COLMAP inputs")
    times=read_kitti_times_ns(times_path)
    glim=GlimTrajectory.from_array(np.loadtxt(glim_path,ndmin=2))
    LR=Rotation.from_quat(glim.quaternions_xyzw).as_matrix()
    LT=glim.positions
    if first<0 or first+len(LR)>len(times):
        raise ValueError("GLIM row coverage mismatch")
    entries={int(e["segment_id"]):e for e in archived["segments"]}
    if sorted(entries)!=list(range(K)):
        raise ValueError("Require complete archived reconstruction")
    scans={}
    for path in scan_dir.glob("*.bin"):
        f=frame_id_from_name(path.name)
        if f in scans: raise ValueError("Duplicate raw scan frame")
        scans[f]=path
    windows=sorted(set(args.windows)|{1})
    support={w:[] for w in windows}
    report={"run":str(run),"settings":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            "local_voxel_m":voxel if args.support_mode=="voxel" else None,
            "support_mode":args.support_mode,
            "voxel_rule":"nearest actual return to local voxel centroid, then fixed A0 normal gate" if args.support_mode=="voxel" else None,
            "archived_cost":expected,"recomputed_cost":float(cost),"archived_pair_count":int(mask.sum()),
            "baseline":{"T_camera_lidar":X.tolist(),"scales":s.tolist()},
            "segments":[],"experiments":{},"oracle_experiments":{},
            "warning":"Scene objective reweights trajectory consistency, not new cross-modal observations. Fixed source IDs and A0 normals. Overlapping returns correlated."}
    args.out.mkdir(parents=True,exist_ok=True)
    gt=kitti_gt_camera_lidar(args.gt_calib,int(cfg["camera_index"])) if args.gt_calib else None
    if gt is not None:
        report["baseline"]["gt_evaluation"]=evaluate_camera_lidar(X,gt)
    ground=[];ground_doc=None
    if args.ground_run:
        ground,ground_doc=load_ground(args.ground_run,run,X,s,expected)
        for key,val in ground_doc["experiments"].items():
            if key=="B4c_w0.1" and val.get("success"):
                report["B4c_reference"]=val
    anchors={}
    for k,(a,b) in enumerate(partition.intervals):
        e=entries[k];anchor_idx=(a+b-1)//2;anchor=int(frames[anchor_idx])
        if (int(e["anchor_frame"])!=anchor or not np.isclose(e["scale"],s[k],rtol=1e-10)
                or e["arms"].get("A0",{}).get("status")!="ok"):
            raise ValueError("Archived A0 anchor/scale mismatch")
        anchors[anchor]=k
        source=sorted(map(int,e["scan_frames"]))
        if (not source or source[0]<frames[a] or source[-1]>frames[b-1]
                or any(f not in scans or not first<=f<first+len(LR) for f in source)):
            raise ValueError("Archived raw scan support missing")
        tree=cKDTree(read_xyz_ply(run/f"segment_{k:02d}"/"A0"/"cloud_voxel.ply").astype(float))
        if tree.n<16: raise ValueError("Insufficient A0 map support")
        R0,t0=LR[anchor-first],LT[anchor-first]
        cache={}
        for f in source:
            scan=load_kitti_bin_scan(scans[f],scan_id=0,timestamp_ns=int(times[f]),
                                    max_range_m=float(cfg["max_range"]))
            if args.support_mode=="voxel":
                p=(scan.xyz@LR[f-first].T+LT[f-first]-t0)@R0
                valid,n=full_normal_support(tree,p)
                cache[f]=(p,n,scan.source_return_index,valid)
            else:
                rng=np.random.default_rng([args.seed,f])
                ids=rng.choice(len(scan.xyz),min(len(scan.xyz),args.source_cap),replace=False)
                p=(scan.xyz[ids]@LR[f-first].T+LT[f-first]-t0)@R0
                valid,n=normal_support(tree,p)
                cache[f]=(p[valid],n[valid],scan.source_return_index[ids][valid])
        centers=[]
        per={w:{"q":[],"src":[],"factors":[],"counts":[],"raw":[],"voxel":[],"scan_counts":[],"used":{}} for w in windows}
        for i in range(a,b,args.center_stride):
            f=int(frames[i])
            if f==anchor or f not in cache: continue
            # Same camera centers; voxel mode intentionally does NOT match point counts.
            # Clip raw-frame windows at segment boundary; no cross-segment scan.
            Ra=R0.T@LR[f-first];ta=R0.T@(LT[f-first]-t0)
            cand={}
            for w in windows:
                use=[j for j in range(f-w//2,f+w//2+1) if j in cache]
                if sum(len(cache[j][0]) for j in use)<args.min_points: break
                if args.support_mode=="voxel":
                    q,n,src,raw_count,voxel_count=voxel_support(cache,use,Ra,ta,voxel)
                    if len(q)<args.min_points: break
                    cand[w]=(q,n,src,raw_count,voxel_count)
                else:
                    xyz,n,src=choose_support(cache,use,args.points_per_center,[args.seed,f,w])
                    cand[w]=((xyz-ta)@Ra,n,src,None,None)
            if len(cand)!=len(windows): continue
            common=min(len(v[0]) for v in cand.values())
            centers.append(i)
            for w,(q,n,src,raw_count,voxel_count) in cand.items():
                if args.support_mode=="sampled":
                    keep=np.random.default_rng([args.seed,f,w,1]).choice(len(q),common,replace=False)
                    q,n,src=q[keep],n[keep],src[keep]
                d=per[w]
                d["factors"].append(scene_factors(q,n))
                d["counts"].append(len(q))
                if raw_count is not None:
                    d["raw"].append(raw_count);d["voxel"].append(voxel_count)
                scan_ids=np.unique(src[:,0])
                d["scan_counts"].append(len(scan_ids))
                for j in scan_ids:
                    j=int(j)
                    if j not in d["used"]:
                        d["used"][j]=np.zeros(int(cache[j][2].max())+1,dtype=bool)
                    d["used"][j][src[src[:,0]==j,1]]=True
                if args.export_maps:
                    d["q"].append(q.astype(np.float32))
                    d["src"].append(src.astype(np.int32))
            if len(centers)%25==0:
                LOG.info("Segment %d: prepared %d centers",k,len(centers))
        if len(centers)<3: raise ValueError(f"Segment {k}: too few common centers")
        ci=np.asarray(centers,dtype=int)
        Ra=R0.T@samples.lidar_R_world[ci]
        ta=(samples.lidar_t_world_m[ci]-t0)@R0
        C0=samples.camera_R_world_sfm[anchor_idx]
        Rb=C0.T@samples.camera_R_world_sfm[ci]
        tb=(samples.camera_t_world_sfm[ci]-samples.camera_t_world_sfm[anchor_idx])@C0
        # Confirm the two GLIM access paths use identical pose rows.
        if (not np.allclose(samples.lidar_R_world[ci],LR[frames[ci]-first],atol=1e-8)
                or not np.allclose(samples.lidar_t_world_m[ci],LT[frames[ci]-first],atol=1e-8)):
            raise ValueError("Matched GLIM poses do not match raw GLIM rows")
        coverage={}
        for w in windows:
            d=per[w]
            seg=SceneSegment(k,frames[ci],Ra,ta,Rb,tb,d["q"],[],d["src"],
                             compact_factors=d["factors"],point_counts=np.array(d["counts"]))
            support[w].append(seg)
            coverage[str(w)]={"sampled_returns":int(sum(d["counts"])),
                              "unique_raw_returns":int(sum(np.count_nonzero(v) for v in d["used"].values())),
                              "mean_source_scans_per_center":float(np.mean(d["scan_counts"])),
                              "mean_retained_points_per_center":float(np.mean(d["counts"])),
                              "min_retained_points_per_center":int(min(d["counts"])),
                              "max_retained_points_per_center":int(max(d["counts"]))}
            if d["raw"]:
                coverage[str(w)].update(raw_accumulated_returns=int(sum(d["raw"])),
                                        occupied_voxels_before_normal_gate=int(sum(d["voxel"])),
                                        mean_raw_points_per_center=float(np.mean(d["raw"])),
                                        mean_voxels_before_normal_gate=float(np.mean(d["voxel"])))
        report["segments"].append(dict(segment_id=k,anchor_frame=anchor,
              selected_centers=frames[ci].tolist(),possible_centers=len(range(a,b,args.center_stride))-int(anchor_idx in range(a,b,args.center_stride)),
              raw_scans=len(source),normal_valid_returns=int(sum(np.count_nonzero(x[3]) if args.support_mode=="voxel" else len(x[0]) for x in cache.values())),
              windows=coverage))
        LOG.info("Segment %d: %d common centers",k,len(ci))
    write_json(args.out/"summary.json",report)
    opts=dict(sigma_scene=args.sigma_scene,sigma_ground=args.sigma_ground,
              max_rotation_deg=args.max_rotation_deg,max_translation_m=args.max_translation_m,
              max_log_scale=args.max_log_scale,linear=args.linear)
    def run_fit(name,segments,kind,refit,weight):
        kw=dict(kind=kind,refit_scales=refit,ground_matches=ground,ground_weight=weight,**opts)
        LOG.info("Optimizing %s",name)
        result=fit_scene(segments,X,s,he,**kw)
        if gt is not None and "final" in result:
            result["gt_evaluation"]=evaluate_camera_lidar(np.array(result["final"]["T_camera_lidar"]),gt)
        report["experiments"][name]=result
        if args.oracle:
            result_gt=fit_scene(segments,X,s,he,initial_X=gt,**kw)
            if "final" in result_gt:
                result_gt["gt_evaluation"]=evaluate_camera_lidar(np.array(result_gt["final"]["T_camera_lidar"]),gt)
            report["oracle_experiments"][name]=result_gt
        write_json(args.out/"summary.json",report)
        return result
    # Anchored-motion control: identical camera centers, same segment/center
    # weighting and bounds. Archived lag-10 HE uses a different motion set.
    run_fit("G0_anchor_fixed",support[1],"motion",False,0.)
    run_fit("G0_anchor_scale",support[1],"motion",True,0.)
    if ground:
        run_fit("G0_anchor_scale_ground",support[1],"motion",True,args.ground_weight)
    for w in args.windows:
        segments=support[w]
        if args.export_maps:
            for seg in segments:
                folder=args.out/"maps"/f"window_{w:02d}"/f"segment_{seg.segment_id:02d}"
                folder.mkdir(parents=True,exist_ok=True)
                p,q=seg.maps(X,s[seg.segment_id])
                write_ply(folder/"reference_sampled.ply",p);write_ply(folder/"HE_camera_map.ply",q)
                offsets=np.r_[0,np.cumsum([len(x) for x in seg.q])]
                np.savez_compressed(folder/"support.npz",center_frames=seg.frames,offsets=offsets,
                                    source_frame_return_ids=np.concatenate(seg.source_frames))
        for kind in ("point","plane"):
            for arm,refit,weight in [("G1",False,0.),("G2",True,0.)]+([("G3",True,args.ground_weight)] if ground else []):
                name=f"{arm}_{kind}_w{w}"
                r=run_fit(name,segments,kind,refit,weight)
                if args.export_maps and r.get("success"):
                    Y=np.asarray(r["final"]["T_camera_lidar"]);ss=r["final"]["scales"]
                    for seg in segments:
                        folder=args.out/"maps"/f"window_{w:02d}"/f"segment_{seg.segment_id:02d}"
                        _,q=seg.maps(Y,ss[seg.segment_id])
                        write_ply(folder/f"{name}_camera_map.ply",q)
    if args.image_dir:
        from spade_calib.evaluation.segment_projection import export_projections
        from spade_calib.evaluation.kitti import parse_kitti_odom_calib
        calib=args.gt_calib or times_path.parent/"calib.txt"
        Kcam=parse_kitti_odom_calib(calib,require_tr=False)[f"P{int(cfg['camera_index'])}"][:,:3]
        transforms={"HE":X}
        if "B4c_reference" in report:
            transforms["B4c"]=np.array(report["B4c_reference"]["final"]["T_camera_lidar"])
        for name,r in report["experiments"].items():
            # Show single-scan and largest-window G2/G3, not oracle starts.
            if name.startswith(("G2_","G3_")) and name.rsplit("_w",1)[-1] in {str(min(args.windows)),str(max(args.windows))} and r.get("success"):
                transforms[name]=np.array(r["final"]["T_camera_lidar"])
        if gt is not None: transforms["GT_reference"]=gt
        report["projection"]=export_projections(image_dir=args.image_dir,scan_dir=scan_dir,
                  run_dir=run,out_dir=args.out/"projection",anchors=anchors,extra_frames=set(),arm="A0",
                  cloud_mode="both",K=Kcam,transforms=transforms,read_xyz_ply=read_xyz_ply,
                  max_depth_m=float(cfg["max_range"]))
        write_json(args.out/"summary.json",report)
    LOG.info("Completed %s",args.out)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
