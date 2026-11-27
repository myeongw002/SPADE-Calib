#!/usr/bin/env python3
"""OPT1: fixed-feature extrinsic-only optimization, no GT-guided primary fitting."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
from run_edge_points import read,write,sha,image,overlay,panel
from spade_calib.selection.scale_segmentation import validate_extrinsic
from spade_calib.calibration.multiview_edge_evidence import projected_candidates
from spade_calib.calibration.fixed_edge_refinement import (
    Frame,MODES,edge_distance_field,transform_from_delta,delta_from_transform,
    extrinsic_error,residual_frame,evaluate,fit,residuals,robust_values,
)

DEFAULT_ARMS=["MV1_all","MV1_A","MV1_C_HV","Surface_HV","MV1_A_plus_MV7_V"]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mv8",type=Path,default=Path("outputs/edgepoints07_MV8"))
    p.add_argument("--ep0",type=Path,default=Path("outputs/edgepoints07_EP0"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--arms",nargs="+",default=DEFAULT_ARMS)
    p.add_argument("--modes",nargs="+",choices=list(MODES),default=list(MODES))
    p.add_argument("--holdout",nargs="*",type=int,default=[289,819],
                   help="Predefined pilot validation anchors; empty means fit all")
    p.add_argument("--scale",type=float,default=5.,help="soft_l1 scale in pixels")
    p.add_argument("--rot-bound",type=float,default=2.,help="degrees per camera-axis rotvec coordinate")
    p.add_argument("--trans-bound",type=float,default=1.,help="meters per camera translation coordinate")
    p.add_argument("--max-nfev",type=int,default=150)
    p.add_argument("--border",type=float,default=5.,help="Fixed initial image margin, px")
    p.add_argument("--grid-n",type=int,default=21,help="XZ landscape grid size per axis; 0 disables")
    p.add_argument("--no-oracle",action="store_true")
    p.add_argument("--no-overlays",action="store_true")
    a=p.parse_args(argv)
    if (not np.isfinite([a.scale,a.rot_bound,a.trans_bound,a.border]).all()
        or min(a.scale,a.rot_bound,a.trans_bound)<=0 or a.border<0 or a.max_nfev<1
        or not (a.grid_n==0 or 3<=a.grid_n<=101) or len(set(a.arms))!=len(a.arms)
        or len(set(a.modes))!=len(a.modes)):
        p.error("Invalid settings")
    if a.out.exists() and (not a.out.is_dir() or any(a.out.iterdir())):
        raise FileExistsError("Use a fresh output directory")
    mv=read(a.mv8/"summary.json");ep=read(a.ep0/"summary.json")
    if mv["sources"]["ep0_summary_sha256"]!=sha(a.ep0/"summary.json"):
        raise ValueError("MV8/EP0 provenance mismatch")
    K=np.asarray(ep["K"],float)
    if not np.allclose(K,mv["K"]) or not np.allclose(K[2],[0,0,1]):
        raise ValueError("Intrinsics mismatch")
    X0=validate_extrinsic(np.asarray(mv["transforms"]["B4c"],float))
    if not np.allclose(X0,ep["transforms"]["B4c"],rtol=0,atol=1e-9):
        raise ValueError("B4c mismatch")
    # GT only used for reporting/explicit diagnostic starts, never for support.
    gt=validate_extrinsic(np.asarray(mv["transforms"]["GT_oracle"],float))
    if not np.allclose(gt,ep["transforms"]["GT_oracle"],rtol=0,atol=1e-9):
        raise ValueError("GT mismatch")
    frames=sorted(map(int,mv["frames"]))
    hold=sorted(set(a.holdout));train=[f for f in frames if f not in hold]
    if not train or any(f not in frames for f in hold):
        p.error("Invalid holdout; specify existing anchors, leave at least one train")
    max_depth=float(mv["settings"]["max_depth_m"]);margin=float(mv["settings"]["visibility_margin_m"])
    a.out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    summary=dict(warning="Pilot, no physical matches; fixed B4c visibility support may bias objective. "
                 "GT is evaluation only in primary B4c starts; oracle runs explicitly GT-initialized. "
                 "No ground/HE residual, pose/scale updates or GT tuning. Holdout anchors from same sequence, not independent sequences. "
                 "Bilinear DA2 distance and overflow penalties are heuristics; fixed support may become occluded after motion.",
                 sources=dict(mv8=str(a.mv8),ep0=str(a.ep0),mv8_summary_sha256=sha(a.mv8/"summary.json"),
                              ep0_summary_sha256=sha(a.ep0/"summary.json")),
                 settings=dict(arms=a.arms,modes=a.modes,train=train,holdout=hold,loss="soft_l1",scale_px=a.scale,
                               rot_bound_deg=a.rot_bound,trans_bound_m=a.trans_bound,max_nfev=a.max_nfev,
                               initial_border_px=a.border,grid_n=a.grid_n,max_depth_m=max_depth,
                               initial_visibility_margin_m=margin,oracle=not a.no_oracle,
                               parameter_order=["rx_deg","ry_deg","rz_deg","tx_m","ty_m","tz_m"],
                               support="B4c projection, nearest per pixel, EP0 visibility, fixed thereafter",
                               cost="mean soft_l1 rho(distance^2), pooled fixed points; no frame balancing"),
                 K=K.tolist(),initial_T_camera_lidar=X0.tolist(),gt_T_camera_lidar=gt.tolist(),
                 initial_error=extrinsic_error(X0,gt),arms={})
    # Cache one field and the full real-point array per frame; no GT projections.
    cache={}
    for f in frames:
        src=a.mv8/f"frame_{f:06d}";base=a.ep0/f"frame_{f:06d}"
        with np.load(src/"source_features.npz",allow_pickle=False) as z:xyz=np.asarray(z["xyz_lidar"],float)
        with np.load(base/"camera_jump.npz",allow_pickle=False) as z:cam=np.asarray(z["uv"],float)
        with np.load(base/"B4c_sparse_depth.npz",allow_pickle=False) as z:depth=z["depth_m"]
        im=image(base/"original.png");shape=im.shape[:2]
        if depth.shape!=shape or xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():
            raise ValueError(f"Bad frame {f}")
        cache[f]=(xyz,cam,depth,im,edge_distance_field(cam,shape))
    gt_delta=delta_from_transform(X0,gt)
    for arm in a.arms:
        folder=a.out/arm;folder.mkdir();support={};counts={}
        for f in frames:
            xyz,cam,depth,im,field=cache[f]
            src=a.mv8/f"frame_{f:06d}"
            with np.load(src/f"{arm}_source_ids.npz",allow_pickle=False) as z:ids=z["feature_indices"]
            if (ids.ndim!=1 or ids.dtype.kind not in "iu" or np.any((ids<0)|(ids>=len(xyz)))
                or len(np.unique(ids))!=len(ids)
                or len(ids)!=mv["frames"][str(f)]["source_candidates"][arm]):
                raise ValueError(f"Bad source IDs: {f} {arm}")
            h,w=im.shape[:2]
            q=projected_candidates(xyz[ids],X0,K,(h,w),depth,max_depth=max_depth,margin=margin)
            visible=q["visible"];uv=q["uv"]
            interior=(uv[:,0]>=a.border)&(uv[:,0]<=w-1-a.border)&(uv[:,1]>=a.border)&(uv[:,1]<=h-1-a.border)
            sel=ids[q["candidate_indices"][visible&interior]]
            if not len(sel):raise ValueError(f"No fixed support: {f} {arm}")
            support[f]=Frame(f,xyz[sel],field)
            counts[str(f)]=dict(source=len(ids),initial_projected=len(uv),initial_visible=int(visible.sum()),fixed=len(sel))
            np.savez_compressed(folder/f"frame_{f:06d}_fixed.npz",feature_indices=sel,xyz_lidar=xyz[sel])
        tr=[support[f] for f in train];te=[support[f] for f in hold]
        def report(X):
            return dict(T_camera_lidar=X.tolist(),error=extrinsic_error(X,gt),
                        train=evaluate(tr,X,K,max_depth,a.scale),holdout=evaluate(te,X,K,max_depth,a.scale))
        info=dict(support=counts,initial=report(X0),gt_diagnostic=report(gt),fits={},landscape=None)
        for mode in a.modes:
            active=np.asarray(MODES[mode]);d=np.zeros(6);d[active]=gt_delta[active]
            mode_info=dict(constrained_gt_diagnostic=report(transform_from_delta(X0,d)))
            starts={"B4c":np.zeros(6)}
            if not a.no_oracle:starts["GT_constrained_oracle"]=d
            for label,start in starts.items():
                bound=np.r_[np.full(3,a.rot_bound),np.full(3,a.trans_bound)]
                if np.any(abs(start)>bound+1e-10):
                    mode_info[label]=dict(skipped="Oracle outside B4c-centered bounds")
                    continue
                X,solver=fit(tr,X0,K,mode,start,max_depth=max_depth,scale=a.scale,
                             rot_bound=a.rot_bound,trans_bound=a.trans_bound,max_nfev=a.max_nfev)
                result=report(X);result["solver"]=solver
                result["start"]=report(transform_from_delta(X0,start))
                mode_info[label]=result
                np.savetxt(folder/f"T_{mode}_{label}.txt",X,fmt="%.12f")
                print(f"{arm} {mode} {label}: t={result['error']['translation_m']*100:.2f}cm "
                      f"R={result['error']['rotation_deg']:.4f}deg cost={result['train']['cost']:.3f} "
                      f"success={solver['success']}",flush=True)
                if not a.no_overlays and label=="B4c":
                    for f in frames:
                        fr=support[f];_,cam,_,im,_=cache[f]
                        _,v0,u0=residual_frame(fr,X0,K,max_depth)
                        _,v,u=residual_frame(fr,X,K,max_depth)
                        ref=overlay(im.copy(),cam,(0,255,0))
                        before=panel(overlay(ref.copy(),u0[v0],(0,165,255)),"B4c fixed support")
                        after=panel(overlay(ref.copy(),u[v],(0,165,255)),f"{arm} {mode} {'holdout' if f in hold else 'train'}")
                        if not cv2.imwrite(str(folder/f"frame_{f:06d}_{mode}.jpg"),np.vstack([before,after])):
                            raise OSError("Could not save overlay")
            info["fits"][mode]=mode_info
        if a.grid_n:
            # Diagnostic only, never used as a restart or to select a solution.
            axis=np.linspace(-a.trans_bound,a.trans_bound,a.grid_n)
            grid=np.empty((len(axis),len(axis)));test=np.full_like(grid,np.nan)
            for iz,dz in enumerate(axis):
                for ix,dx in enumerate(axis):
                    X=transform_from_delta(X0,[0.,0.,0.,dx,0.,dz])
                    grid[iz,ix]=robust_values(residuals(tr,X,K,max_depth),a.scale).mean()
                    if te:test[iz,ix]=robust_values(residuals(te,X,K,max_depth),a.scale).mean()
            iz,ix=np.unravel_index(np.argmin(grid),grid.shape)
            best=transform_from_delta(X0,[0.,0.,0.,axis[ix],0.,axis[iz]])
            np.savez_compressed(folder/"xz_landscape.npz",dx_m=axis,dz_m=axis,train_cost=grid,holdout_cost=test)
            info["landscape"]=dict(dx_m=axis.tolist(),dz_m=axis.tolist(),train_cost=grid.tolist(),
                                   holdout_cost=test.tolist() if te else None,
                                   grid_min_delta_xz_m=[float(axis[ix]),float(axis[iz])],
                                   grid_min=report(best),gt_delta_xz_m=gt_delta[[3,5]].tolist(),
                                   note="Fixed B4c rotation and ty; diagnostic grid minimum not a fitted solution.")
        summary["arms"][arm]=info
        write(folder/"summary.json",info);write(a.out/"summary.json",summary)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
