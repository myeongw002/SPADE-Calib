#!/usr/bin/env python3
"""MV7: visibility-only surface proxies on the fixed single-anchor U0.2 grid."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from run_edge_points import read,sha,write,image,save,overlay,panel,percentiles
from spade_calib.io.kitti import accumulate_kitti_rigid_window,read_kitti_times_ns
from spade_calib.io.kitti_bin import load_kitti_bin_scan
from spade_calib.lidar.trajectory import GlimTrajectory
from spade_calib.lidar.ghost_filter import FreeSpaceSettings,target_free_space_mask
from spade_calib.calibration.multiview_edge_evidence import (
    load_beams,recover_rings,ring_edges,recurrence,projected_candidates,
)
from spade_calib.selection.scale_segmentation import validate_extrinsic
from spade_calib.calibration.virtual_ring import virtual_rings,angular_grid,grid_stats
from spade_calib.calibration.uniform_range_grid import spherical_grid,grid_edges
from spade_calib.calibration.surface_visibility import visibility_depths,filter_occluded


def stats(uv,cam,shape):
    xy=np.unique(np.floor(uv).astype(int),axis=0)
    dist=cKDTree(cam).query(xy+.5)[0] if len(cam) else np.full(len(xy),np.inf)
    n=len(xy)
    return dict(pixels=n,coverage=float(n/np.prod(shape)),
                within3=int((dist<=3).sum()),within5=int((dist<=5).sum()),
                within10=int((dist<=10).sum()),
                fraction5=float((dist<=5).sum()/n) if n else None,
                median_camera_distance_px=float(np.median(dist)) if n and len(cam) else None)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=Path("outputs/segment07_trial1"))
    p.add_argument("--cloud-data",type=Path,default=Path("outputs/lines07_M0_ghost"))
    p.add_argument("--ep0",type=Path,default=Path("outputs/edgepoints07_EP0"))
    p.add_argument("--ring-calib",type=Path,required=True,help="HDL-64E S2 factory XML used for synthetic16")
    p.add_argument("--ring-model",choices=["offset","angle"],default="offset")
    p.add_argument("--ring-ranks",nargs="+",type=int,help="Optional allowed ascending vertical ranks, 0=lowest; reject after full 64-way classification")
    p.add_argument("--ring-error",type=float,default=.20)
    p.add_argument("--ring-margin",type=float,default=.04)
    p.add_argument("--lidar-dir",type=Path,help="Override original synthetic16 scan directory")
    p.add_argument("--glim",type=Path,help="Override original GLIM file")
    p.add_argument("--times",type=Path)
    p.add_argument("--frames",nargs="+",type=int)
    p.add_argument("--az-gap",type=float,default=.8)
    p.add_argument("--bin-deg",type=float,default=.2)
    p.add_argument("--el-step",type=float,default=.2)
    p.add_argument("--splat-deg",type=float,default=.3)
    p.add_argument("--patch-radius",type=int,default=1)
    p.add_argument("--surface-max-edge",type=float,default=.5)
    p.add_argument("--surface-min-cos",type=float,default=.15)
    p.add_argument("--occlusion-margin",type=float,default=.5)
    p.add_argument("--save-mode",choices=["minimal","review","full"],default="review")
    p.add_argument("--el-min",type=float,default=-26.)
    p.add_argument("--el-max",type=float,default=4.)
    p.add_argument("--el-gap",type=float,default=.8)
    p.add_argument("--virtual-error",type=float,default=.20)
    p.add_argument("--virtual-margin",type=float,default=.04)
    p.add_argument("--visibility-margin",type=float,default=.5)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    vals=[a.ring_error,a.az_gap,a.virtual_error,a.ring_margin,a.visibility_margin,a.virtual_margin,a.bin_deg]
    if not np.isfinite(vals).all() or min(vals[:3])<=0 or min(vals[3:6])<0 or not .01<=a.bin_deg<=10 or not np.isfinite([a.el_min,a.el_max,a.el_gap,a.el_step,a.splat_deg,a.surface_max_edge,a.surface_min_cos,a.occlusion_margin]).all() or not -90<=a.el_min<a.el_max<=90 or a.el_gap<=0 or not .05<=a.bin_deg<=5 or not .05<=a.el_step<=5 or not 0<a.splat_deg<=2 or not 1<=a.patch_radius<=3 or a.surface_max_edge<=0 or not 0<a.surface_min_cos<=1 or a.occlusion_margin<0:
        p.error("Invalid settings")
    if a.out.exists() and (not a.out.is_dir() or any(a.out.iterdir())):
        raise FileExistsError("Use fresh output")
    cfg=read(a.run/"config.json");m=read(a.cloud_data/"manifest.json");ep=read(a.ep0/"summary.json")
    if (Path(m["run"]).name!=a.run.name or ep["sources"]["manifest_sha256"]!=sha(a.cloud_data/"manifest.json")
        or ep["sources"]["run_sha256"]!=sha(a.run/"summary.json")):
        raise ValueError("EP0, M0 and run provenance mismatch")
    K=np.asarray(m["K"],float)
    if not np.allclose(K,ep["K"]):raise ValueError("EP0 intrinsics mismatch")
    poses={name:validate_extrinsic(np.asarray(T,float)) for name,T in ep["transforms"].items()}
    if "B4c" not in poses:raise ValueError("Missing B4c")
    if ep["settings"]["non_ground"]:
        raise ValueError("Use EP0 all-ground-included baseline for this pilot")
    max_depth=float(ep["settings"]["max_depth_m"])
    jump_abs=float(ep["settings"]["jump_abs_m"]);jump_rel=float(ep["settings"]["jump_rel"])
    same_abs=float(ep["settings"]["same_abs_m"]);same_rel=float(ep["settings"]["same_rel"])
    max_range=float(m["max_range_m"])
    beams=load_beams(a.ring_calib)
    if a.ring_ranks is not None and (len(set(a.ring_ranks))!=16 or len(a.ring_ranks)!=16 or min(a.ring_ranks)<0 or max(a.ring_ranks)>=64):
        p.error("Explicit synthetic16 list requires 16 unique ranks in 0..63")
    lidar=a.lidar_dir or Path(cfg["lidar_dir"])
    glim=a.glim or Path(cfg["glim"])
    times=a.times or Path(cfg["sequence"])/"times.txt"
    stamps=read_kitti_times_ns(times)
    traj=GlimTrajectory.from_array(np.loadtxt(glim,ndmin=2))
    R=Rotation.from_quat(traj.quaternions_xyzw).as_matrix()
    first=int(cfg["glim_first_frame"])
    scans={}
    for path in lidar.iterdir():
        if path.is_file() and path.suffix.lower()==".bin" and path.stem.isdigit():
            f=int(path.stem)
            if f in scans:raise ValueError("Duplicate scan ID")
            scans[f]=path
    records={int(r["frame"]):r for r in m["frames"]}
    frames=sorted(set(a.frames or map(int,ep["frames"])))
    if not frames or any(f not in records or str(f) not in ep["frames"] for f in frames):
        raise ValueError("Missing M0 or EP0 frame")
    for f in frames:
        source=records[f]["source_frames"]
        if f not in source or any(j not in scans or j<first or j-first>=len(R) or j>=len(stamps) for j in source):
            raise ValueError(f"Missing raw scan or pose in frame {f}")
        base=a.ep0/f"frame_{f:06d}"
        for path in [base/"original.png",base/"camera_jump.npz",
                     *[base/f"{n}_{kind}.npz" for n in poses for kind in ("sparse_depth","edges")]]:
            if not path.is_file():raise FileNotFoundError(path)
    ghost=FreeSpaceSettings(**m["ghost_filter"]["settings"]) if m["ghost_filter"]["enabled"] else None
    a.out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    summary=dict(warning="Ring recovery is approximate; XML/decoder consistency must be checked. "
                 "No line fit, camera-guided feature selection, matching or optimization. "
                 "Splat/triangle depths only reject original returns; never inserted into extraction grid. "
                 "Anchor returns protected, unknown rays retained. Local surface hypotheses are approximate; may bridge true gaps or miss grazing surfaces. "
                 "Same EP0 camera-side reference for every arm. DA2 proximity is diagnostic, not correspondence precision.",
                 sources=dict(run=str(a.run),ep0=str(a.ep0),cloud_data=str(a.cloud_data),
                              lidar_dir=str(lidar),glim=str(glim),times=str(times),
                              xml=str(a.ring_calib),xml_sha256=sha(a.ring_calib),
                              glim_sha256=sha(glim),ep0_summary_sha256=sha(a.ep0/"summary.json"),
                              manifest_sha256=sha(a.cloud_data/"manifest.json")),
                 beams_columns=["laser_id","vertical_deg","vertical_offset_m","horizontal_offset_m"],
                 beams_ascending=beams.tolist(),
                 settings=dict(ring_model=a.ring_model,ring_error_deg=a.ring_error,ring_margin_deg=a.ring_margin,
                               allowed_ranks=a.ring_ranks,az_gap_deg=a.az_gap,el_gap_deg=a.el_gap,
                               el_step_deg=a.el_step,el_min_deg=a.el_min,el_max_deg=a.el_max,
                               splat_deg=a.splat_deg,patch_radius=a.patch_radius,surface_max_edge_m=a.surface_max_edge,
                               surface_min_cos=a.surface_min_cos,occlusion_margin_m=a.occlusion_margin,save_mode=a.save_mode,
                               protect_target_returns=True,
                               virtual_error_deg=a.virtual_error,virtual_margin_deg=a.virtual_margin,
                               bin_deg_requested=a.bin_deg,bin_deg_effective=360/np.ceil(360/a.bin_deg),visibility_margin_m=a.visibility_margin,
                               jump_abs_m=jump_abs,jump_rel=jump_rel,same_abs_m=same_abs,same_rel=same_rel,
                               max_depth_m=max_depth,max_source_range_m=max_range,ghost_filter=m["ghost_filter"]),
                 transforms={k:v.tolist() for k,v in poses.items()},K=K.tolist(),frames={})
    for f in frames:
        source=list(map(int,records[f]["source_frames"]));target=source.index(f)
        base=a.ep0/f"frame_{f:06d}";folder=a.out/f"frame_{f:06d}";folder.mkdir()
        im=image(base/"original.png");shape=im.shape[:2]
        with np.load(base/"camera_jump.npz",allow_pickle=False) as z:cam=np.asarray(z["uv"],float)
        if cam.ndim!=2 or cam.shape[1]!=2 or not np.isfinite(cam).all():
            raise ValueError("Invalid EP0 camera points")
        # Reuse exact existing rigid accumulation and free-space filter.
        cloud=accumulate_kitti_rigid_window(lidar,glim,times,source,
                  target_frame_id=f,glim_first_frame=first,max_range_m=max_range)
        keep=target_free_space_mask(cloud,target_scan_id=target,settings=ghost) if ghost else np.ones(len(cloud.xyz),bool)
        origins=(traj.positions[np.asarray(source)-first]-traj.positions[f-first])@R[f-first]
        chunks=[];far_chunks=[];back_chunks=[];local=[];ranks=[];lasers=[];errs=[];margins=[]
        diagnostics=[]
        source_ring=np.full(len(cloud.xyz),-1,int)
        for j,s in enumerate(source):
            raw=load_kitti_bin_scan(scans[s],scan_id=j,timestamp_ns=int(stamps[s]),max_range_m=max_range)
            idx=np.flatnonzero(cloud.scan_id==j)
            if not np.array_equal(raw.source_return_index,cloud.source_return_index[idx]):
                raise ValueError("Raw/accumulated identity mismatch")
            ring=recover_rings(raw.xyz,beams,model=a.ring_model,max_error_deg=a.ring_error,
                               margin_deg=a.ring_margin,allowed_ranks=a.ring_ranks)
            source_ring[idx]=ring["ring"]
            triples=ring_edges(raw.xyz,ring["ring"],az_gap_deg=a.az_gap,
                               jump_abs=jump_abs,jump_rel=jump_rel,same_abs=same_abs,same_rel=same_rel)
            n0=len(triples)
            # Target free-space conflicts: reject near candidate only; far/back
            # are source-view evidence, not claims of target-view visibility.
            triples=triples[keep[idx[triples[:,0]]]]
            near,far,back=triples.T
            chunks.append(idx[near]);far_chunks.append(idx[far]);back_chunks.append(idx[back])
            local.append(np.full(len(near),j,int));ranks.append(ring["ring"][near])
            lasers.append(ring["laser_id"][near]);errs.append(ring["error_deg"][near]);margins.append(ring["margin_deg"][near])
            counts=np.bincount(ring["ring"][ring["ring"]>=0],minlength=64)
            diagnostics.append(dict(frame=s,raw_points=len(raw.xyz),assigned=int(counts.sum()),
                 assigned_fraction=float(counts.sum()/max(len(raw.xyz),1)),
                 assigned_rank_counts=counts.tolist(),
                 best_rank_counts=np.bincount(ring["best_rank"],minlength=64).tolist(),
                 occupied_ranks=int((counts>0).sum()),
                 error_deg_p10_p50_p90=percentiles(ring["error_deg"]),
                 margin_deg_p10_p50_p90=percentiles(ring["margin_deg"]),
                 edge_before_ghost=n0,edge_after_ghost=len(near),scan_sha256=sha(scans[s])))
        ni,fi,bi=map(np.concatenate,(chunks,far_chunks,back_chunks))
        # Keep all original return identities, including the source-ring metadata.
        # No voxel downsampling or fabricated points before angular sampling.
        xyz=np.asarray(cloud.xyz,float)
        sid=np.asarray(cloud.scan_id,int)
        target_mask=sid==target
        eligible=keep&(np.linalg.norm(xyz,axis=1)>=.5)&(np.linalg.norm(xyz,axis=1)<=max_range)
        use=np.flatnonzero(eligible)
        kw=dict(jump_abs=jump_abs,jump_rel=jump_rel,same_abs=same_abs,same_rel=same_rel)
        gkw=dict(az_deg=a.bin_deg,el_deg=a.el_step,el_min=a.el_min,el_max=a.el_max)
        g,meta=spherical_grid(xyz,eligible,**gkw)
        ds,dt=visibility_depths(xyz,g,eligible,el_min=a.el_min,el_max=a.el_max,
                   splat_deg=a.splat_deg,patch_radius=a.patch_radius,
                   max_edge_m=a.surface_max_edge,min_cos=a.surface_min_cos)
        masks={"Raw":eligible}
        visibility={}
        for arm,d in (("Splat",ds),("Surface",dt)):
            masks[arm],visibility[arm]=filter_occluded(xyz,eligible,target_mask,d,margin_m=a.occlusion_margin)
        # Separate conservative surface from splat rather than combine filters.
        # Keep the same full-cloud camera-side visibility reference across arms.
        triples={"MV1_target":np.column_stack((ni,fi,bi))[sid[ni]==target],
                 "MV1_all":np.column_stack((ni,fi,bi))}
        grids={};grid_info={}
        for arm,mask in masks.items():
            grid=g if arm=="Raw" else spherical_grid(xyz,mask,**gkw)[0]
            grids[arm]=grid
            h,v=grid_edges(xyz,grid,az_gap_deg=a.az_gap,el_gap_deg=a.el_gap,**kw)
            triples[arm+"_H"]=h;triples[arm+"_V"]=v
            triples[arm+"_HV"]=np.concatenate((h,v))
            gi=grid_stats(grid,g,np.arange(g.shape[0]),gap_deg=a.az_gap)
            gi.update(horizontal=len(h),vertical=len(v),unique_hv=len(np.unique(triples[arm+"_HV"][:,0])),
                      raw_grid_bins_lost=int(((g>=0)&(grid<0)).sum()),
                      raw_grid_bins_replaced=int(((g>=0)&(grid>=0)&(grid!=g)).sum()))
            grid_info[arm]=gi
            if a.save_mode=="full":
                selected=grid[grid>=0]
                ranges=np.full(grid.shape,np.nan,np.float32);ranges[grid>=0]=np.linalg.norm(xyz[selected],axis=1)
                np.savez_compressed(folder/f"{arm}_grid.npz",index_map=grid,range_m=ranges)
        # Fixed raw-edge evidence diagnostic: a raw triple invalidates if ANY
        # of its near/far/back observations was removed by the visibility filter.
        # Distinguish points having no surviving hypothesis from points losing
        # one hypothesis but retaining another.
        invalid={}
        for arm in ("Splat","Surface"):
            invalid[arm]={}
            for direction in ("H","V","HV"):
                e=triples["Raw_"+direction];bad=np.any(~masks[arm][e],axis=1)
                surviving=np.unique(e[~bad,0]);lost=np.setdiff1d(np.unique(e[:,0]),surviving)
                invalid[arm][direction]=dict(raw_triples=len(e),invalid_triples=int(bad.sum()),
                         near_removed=int((~masks[arm][e[:,0]]).sum()),
                         far_removed=int((~masks[arm][e[:,1]]).sum()),
                         back_removed=int((~masks[arm][e[:,2]]).sum()),
                         raw_near_all_hypotheses_invalid=len(lost))
                if direction=="HV":
                    triples[arm+"_fixed_surviving_HV"]=e[~bad]
                    # Subset of original triples, projected independently for diagnostic.
                    triples[arm+"_fixed_lost_HV"]=e[np.isin(e[:,0],lost)]
        all_idx=np.unique(np.concatenate([e.ravel() for e in triples.values()]))
        arms={}
        for arm,e in triples.items():
            arms[arm]=np.unique(e[:,0])
            if a.save_mode!="minimal" and not arm.endswith("_HV"):
                np.savez_compressed(folder/f"{arm}_edge_triples.npz",feature_indices=np.searchsorted(all_idx,e))
        for arm in ("Splat","Surface"):
            arms[arm+"_new_HV"]=np.setdiff1d(arms[arm+"_HV"],arms["Raw_HV"])
            arms[arm+"_lost_HV"]=np.setdiff1d(arms["Raw_HV"],arms[arm+"_HV"])
        if a.save_mode!="minimal":
            np.savez_compressed(folder/"source_features.npz",xyz_lidar=xyz[all_idx],
                 accumulated_index=all_idx,scan_ids=sid[all_idx],scan_frames=np.asarray(source)[sid[all_idx]],
                 source_return_index=cloud.source_return_index[all_idx],source_ring=source_ring[all_idx],
                 sensor_origins=origins,source_frames=np.asarray(source),
                 splat_depth_m=ds[all_idx],surface_depth_m=dt[all_idx],
                 splat_retained=masks["Splat"][all_idx],surface_retained=masks["Surface"][all_idx])
            for arm,ids in arms.items():
                np.savez_compressed(folder/f"{arm}_source_ids.npz",feature_indices=np.searchsorted(all_idx,ids))
        if a.save_mode=="full":
            np.savez_compressed(folder/"visibility_full.npz",xyz_lidar=xyz.astype(np.float32),
                    scan_ids=sid,source_return_index=cloud.source_return_index,eligible=eligible,
                    splat_depth_m=ds.astype(np.float32),surface_depth_m=dt.astype(np.float32),
                    splat_retained=masks["Splat"],surface_retained=masks["Surface"])
        info=dict(source_frames=source,scan_diagnostics=diagnostics,grid=grid_info,
                  raw_grid_meta=meta,visibility=visibility,raw_edge_invalidation=invalid,
                  source_candidates={k:len(v) for k,v in arms.items()},poses={})
        panels=[panel(im,"Original RGB"),panel(overlay(im.copy(),cam,(0,255,0)),"DA2 jump (reference only)")]
        for name,T in poses.items():
            with np.load(base/f"{name}_sparse_depth.npz",allow_pickle=False) as z:depth=z["depth_m"]
            with np.load(base/f"{name}_edges.npz",allow_pickle=False) as z:uv0=z["uv"]
            result={"MV0_EP0":stats(uv0,cam,shape)}
            for arm,ids in arms.items():
                q=projected_candidates(xyz[ids],T,K,shape,depth,max_depth=max_depth,margin=a.visibility_margin)
                sel=q["visible"];ii=ids[q["candidate_indices"]]
                uv=q["uv"][sel];r=stats(uv,cam,shape)
                r.update(preprojection_candidates=len(ids),projected_pixels=len(ii),
                         rejected_occluded=int((~sel).sum()),visible_unknown=int((sel&~q["reference_known"]).sum()))
                result[arm]=r
                q["candidate_indices"]=np.searchsorted(all_idx,ii)
                if a.save_mode!="minimal":np.savez_compressed(folder/f"{name}_{arm}.npz",**q)
                if a.save_mode!="minimal" and arm in ("MV1_all","Raw_HV","Splat_HV","Surface_HV","Surface_fixed_lost_HV"):
                    vis=overlay(overlay(im.copy(),cam,(0,255,0)),uv,(0,165,255))
                    panels.append(panel(vis,f"{name} {arm} | {r['pixels']} pixels"))
                if a.save_mode=="full":save(folder/f"{name}_{arm}_overlay.png",overlay(im.copy(),uv,(0,165,255)))
            info["poses"][name]=result
        if a.save_mode!="minimal":
            if not cv2.imwrite(str(folder/"comparison.jpg"),np.vstack(panels),[cv2.IMWRITE_JPEG_QUALITY,85]):
                raise OSError("Could not save comparison image")
        write(folder/"summary.json",info);summary["frames"][str(f)]=info
        print(f"{f:06d}: {info['source_candidates']}",flush=True)
    summary["aggregate"]={}
    for name in poses:
        summary["aggregate"][name]={}
        for arm in ["MV0_EP0",*arms]:
            rows=[v["poses"][name][arm] for v in summary["frames"].values()]
            n=sum(r["pixels"] for r in rows);r=dict(pixels=n)
            for d in (3,5,10):
                c=sum(x[f"within{d}"] for x in rows)
                r[f"within{d}"]=c;r[f"fraction{d}"]=c/n if n else None
            summary["aggregate"][name][arm]=r
    write(a.out/"summary.json",summary)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
