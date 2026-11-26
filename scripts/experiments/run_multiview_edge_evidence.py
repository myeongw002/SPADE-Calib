#!/usr/bin/env python3
"""MV1/MV2 source-ring edge accumulation; compare existing EP0 (MV0)."""
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
    p.add_argument("--support-radius",type=float,default=.25)
    p.add_argument("--min-scans",type=int,nargs="+",default=[2,3,5])
    p.add_argument("--visibility-margin",type=float,default=.5)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    vals=[a.ring_error,a.az_gap,a.support_radius,a.ring_margin,a.visibility_margin]
    if not np.isfinite(vals).all() or min(vals[:3])<=0 or min(vals[3:])<0 or min(a.min_scans)<2:
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
                 "Repeat support is a spatial proxy, not identity; moving silhouettes may be lost. "
                 "MV0 uses voxel returns, MV1/2 raw returns before voxel (not a controlled equal-density comparison). "
                 "Camera proximity is diagnostic only; DA2 is not GT. Same-pixel visibility remains approximate.",
                 sources=dict(run=str(a.run),ep0=str(a.ep0),cloud_data=str(a.cloud_data),
                              lidar_dir=str(lidar),glim=str(glim),times=str(times),
                              xml=str(a.ring_calib),xml_sha256=sha(a.ring_calib),
                              glim_sha256=sha(glim),ep0_summary_sha256=sha(a.ep0/"summary.json"),
                              manifest_sha256=sha(a.cloud_data/"manifest.json")),
                 beams_columns=["laser_id","vertical_deg","vertical_offset_m","horizontal_offset_m"],
                 beams_ascending=beams.tolist(),
                 settings=dict(ring_model=a.ring_model,ring_error_deg=a.ring_error,ring_margin_deg=a.ring_margin,
                               allowed_ranks=a.ring_ranks,az_gap_deg=a.az_gap,support_radius_m=a.support_radius,
                               min_scans=sorted(set(a.min_scans)),visibility_margin_m=a.visibility_margin,
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
        for j,s in enumerate(source):
            raw=load_kitti_bin_scan(scans[s],scan_id=j,timestamp_ns=int(stamps[s]),max_range_m=max_range)
            idx=np.flatnonzero(cloud.scan_id==j)
            if not np.array_equal(raw.source_return_index,cloud.source_return_index[idx]):
                raise ValueError("Raw/accumulated identity mismatch")
            ring=recover_rings(raw.xyz,beams,model=a.ring_model,max_error_deg=a.ring_error,
                               margin_deg=a.ring_margin,allowed_ranks=a.ring_ranks)
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
        sid=np.concatenate(local);xyz=cloud.xyz[ni].astype(float)
        support=recurrence(xyz,sid,origins,radius=a.support_radius)
        count=support["scan_count"]
        np.savez_compressed(folder/"source_edges.npz",xyz_lidar=xyz,
            background_xyz_lidar=cloud.xyz[fi],back_xyz_lidar=cloud.xyz[bi],
            source_return_index=cloud.source_return_index[ni],
            background_return_index=cloud.source_return_index[fi],
            back_return_index=cloud.source_return_index[bi],
            scan_frames=np.asarray(source)[sid],scan_ids=sid,source_frames=np.asarray(source),
            sensor_origins=origins,ring_rank=np.concatenate(ranks),laser_id=np.concatenate(lasers),
            ring_error_deg=np.concatenate(errs),ring_margin_deg=np.concatenate(margins),**support)
        info=dict(source_frames=source,raw_edge_hypotheses=len(xyz),
                  unique_source_returns=len(np.unique(np.column_stack((sid,cloud.source_return_index[ni])),axis=0)),
                  scan_diagnostics=diagnostics,support_histogram=np.bincount(count,minlength=len(source)+1).tolist(),
                  view_angle_deg_p10_p50_p90=percentiles(support["view_angle_deg"]),poses={})
        panels=[panel(im,"Original RGB"),panel(overlay(im.copy(),cam,(0,255,0)),"DA2 jump (reference only)")]
        arms={"MV1_target":sid==target,"MV1_all":np.ones(len(xyz),bool)}
        arms.update({f"MV2_n{n}":count>=n for n in sorted(set(a.min_scans))})
        for name,T in poses.items():
            with np.load(base/f"{name}_sparse_depth.npz",allow_pickle=False) as z:depth=z["depth_m"]
            with np.load(base/f"{name}_edges.npz",allow_pickle=False) as z:uv0=z["uv"]
            result={"MV0_EP0":stats(uv0,cam,shape)}
            panels.append(panel(overlay(im.copy(),uv0,(0,165,255)),f"{name} MV0 EP0"))
            for arm,mask in arms.items():
                ids=np.flatnonzero(mask)
                q=projected_candidates(xyz[ids],T,K,shape,depth,max_depth=max_depth,margin=a.visibility_margin)
                sel=q["visible"];ii=ids[q["candidate_indices"]]
                r=stats(q["uv"][sel],cam,shape)
                r.update(preprojection_candidates=len(ids),projected_pixels=len(ii),
                         rejected_occluded=int((~sel).sum()),visible_unknown=int((sel&~q["reference_known"]).sum()))
                result[arm]=r
                q["candidate_indices"]=ii
                np.savez_compressed(folder/f"{name}_{arm}.npz",**q)
                vis=overlay(im.copy(),q["uv"][sel],(0,165,255))
                both=overlay(overlay(im.copy(),cam,(0,255,0)),q["uv"][sel],(0,165,255))
                save(folder/f"{name}_{arm}_overlay.png",vis)
                save(folder/f"{name}_{arm}_camera_overlay.png",both)
                panels.append(panel(vis,f"{name} {arm} | {r['pixels']} pixels"))
            info["poses"][name]=result
        save(folder/"comparison.png",np.vstack(panels))
        write(folder/"summary.json",info);summary["frames"][str(f)]=info
        assigned=sum(r["assigned"] for r in diagnostics);total=sum(r["raw_points"] for r in diagnostics)
        print(f"{f:06d}: ring assigned {assigned/max(total,1):.1%}, edges {len(xyz)}, "
              f"MV2>=3 {int((count>=3).sum())}",flush=True)
    write(a.out/"summary.json",summary)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
