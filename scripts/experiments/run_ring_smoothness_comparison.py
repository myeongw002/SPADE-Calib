#!/usr/bin/env python3
"""MV3: compare source-ring jump and LOAM-style smoothness, no matching."""
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
from spade_calib.calibration.ring_smoothness import ring_smoothness,select_top


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
    p.add_argument("--half-window",type=int,default=5)
    p.add_argument("--occlusion-abs",type=float,default=.3)
    p.add_argument("--occlusion-rel",type=float,default=.03)
    p.add_argument("--visibility-margin",type=float,default=.5)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    vals=[a.ring_error,a.az_gap,a.occlusion_abs,a.occlusion_rel,a.ring_margin,a.visibility_margin]
    if not np.isfinite(vals).all() or min(vals[:4])<=0 or min(vals[4:])<0 or a.half_window<1:
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
        for path in [base/"original.png",base/"camera_jump.npz",base/"camera_crease.npz",
                     *[base/f"{n}_{kind}.npz" for n in poses for kind in ("sparse_depth","edges")]]:
            if not path.is_file():raise FileNotFoundError(path)
    ghost=FreeSpaceSettings(**m["ghost_filter"]["settings"]) if m["ghost_filter"]["enabled"] else None
    a.out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    summary=dict(warning="Ring recovery is approximate; XML/decoder consistency must be checked. "
                 "No line fit, camera-guided feature selection, matching or optimization. "
                 "LOAM-inspired normalized score, not full LOAM. Clean rejects strong radial jumps, not a proven crease detector. "
                 "Per-scan preprojection budgets matched to unique jump returns where enough eligible smoothness points exist. "
                 "Union has a larger budget. No NMS, angular-resampling or recurrence gate. "
                 "DA2 jump/crease are proxies, not GT labels. Same-pixel visibility is approximate.",
                 sources=dict(run=str(a.run),ep0=str(a.ep0),cloud_data=str(a.cloud_data),
                              lidar_dir=str(lidar),glim=str(glim),times=str(times),
                              xml=str(a.ring_calib),xml_sha256=sha(a.ring_calib),
                              glim_sha256=sha(glim),ep0_summary_sha256=sha(a.ep0/"summary.json"),
                              manifest_sha256=sha(a.cloud_data/"manifest.json")),
                 beams_columns=["laser_id","vertical_deg","vertical_offset_m","horizontal_offset_m"],
                 beams_ascending=beams.tolist(),
                 settings=dict(ring_model=a.ring_model,ring_error_deg=a.ring_error,ring_margin_deg=a.ring_margin,
                               allowed_ranks=a.ring_ranks,az_gap_deg=a.az_gap,half_window=a.half_window,
                               occlusion_abs_m=a.occlusion_abs,occlusion_rel=a.occlusion_rel,
                               budget_rule="unique post-ghost jump returns per source scan; raw and clean top-K independently",
                               visibility_margin_m=a.visibility_margin,
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
        with np.load(base/"camera_crease.npz",allow_pickle=False) as z:crease=np.asarray(z["uv"],float)
        if crease.ndim!=2 or crease.shape[1]!=2 or not np.isfinite(crease).all():
            raise ValueError("Invalid EP0 crease points")
        camera={"jump":cam,"crease":crease,"union":np.unique(np.vstack((cam,crease)),axis=0)}
        # Reuse exact existing rigid accumulation and free-space filter.
        cloud=accumulate_kitti_rigid_window(lidar,glim,times,source,
                  target_frame_id=f,glim_first_frame=first,max_range_m=max_range)
        keep=target_free_space_mask(cloud,target_scan_id=target,settings=ghost) if ghost else np.ones(len(cloud.xyz),bool)
        origins=(traj.positions[np.asarray(source)-first]-traj.positions[f-first])@R[f-first]
        chunks=[];far_chunks=[];back_chunks=[];local=[];ranks=[];lasers=[];errs=[];margins=[]
        raw_sel=[];clean_sel=[];smooth_ids=[];smooth_scores=[];smooth_clean=[];smooth_ranks=[]
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
            budget=len(np.unique(near))
            score,clean=ring_smoothness(raw.xyz,ring["ring"],half_window=a.half_window,
                       az_gap_deg=a.az_gap,occlusion_abs=a.occlusion_abs,occlusion_rel=a.occlusion_rel)
            eligible=keep[idx]&np.isfinite(score)
            sr=select_top(score,eligible,budget)
            sc=select_top(score,eligible&clean,budget)
            su=np.union1d(sr,sc)
            raw_sel.append(idx[sr]);clean_sel.append(idx[sc])
            smooth_ids.append(idx[su]);smooth_scores.append(score[su])
            smooth_clean.append(clean[su]);smooth_ranks.append(ring["ring"][su])
            counts=np.bincount(ring["ring"][ring["ring"]>=0],minlength=64)
            diagnostics.append(dict(frame=s,raw_points=len(raw.xyz),assigned=int(counts.sum()),
                 assigned_fraction=float(counts.sum()/max(len(raw.xyz),1)),
                 assigned_rank_counts=counts.tolist(),
                 best_rank_counts=np.bincount(ring["best_rank"],minlength=64).tolist(),
                 occupied_ranks=int((counts>0).sum()),
                 error_deg_p10_p50_p90=percentiles(ring["error_deg"]),
                 margin_deg_p10_p50_p90=percentiles(ring["margin_deg"]),
                 edge_before_ghost=n0,edge_after_ghost=len(near),jump_budget=budget,
                 smooth_eligible=int(eligible.sum()),smooth_clean_eligible=int((eligible&clean).sum()),
                 smooth_raw_selected=len(sr),smooth_clean_selected=len(sc),
                 smooth_raw_score_p10_p50_p90=percentiles(score[sr]),
                 smooth_clean_score_p10_p50_p90=percentiles(score[sc]),
                 raw_clean_overlap=len(np.intersect1d(sr,sc)),
                 jump_raw_overlap=len(np.intersect1d(near,sr)),
                 jump_clean_overlap=len(np.intersect1d(near,sc)),
                 scan_sha256=sha(scans[s])))
        ni,fi,bi=map(np.concatenate,(chunks,far_chunks,back_chunks))
        sid=np.concatenate(local)
        # The old jump hypotheses and their source-view far/back evidence remain intact.
        np.savez_compressed(folder/"source_jump_edges.npz",xyz_lidar=cloud.xyz[ni],
            background_xyz_lidar=cloud.xyz[fi],back_xyz_lidar=cloud.xyz[bi],
            source_return_index=cloud.source_return_index[ni],
            background_return_index=cloud.source_return_index[fi],
            back_return_index=cloud.source_return_index[bi],
            scan_ids=sid,scan_frames=np.asarray(source)[sid],ring_rank=np.concatenate(ranks),
            laser_id=np.concatenate(lasers),ring_error_deg=np.concatenate(errs),
            ring_margin_deg=np.concatenate(margins),source_frames=np.asarray(source),
            sensor_origins=origins)
        raw_idx=np.concatenate(raw_sel);clean_idx=np.concatenate(clean_sel)
        smooth_idx=np.concatenate(smooth_ids)
        all_idx=np.union1d(ni,smooth_idx)
        xyz=cloud.xyz[all_idx].astype(float)
        sid=cloud.scan_id[all_idx]
        rank=np.full(len(all_idx),-1,int)
        rank[np.searchsorted(all_idx,ni)]=np.concatenate(ranks)
        si=np.searchsorted(all_idx,smooth_idx)
        rank[si]=np.concatenate(smooth_ranks)
        score=np.full(len(all_idx),np.nan)
        score[si]=np.concatenate(smooth_scores)
        clean=np.zeros(len(all_idx),bool);clean[si]=np.concatenate(smooth_clean)
        jump_mask=np.isin(all_idx,ni)
        raw_mask=np.isin(all_idx,raw_idx)
        clean_mask=np.isin(all_idx,clean_idx)
        np.savez_compressed(folder/"source_features.npz",xyz_lidar=xyz,
            source_return_index=cloud.source_return_index[all_idx],
            scan_ids=sid,scan_frames=np.asarray(source)[sid],source_frames=np.asarray(source),
            sensor_origins=origins,ring_rank=rank,loam_score=score,
            smoothness_window_clean=clean,is_jump=jump_mask,
            is_smooth_raw=raw_mask,is_smooth_clean=clean_mask)
        arms={"Jump_target":jump_mask&(sid==target),
              "Jump_all":jump_mask,
              "Smooth_raw":raw_mask,
              "Smooth_clean":clean_mask,
              "Jump_plus_Smooth_clean":jump_mask|clean_mask}
        info=dict(source_frames=source,scan_diagnostics=diagnostics,
                  source_candidates={k:int(v.sum()) for k,v in arms.items()},
                  source_overlap=dict(jump_raw=int((jump_mask&raw_mask).sum()),
                      jump_clean=int((jump_mask&clean_mask).sum()),
                      raw_clean=int((raw_mask&clean_mask).sum())),poses={})
        panels=[panel(im,"Original RGB"),
                panel(overlay(im.copy(),cam,(0,255,0)),"DA2 jump"),
                panel(overlay(im.copy(),crease,(255,0,255)),"DA2 crease proxy")]
        for name,T in poses.items():
            with np.load(base/f"{name}_sparse_depth.npz",allow_pickle=False) as z:depth=z["depth_m"]
            with np.load(base/f"{name}_edges.npz",allow_pickle=False) as z:uv0=z["uv"]
            result={"MV0_EP0":{k:stats(uv0,v,shape) for k,v in camera.items()}}
            for arm,mask in arms.items():
                ids=np.flatnonzero(mask)
                q=projected_candidates(xyz[ids],T,K,shape,depth,max_depth=max_depth,margin=a.visibility_margin)
                sel=q["visible"];ii=ids[q["candidate_indices"]]
                uv=q["uv"][sel]
                r={k:stats(uv,v,shape) for k,v in camera.items()}
                r.update(preprojection_candidates=len(ids),projected_pixels=len(ii),
                         rejected_occluded=int((~sel).sum()),visible_unknown=int((sel&~q["reference_known"]).sum()))
                result[arm]=r
                q["candidate_indices"]=ii
                np.savez_compressed(folder/f"{name}_{arm}.npz",**q)
                vis=overlay(im.copy(),uv,(0,165,255))
                both=overlay(overlay(overlay(im.copy(),cam,(0,255,0)),crease,(255,0,255)),uv,(0,165,255))
                save(folder/f"{name}_{arm}_overlay.png",vis)
                save(folder/f"{name}_{arm}_camera_overlay.png",both)
                panels.append(panel(vis,f"{name} {arm} | {r['jump']['pixels']} pixels"))
            info["poses"][name]=result
        save(folder/"comparison.png",np.vstack(panels))
        write(folder/"summary.json",info);summary["frames"][str(f)]=info
        print(f"{f:06d}: {info['source_candidates']}",flush=True)
    summary["aggregate"]={}
    for name in poses:
        summary["aggregate"][name]={}
        for arm in ["MV0_EP0",*arms]:
            result={}
            for kind in camera:
                rows=[v["poses"][name][arm][kind] for v in summary["frames"].values()]
                n=sum(r["pixels"] for r in rows)
                r=dict(pixels=n)
                for d in (3,5,10):
                    c=sum(x[f"within{d}"] for x in rows)
                    r[f"within{d}"]=c;r[f"fraction{d}"]=c/n if n else None
                result[kind]=r
            summary["aggregate"][name][arm]=result
    write(a.out/"summary.json",summary)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
