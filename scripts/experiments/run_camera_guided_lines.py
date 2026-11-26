#!/usr/bin/env python3
"""C0 camera-guided local extraction vs archived global L0 proposals."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict,replace
from pathlib import Path
from time import perf_counter
import csv
import json
import cv2
import numpy as np
from run_lidar_line_extraction import sha,write,save_lines
from run_lidar_line_support import load_archive,read
from spade_calib.calibration.camera_guided_lines import GuideSettings,extract_guided,associate
from spade_calib.calibration.lidar_line_extraction import Settings
from spade_calib.calibration.lidar_line_support import manual_matches
from spade_calib.selection.scale_segmentation import validate_extrinsic


def line_draw(img,uv,color,width=1):
    q=np.rint(uv).astype(int)
    cv2.line(img,tuple(q[0]),tuple(q[1]),color,width,cv2.LINE_AA)


def save_img(p,img):
    if not cv2.imwrite(str(p),img):raise OSError(f"Cannot save {p}")


def panel(img,label):
    out=cv2.copyMakeBorder(img,25,0,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
    cv2.putText(out,label,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--l0",type=Path,required=True)
    p.add_argument("--edge-data",type=Path,help="Reuse archived ELSED arrays where available")
    p.add_argument("--out",type=Path,required=True);p.add_argument("--frames",type=int,nargs="+")
    p.add_argument("--cell",type=float,default=2.);p.add_argument("--roi-width",type=float,default=30.)
    p.add_argument("--min-image-length",type=float,default=40.)
    p.add_argument("--review-per-frame",type=int,default=12);p.add_argument("--seed",type=int,default=17)
    a=p.parse_args(argv)
    guide=replace(GuideSettings(),roi_width_px=a.roi_width,min_image_length_px=a.min_image_length);guide.validate()
    if a.review_per_frame<0:p.error("Invalid review count")
    m=a.data/"manifest.json";ap=a.data/"annotations.json";l0p=a.l0/"summary.json"
    manifest=read(m);ann=read(ap) if ap.exists() else dict(records=[]);l0=read(l0p)
    digest=sha(m);ad=sha(ap) if ap.exists() else None
    if l0["manifest_sha256"]!=digest or l0["annotations_sha256"]!=ad or (ap.exists() and ann["manifest_sha256"]!=digest):
        raise ValueError("M0/L0 provenance mismatch")
    if l0.get("remove_ground"):raise ValueError("C0 requires L0 without ground removal")
    cell=f"cell_{a.cell:g}m";arm=l0["arms"][cell]
    cfg=Settings(**arm["settings"]);cfg.validate()
    edge=read(a.edge_data/"summary.json") if a.edge_data else None
    if edge and (edge["manifest_sha256"]!=digest or edge["annotations_sha256"]!=ad
                 or edge["camera_index"]!=manifest["camera_index"]):
        raise ValueError("ELSED provenance mismatch")
    rows={int(r["frame"]):r for r in manifest["frames"]}
    frames=sorted(set(a.frames or map(int,arm["frames"])))
    if not frames or any(str(f) not in arm["frames"] or f not in rows for f in frames):raise ValueError("Missing frame")
    X=validate_extrinsic(np.asarray(manifest["T_camera_lidar"],float));K=np.asarray(manifest["K"],float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or not np.allclose(K[2],[0,0,1]):raise ValueError("Invalid K")
    # Import before creating output if detection is needed.
    need_detect=any(edge is None or str(f) not in edge["frames"] for f in frames)
    if need_detect:
        import pyelsed
    if a.out.exists() and any(a.out.iterdir()):raise FileExistsError("Use fresh output")
    a.out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    out=dict(manifest_sha256=digest,annotations_sha256=ad,l0_summary_sha256=sha(l0p),
             plane_settings=asdict(cfg),guide_settings=asdict(guide),review_seed=a.seed,
             review_per_frame=a.review_per_frame,edge_summary_sha256=sha(a.edge_data/"summary.json") if edge else None,
             warning="Single-view proposals only; no semantic segmentation, no motion consistency, no occlusion or calibration. Image selects cells; fitting uses all local LiDAR points. Archived L0 uses different RNG sequence, so comparisons not exact cell-level ablations. Multiple fragments per image line are unverified. Same-cloud manual refs not GT.",frames={})
    review=[]
    for f in frames:
        src=a.data/f"frame_{f:06d}";folder=a.out/f"frame_{f:06d}";folder.mkdir()
        cp=src/"cloud.npz";ip=src/"image.png";old=arm["frames"][str(f)]
        if sha(cp)!=rows[f]["cloud_sha256"] or sha(cp)!=old["cloud_sha256"] or sha(ip)!=rows[f]["image_sha256"] or sha(ip)!=old["image_sha256"]:
            raise ValueError("M0 archive changed")
        with np.load(cp,allow_pickle=False) as z:cloud=np.asarray(z["xyz"],float);scans=z["scan_frames"]
        if len(cloud)!=rows[f]["points"] or not np.isfinite(cloud).all():raise ValueError("Invalid cloud")
        img=cv2.imread(str(ip))
        if img is None:raise ValueError("Unreadable image")
        if edge and str(f) in edge["frames"]:
            ef=a.edge_data/f"frame_{f:06d}"
            ei=cv2.imread(str(ef/"original.png"))
            if ei is None or not np.array_equal(ei,img):raise ValueError("ELSED image mismatch")
            ep=ef/"elsed_segments.npz"
            with np.load(ep,allow_pickle=False) as z:s=np.asarray(z["segments"],float);scores=np.asarray(z["scores"])
            if len(s)!=edge["frames"][str(f)]["elsed"]["segments"]:raise ValueError("ELSED count mismatch")
            source=dict(kind="archive",sha256=sha(ep))
        else:
            s,scores=pyelsed.detect(np.ascontiguousarray(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)))
            s=np.asarray(s,float);scores=np.asarray(scores)
            source=dict(kind="new",module=str(pyelsed.__file__))
        s=s.reshape(-1,4);scores=scores.reshape(-1)
        if len(scores)!=len(s) or not np.isfinite(s).all() or not np.isfinite(scores).all():raise ValueError("Invalid ELSED arrays")
        np.savez_compressed(folder/"elsed_segments.npz",segments=s,scores=scores)
        source["saved_sha256"]=sha(folder/"elsed_segments.npz")
        start=perf_counter();guided,stats=extract_guided(cloud,scans,s,X,K,img.shape[:2],cfg,guide)
        elapsed=perf_counter()-start
        save_lines(folder,"guided_lines",guided)
        oldxyz,_,hashes=load_archive(a.l0/cell/f"frame_{f:06d}","raw_lines",len(cloud))
        if len(oldxyz)!=old["raw_lines"]:raise ValueError("L0 raw count mismatch")
        baseline=[dict(xyz=x) for x in oldxyz]
        refs=[r for r in ann["records"] if int(r["frame"])==f and r.get("manual_identity_confirmed")]
        eligible=s[np.linalg.norm(s[:,2:]-s[:,:2],axis=1)>=guide.min_image_length_px]
        camera=img.copy()
        for uv in eligible:line_draw(camera,uv.reshape(2,2),(0,255,0))
        panels=[panel(img,"Original"),panel(camera,"ELSED eligible")]
        info=dict(extraction=stats,extraction_seconds=elapsed,elsed=source,l0_hashes=hashes,arms={})
        for method,lines,require in (("global",baseline,False),("guided",guided,True)):
            matches=associate(lines,s,X,K,img.shape[:2],guide,require_guide=require)
            write(folder/(method+"_matches.json"),matches)
            proposed=[r for r in matches if r["status"]=="proposed"]
            coords=np.asarray([r["xyz_lidar"] for r in proposed],float).reshape(-1,2,3)
            reference=[dict(id=r["id"],**manual_matches(np.asarray(r["xyz_lidar"]),coords)) for r in refs]
            # Translate diagnostic best index back to original line ID.
            for r in reference:
                if r["best"] is not None:r["best"]["line_id"]=proposed[r["best"]["index"]]["id"]
            counts=Counter(r["status"] for r in matches)
            info["arms"][method]=dict(counts=dict(counts),candidates=len(matches),proposals=len(proposed),
                distinct_image_lines=len({r["selected_index"] for r in proposed}),
                shared_image_line_proposals=sum(r["image_line_proposal_count"]>1 for r in proposed),
                manual_3d_overlap=reference)
            auto=img.copy();all_proj=img.copy()
            for r in matches:
                if r["projected_uv"] is not None:line_draw(all_proj,r["projected_uv"],(255,255,0))
            for r in proposed:
                line_draw(auto,r["projected_uv"],(0,0,255))
                line_draw(auto,s[r["selected_index"]].reshape(2,2),(0,255,0))
            panels.extend((panel(all_proj,method+" all projectable"),panel(auto,method+" proposals: red 3D, green 2D")))
            rng=np.random.default_rng(np.random.SeedSequence([a.seed,f,0 if method=="global" else 1]))
            random_ids=set(map(int,rng.choice(len(proposed),min(a.review_per_frame,len(proposed)),replace=False))) if proposed else set()
            ref_ids={r["best"]["index"] for r in reference if r["best"] is not None}
            rd=folder/method;rd.mkdir()
            for j in sorted(random_ids|ref_ids):
                r=proposed[j];i=r["id"];sid=r["selected_index"]
                check=img.copy();line_draw(check,r["projected_uv"],(0,0,255),2);line_draw(check,s[sid].reshape(2,2),(0,255,0),2)
                save_img(rd/f"line_{i:04d}.png",np.vstack((panel(img,"Original"),panel(check,"red 3D / green 2D"))))
                review.append(dict(frame=f,method=method,line_id=i,segment_id=sid,random_sample=j in random_ids,
                                   reference_candidate=j in ref_ids,label="",notes=""))
        save_img(folder/"comparison.png",np.vstack(panels))
        out["frames"][str(f)]=info;write(folder/"summary.json",info);write(a.out/"summary.json",out)
        print(f"{f:06d}: {stats['active_cells']}/{stats['occupied_cells']} active cells; global {info['arms']['global']['proposals']} / guided {info['arms']['guided']['proposals']} proposals",flush=True)
    with (a.out/"review.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["frame","method","line_id","segment_id","random_sample","reference_candidate","label","notes"])
        w.writeheader();w.writerows(review)
    return 0


if __name__=="__main__":raise SystemExit(main())
