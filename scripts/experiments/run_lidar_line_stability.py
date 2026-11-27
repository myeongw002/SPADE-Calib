#!/usr/bin/env python3
"""L2 frozen L0 scan stability, visible candidate review, no selection training."""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
from pathlib import Path
import csv
import json
from time import perf_counter
import cv2
import numpy as np
from run_lidar_line_support import load_archive,read
from run_lidar_line_extraction import sha,write,draw
from spade_calib.calibration.lidar_line_stability import StabilitySettings,evaluate
from spade_calib.calibration.lidar_line_support import manual_matches
from spade_calib.calibration.line_matching import clip_projected_line
from spade_calib.selection.scale_segmentation import validate_extrinsic


def flatten(r):
    result={}
    for k in ("odd_even","temporal"):
        x=r[k];c=x["comparison"] or {}
        result.update({k+"_status":x["status"],k+"_A_status":x["A"]["status"],k+"_B_status":x["B"]["status"],
                       k+"_rmse_m":c.get("rmse_m"),k+"_angle_deg":c.get("angle_deg"),
                       k+"_interval_length_m":c.get("interval_length_m")})
    b=r["bootstrap"]
    result.update(bootstrap_valid_fraction=b["valid_fraction"],bootstrap_rmse_p90_m=b["rmse_m_p90"],
                  bootstrap_angle_p90_deg=b["angle_deg_p90"],bootstrap_statuses=json.dumps(b["statuses"]))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--l0",type=Path,required=True)
    p.add_argument("--l1",type=Path,help="Optional L1 metrics for correlation; never a prefilter")
    p.add_argument("--out",type=Path,required=True);p.add_argument("--cell",type=float,default=2.)
    p.add_argument("--group",choices=["raw_lines","merged_lines"],default="raw_lines")
    p.add_argument("--frames",type=int,nargs="+")
    p.add_argument("--bootstrap",type=int,default=50);p.add_argument("--seed",type=int,default=17)
    p.add_argument("--review-per-frame",type=int,default=12)
    a=p.parse_args(argv)
    cfg=replace(StabilitySettings(),bootstrap=a.bootstrap,seed=a.seed);cfg.validate()
    if a.review_per_frame<0:p.error("Invalid review count")
    m=a.data/"manifest.json";ap=a.data/"annotations.json";l0p=a.l0/"summary.json"
    manifest=read(m);ann=read(ap) if ap.exists() else dict(records=[]);l0=read(l0p)
    digest=sha(m);ad=sha(ap) if ap.exists() else None
    if l0["manifest_sha256"]!=digest or l0["annotations_sha256"]!=ad or (ap.exists() and ann["manifest_sha256"]!=digest):
        raise ValueError("M0/L0 provenance mismatch")
    cell=f"cell_{a.cell:g}m";arm=l0["arms"][cell]
    l1=None
    if a.l1:
        l1=read(a.l1/"summary.json")
        if l1["manifest_sha256"]!=digest or l1["l0_summary_sha256"]!=sha(l0p) or l1["cell"]!=cell:
            raise ValueError("L1 provenance mismatch")
    rows={int(r["frame"]):r for r in manifest["frames"]}
    frames=sorted(set(a.frames or map(int,arm["frames"])))
    if not frames or any(str(f) not in arm["frames"] or f not in rows for f in frames):raise ValueError("Missing frame")
    X=validate_extrinsic(np.asarray(manifest["T_camera_lidar"],float));K=np.asarray(manifest["K"],float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or not np.allclose(K[2],[0,0,1]):raise ValueError("Invalid rectified K")
    if a.out.exists() and any(a.out.iterdir()):raise FileExistsError("Use fresh output")
    a.out.mkdir(parents=True,exist_ok=True)
    out=dict(manifest_sha256=digest,annotations_sha256=ad,l0_summary_sha256=sha(l0p),
             l1_summary_sha256=sha(a.l1/"summary.json") if a.l1 else None,
             cell=cell,group=a.group,settings=asdict(cfg),review_per_frame=a.review_per_frame,
             warning="Conditional stability: L0 plane assignments selected on full cloud; scan provenance after voxel selection, shared GLIM error. Not independent validation or absolute accuracy. Fixed interval bootstrap conditional on successful trials. No stability threshold or calibration.",frames={})
    flat=[];review=[]
    for f in frames:
        start=perf_counter()
        src=a.data/f"frame_{f:06d}";folder=a.out/f"frame_{f:06d}";folder.mkdir()
        cp=src/"cloud.npz";ip=src/"image.png";old=arm["frames"][str(f)]
        if sha(cp)!=rows[f]["cloud_sha256"] or sha(cp)!=old["cloud_sha256"] or sha(ip)!=rows[f]["image_sha256"] or sha(ip)!=old["image_sha256"]:
            raise ValueError("M0 archive changed")
        with np.load(cp,allow_pickle=False) as z:cloud=np.asarray(z["xyz"],float);scans=z["scan_frames"]
        if len(cloud)!=rows[f]["points"] or scans.shape!=(len(cloud),) or scans.dtype.kind not in "iu" or not np.isfinite(cloud).all():
            raise ValueError("Invalid cloud")
        img=cv2.imread(str(ip))
        if img is None:raise ValueError("Unreadable image")
        xyz,ids,hashes=load_archive(a.l0/cell/f"frame_{f:06d}",a.group,len(cloud))
        if len(xyz)!=old[a.group]:raise ValueError("L0 count mismatch")
        l1metrics={}
        if l1 is not None:
            g=l1["frames"][str(f)]["groups"][a.group]
            if g["archive_hashes"]!=hashes:raise ValueError("L1 array mismatch")
            for key in g["arms"]:
                values=read(a.l1/f"frame_{f:06d}"/a.group/(key+"_metrics.json"))
                if len(values)!=len(xyz) or any(r["parent_id"]!=i for i,r in enumerate(values)):raise ValueError("L1 parent mismatch")
                l1metrics[key]=values
        refs=[r for r in ann["records"] if int(r["frame"])==f and r.get("manual_identity_confirmed")]
        manual=[dict(id=r["id"],**manual_matches(np.asarray(r["xyz_lidar"]),xyz)) for r in refs]
        manual_ids={r["best"]["index"] for r in manual if r["best"] is not None}
        visible=[];all_results=[]
        for i,(line,pair) in enumerate(zip(xyz,ids)):
            uv=clip_projected_line(line@X[:3,:3].T+X[:3,3],K,img.shape[:2],min_length=15.)
            if uv is not None:visible.append(i)
            # Stable per-frame/per-parent seed, independent of loop order.
            seed=np.random.SeedSequence([cfg.seed,f,i]).generate_state(1)[0]
            r=evaluate(line,[cloud[q] for q in pair],[scans[q] for q in pair],rows[f]["source_frames"],cfg,seed=seed)
            r.update(id=i,xyz_lidar=line.tolist(),visible=uv is not None,length_m=float(np.linalg.norm(np.diff(line,axis=0))),
                     manual_best=i in manual_ids)
            if l1metrics:r["l1_coverage"]={k:v[i]["length_coverage"] for k,v in l1metrics.items()}
            all_results.append(r)
            flat.append(dict(frame=f,id=i,visible=r["visible"],manual_best=r["manual_best"],length_m=r["length_m"],
                             **flatten(r),**{k+"_coverage":v for k,v in r.get("l1_coverage",{}).items()}))
        # Blind simple random review of projected candidates, plus separately
        # marked reference candidates (never count those as random sample).
        rng=np.random.default_rng(np.random.SeedSequence([cfg.seed,f,999]))
        random_ids=set(map(int,rng.choice(visible,min(len(visible),a.review_per_frame),replace=False))) if visible else set()
        review_dir=folder/"review";review_dir.mkdir()
        for i in sorted(random_ids|manual_ids):
            image=img.copy();count=draw(image,[dict(xyz=xyz[i])],X,K,(0,255,0))
            cv2.putText(image,f"Frame {f} | candidate {i}",(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            if not cv2.imwrite(str(review_dir/f"line_{i:04d}.png"),np.vstack((img,image))):raise OSError("Cannot save review")
            review.append(dict(frame=f,id=i,random_sample=i in random_ids,reference_candidate=i in manual_ids,
                               projected=bool(count),label="",notes=""))
        write(folder/"stability.json",all_results)
        checks=[]
        for r in manual:
            best=r["best"]
            checks.append(dict(id=r["id"],match=best,stability=flatten(all_results[best["index"]]) if best else None))
        def summarize(group):
            result=dict(count=len(group))
            for k in ("odd_even","temporal"):
                valid=[r[k]["comparison"] for r in group if r[k]["comparison"] is not None]
                result[k]=dict(valid=len(valid),median_rmse_m=float(np.median([x["rmse_m"] for x in valid])) if valid else None,
                               median_angle_deg=float(np.median([x["angle_deg"] for x in valid])) if valid else None)
            result["bootstrap_median_valid_fraction"]=float(np.median([r["bootstrap"]["valid_fraction"] for r in group])) if group else None
            return result
        info=dict(archive_hashes=hashes,all=summarize(all_results),visible=summarize([all_results[i] for i in visible]),
                  manual=checks,random_review_count=len(random_ids),seconds=perf_counter()-start)
        out["frames"][str(f)]=info;write(folder/"summary.json",info);write(a.out/"summary.json",out)
        print(f"{f:06d}: {len(xyz)} candidates, {len(visible)} visible; odd-even {info['all']['odd_even']['valid']}, temporal {info['all']['temporal']['valid']}, {info['seconds']:.1f}s",flush=True)
    for name,data,fields in (("stability.csv",flat,None),("review.csv",review,["frame","id","random_sample","reference_candidate","projected","label","notes"])):
        if not data and fields is None:continue
        with (a.out/name).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields or list(data[0]));w.writeheader();w.writerows(data)
    return 0


if __name__=="__main__":raise SystemExit(main())
