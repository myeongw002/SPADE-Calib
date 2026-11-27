#!/usr/bin/env python3
"""L1 controlled post-filter of L0 raw/merged lines on original M0 points."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import csv
import cv2
import numpy as np
from run_lidar_line_extraction import sha,write,draw,save_ply
from spade_calib.calibration.lidar_line_support import sample_support,support_metrics,manual_matches,paired_merge_error
from spade_calib.selection.scale_segmentation import validate_extrinsic


def read(p):return json.loads(p.read_text())


def load_archive(folder,name,point_count):
    path=folder/(name+".npz");jp=folder/(name+".json")
    with np.load(path,allow_pickle=False) as z:
        xyz=np.asarray(z["xyz_lidar"],float);ids=z["plane_point_indices"];off=z["plane_offsets"]
    meta=read(jp)
    if (xyz.ndim!=3 or xyz.shape[1:]!=(2,3) or not np.isfinite(xyz).all()
            or len(meta)!=len(xyz) or ids.ndim!=1 or ids.dtype.kind not in "iu"
            or off.shape!=(2*len(xyz)+1,) or off.dtype.kind not in "iu"
            or off[0]!=0 or off[-1]!=len(ids) or np.any(np.diff(off)<=0)
            or np.any(ids<0) or np.any(ids>=point_count)):
        raise ValueError(f"Invalid L0 archive {path}")
    pairs=[]
    for i,r in enumerate(meta):
        if r["id"]!=i or not np.array_equal(r["xyz_lidar"],xyz[i]):
            raise ValueError("L0 JSON/NPZ endpoint mismatch")
        pair=[ids[off[2*i+j]:off[2*i+j+1]] for j in range(2)]
        if any(len(q)!=r["planes"][j]["inliers"] or len(np.unique(q))!=len(q) for j,q in enumerate(pair)):
            raise ValueError("L0 support index mismatch")
        pairs.append(pair)
    return xyz,pairs,dict(npz_sha256=sha(path),json_sha256=sha(jp))


def aggregate(metrics,xyz):
    length=np.linalg.norm(xyz[:,1]-xyz[:,0],axis=1)
    def median(key):
        return float(np.median([m[key] for m in metrics])) if metrics else None
    return dict(parent_count=len(metrics),parents_with_span=sum(m["accepted"] for m in metrics),
                original_like_count_pass=sum(m["original_like_sample_count_pass"] for m in metrics),
                median_length_coverage=median("length_coverage"),
                median_longest_span_m=median("longest_supported_span_m"),
                median_far_distance_m=median("median_far_distance_m"),
                total_parent_length_m=float(length.sum()),
                total_supported_length_m=float(sum(m["supported_length_m"] for m in metrics)),
                accepted_span_count=sum(len(m["accepted_spans"]) for m in metrics))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--l0",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--cell",type=float,default=2.)
    p.add_argument("--frames",type=int,nargs="+")
    p.add_argument("--threshold-pairs",type=float,nargs="+",default=[.03,.06,.05,.10,.10,.20,.15,.30])
    p.add_argument("--sample-step",type=float,default=.005)
    p.add_argument("--min-span",type=float,default=.30)
    p.add_argument("--save-ply",action="store_true",help="Optional larger files for each arm")
    a=p.parse_args(argv)
    if len(a.threshold_pairs)%2 or not np.isfinite(a.threshold_pairs).all() or not np.isfinite([a.sample_step,a.min_span]).all() or min(a.sample_step,a.min_span)<=0:
        p.error("Invalid thresholds/step/span")
    pairs=[tuple(x) for x in np.asarray(a.threshold_pairs).reshape(-1,2)]
    if len(set(pairs))!=len(pairs) or any(not 0<x<=y for x,y in pairs):p.error("Require unique positive near <= far pairs")
    names=[f"nn_{x:g}_{y:g}" for x,y in pairs]
    m=a.data/"manifest.json";ap=a.data/"annotations.json";l0p=a.l0/"summary.json"
    manifest=read(m);ann=read(ap) if ap.exists() else dict(records=[]);l0=read(l0p)
    digest=sha(m);ad=sha(ap) if ap.exists() else None
    if l0["manifest_sha256"]!=digest or l0["annotations_sha256"]!=ad or (ap.exists() and ann["manifest_sha256"]!=digest):
        raise ValueError("M0/L0 archive provenance mismatch")
    cell=f"cell_{a.cell:g}m";arm=l0["arms"][cell]
    rows={int(r["frame"]):r for r in manifest["frames"]}
    frames=sorted(set(a.frames or map(int,arm["frames"])))
    if not frames or any(str(f) not in arm["frames"] or f not in rows for f in frames):
        raise ValueError("Frame missing from M0/L0")
    X=validate_extrinsic(np.asarray(manifest["T_camera_lidar"],float));K=np.asarray(manifest["K"],float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or not np.allclose(K[2],[0,0,1]):raise ValueError("Invalid rectified K")
    if a.out.exists() and any(a.out.iterdir()):raise FileExistsError("Use fresh output")
    a.out.mkdir(parents=True,exist_ok=True)
    out=dict(manifest_sha256=digest,annotations_sha256=ad,l0_summary_sha256=sha(l0p),
             cell=cell,settings=dict(sample_step_m=a.sample_step,min_contiguous_span_m=a.min_span,
                                    threshold_pairs=[list(x) for x in pairs]),
             warning="Post-filter only: samples within frozen L0 endpoints, no recovery of rejected plane pairs or voxel clipping. L0 manual references same-cloud, not GT. Span criterion differs from original Yuan count gate. No matching/optimization.",frames={})
    csvrows=[]
    for f in frames:
        start=perf_counter()
        src=a.data/f"frame_{f:06d}";folder=a.out/f"frame_{f:06d}";folder.mkdir()
        cp=src/"cloud.npz";ip=src/"image.png";old=arm["frames"][str(f)]
        if sha(cp)!=rows[f]["cloud_sha256"] or sha(cp)!=old["cloud_sha256"] or sha(ip)!=rows[f]["image_sha256"] or sha(ip)!=old["image_sha256"]:
            raise ValueError("M0 cloud/image changed")
        with np.load(cp,allow_pickle=False) as z:cloud=np.asarray(z["xyz"],float)
        if len(cloud)!=rows[f]["points"] or not np.isfinite(cloud).all():raise ValueError("Invalid cloud")
        img=cv2.imread(str(ip))
        if img is None:raise ValueError("Unreadable image")
        refs=[r for r in ann["records"] if int(r["frame"])==f and r.get("manual_identity_confirmed")]
        manual=[dict(xyz=np.asarray(r["xyz_lidar"],float)) for r in refs]
        arrays={};info=dict(groups={},paired_merge=[])
        for name in ("raw_lines","merged_lines"):
            xyz,ids,hashes=load_archive(a.l0/cell/f"frame_{f:06d}",name,len(cloud));arrays[name]=xyz
            if len(xyz)!=old[name]:raise ValueError("L0 count mismatch")
            group=folder/name;group.mkdir()
            metrics={k:[] for k in names};spans={k:[] for k in names}
            # All threshold arms share exact same NN distances and plane support.
            for i,(line,indices) in enumerate(zip(xyz,ids)):
                t,q,d=sample_support(line,cloud[indices[0]],cloud[indices[1]],a.sample_step)
                for key,(near,far) in zip(names,pairs):
                    m1=support_metrics(t,q,d,near,far,a.min_span);m1["parent_id"]=i
                    metrics[key].append(m1)
                    spans[key].extend(dict(xyz=np.asarray(s["xyz_lidar"]),parent_id=i) for s in m1["accepted_spans"])
            baseline=[dict(xyz=q) for q in xyz]
            base_overlay=img.copy();visible=draw(base_overlay,baseline,X,K,(255,255,0))
            base_checks=[dict(id=r["id"],**manual_matches(v["xyz"],xyz)) for r,v in zip(refs,manual)]
            g=dict(archive_hashes=hashes,baseline=dict(parent_count=len(xyz),visible_lines=visible,
                   manual_comparison=base_checks),arms={})
            info["groups"][name]=g
            panels=[("Original",img),("L0 "+name,base_overlay)]
            for key in names:
                saved=spans[key];coords=np.asarray([s["xyz"] for s in saved],float).reshape(-1,2,3)
                parents=np.asarray([s["parent_id"] for s in saved],int)
                np.savez_compressed(group/(key+"_spans.npz"),xyz_lidar=coords,parent_ids=parents)
                write(group/(key+"_metrics.json"),metrics[key])
                overlay=img.copy();count=draw(overlay,saved,X,K,(0,255,0))
                panels.append((key+" | "+str(len(saved))+" spans",overlay))
                if a.save_ply:save_ply(group/(key+".ply"),cloud,saved,manual)
                stat=aggregate(metrics[key],xyz)
                stat.update(visible_spans=count,manual_comparison=[dict(id=r["id"],**manual_matches(v["xyz"],coords)) for r,v in zip(refs,manual)])
                g["arms"][key]=stat
                csvrows.append(dict(frame=f,group=name,arm=key,**{k:v for k,v in stat.items() if k!="manual_comparison"}))
            ref_panel=img.copy();draw(ref_panel,manual,X,K,(255,0,255));panels.append(("M0 manual reference",ref_panel))
            images=[]
            for label,image in panels:
                panel=cv2.copyMakeBorder(image,25,0,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
                cv2.putText(panel,label,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA);images.append(panel)
            if not cv2.imwrite(str(group/"comparison.png"),np.vstack(images)):raise OSError("Cannot write comparison")
        for r,v in zip(refs,manual):
            info["paired_merge"].append(dict(id=r["id"],comparison=paired_merge_error(v["xyz"],arrays["raw_lines"],arrays["merged_lines"])))
        info["seconds"]=perf_counter()-start;out["frames"][str(f)]=info
        write(folder/"summary.json",info);write(a.out/"summary.json",out)
        print(f"{f:06d}: "+", ".join(f"{key} raw={info['groups']['raw_lines']['arms'][key]['parents_with_span']} merged={info['groups']['merged_lines']['arms'][key]['parents_with_span']}" for key in names),flush=True)
    if csvrows:
        with (a.out/"counts.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
    return 0


if __name__=="__main__":raise SystemExit(main())
