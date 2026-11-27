#!/usr/bin/env python3
"""L0 automatic LiDAR line extraction on prepared M0 clouds."""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from spade_calib.calibration.lidar_line_extraction import Settings,extract_lines,line_relation
from spade_calib.calibration.line_matching import clip_projected_line
from spade_calib.selection.scale_segmentation import validate_extrinsic


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")


def serialize(line,i):
    return dict(id=i,xyz_lidar=line["xyz"].tolist(),cells=line["cells"],
                length_m=float(np.linalg.norm(np.diff(line["xyz"],axis=0))),
                plane_angle_deg=line["angle_deg"],horizontal_plane=line["horizontal_plane"],
                support=line["support"],planes=[dict(normal=p["normal"].tolist(),offset_m=p["offset"],
                    rmse_m=p["rmse_m"],spread_m=p["spread_m"],thickness_ratio=p["thickness_ratio"],
                    inliers=len(p["ids"])) for p in line["planes"]])


def compare(reference,lines):
    rows=[]
    for i,l in enumerate(lines):
        rel=line_relation(reference,l["xyz"])
        # Infinite-line distance on manual endpoints alone can favor short fragments:
        # also require a meaningful overlapping span. No GT.
        if rel["angle_deg"]<=10 and rel["mean_distance_m"]<=.30 and rel["reference_overlap_fraction"]>=.30:
            rows.append(dict(id=i,**rel))
    rows.sort(key=lambda r:(r["mean_distance_m"],-r["reference_overlap_fraction"]))
    return dict(candidate_count=len(rows),best=rows[0] if rows else None,candidates=rows[:10])


def save_lines(folder,name,lines):
    arr=np.asarray([r["xyz"] for r in lines],float).reshape(-1,2,3)
    ids=[];offsets=[0]
    for r in lines:
        for p in r["planes"]:
            ids.extend(p["ids"].tolist());offsets.append(len(ids))
    np.savez_compressed(folder/(name+".npz"),xyz_lidar=arr,plane_point_indices=np.asarray(ids,int),
                        plane_offsets=np.asarray(offsets,int))
    write(folder/(name+".json"),[serialize(r,i) for i,r in enumerate(lines)])


def draw(img,lines,X,K,color,labels=False):
    count=0
    for i,l in enumerate(lines):
        q=l["xyz"]@X[:3,:3].T+X[:3,3]
        uv=clip_projected_line(q,K,img.shape[:2],min_length=5.)
        if uv is None:continue
        xy=np.rint(uv).astype(int)
        cv2.line(img,tuple(xy[0]),tuple(xy[1]),color,1,cv2.LINE_AA)
        if labels:
            cv2.putText(img,str(i),tuple(np.rint(uv.mean(0)).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1,cv2.LINE_AA)
        count+=1
    return count


def save_ply(path,p,lines,manual):
    # Portable colored PLY (sampled points for lines; exact endpoints in NPZ).
    blocks=[p];colors=[np.full((len(p),3),105,np.uint8)]
    for group,color in ((lines,[0,255,255]),(manual,[255,0,255])):
        for l in group:
            n=max(2,int(np.ceil(np.linalg.norm(np.diff(l["xyz"],axis=0))/.02))+1)
            q=np.linspace(l["xyz"][0],l["xyz"][1],n)
            blocks.append(q);colors.append(np.tile(color,(n,1)).astype(np.uint8))
    xyz=np.concatenate(blocks);rgb=np.concatenate(colors)
    dtype=np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("red","u1"),("green","u1"),("blue","u1")])
    data=np.empty(len(xyz),dtype=dtype)
    for i,key in enumerate(("x","y","z")):data[key]=xyz[:,i]
    for i,key in enumerate(("red","green","blue")):data[key]=rgb[:,i]
    with path.open("wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"%len(xyz)).encode())
        data.tofile(f)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--frames",type=int,nargs="+",help="Default: every prepared anchor, not just labeled ones")
    p.add_argument("--cells",type=float,nargs="+",default=[1.,2.])
    p.add_argument("--max-range",type=float,default=60.)
    p.add_argument("--trials",type=int,default=256)
    p.add_argument("--remove-ground",action="store_true",help="Optional global plane removal; not default")
    a=p.parse_args(argv)
    m=a.data/"manifest.json";ap=a.data/"annotations.json"
    manifest=json.loads(m.read_text());ann=json.loads(ap.read_text()) if ap.exists() else dict(records=[])
    if ap.exists() and ann["manifest_sha256"]!=sha(m):raise ValueError("Annotation manifest mismatch")
    X=validate_extrinsic(np.asarray(manifest["T_camera_lidar"],float));K=np.asarray(manifest["K"],float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or not np.allclose(K[2],[0,0,1]):raise ValueError("Invalid rectified K")
    rows={int(r["frame"]):r for r in manifest["frames"]}
    frames=sorted(set(a.frames or rows))
    if not frames or not set(frames)<=set(rows):raise ValueError("Requested unprepared frame")
    settings=[replace(Settings(),cell_m=c,trials=a.trials,max_range_m=a.max_range) for c in sorted(set(a.cells))]
    for cfg in settings:cfg.validate()
    if a.out.exists() and any(a.out.iterdir()):raise FileExistsError("Use fresh output")
    a.out.mkdir(parents=True,exist_ok=True)
    result=dict(manifest_sha256=sha(m),annotations_sha256=sha(ap) if ap.exists() else None,
                window=manifest["window"],voxel_m=manifest["voxel_m"],ghost_filter=manifest.get("ghost_filter"),
                warning="LiDAR-only geometry candidates; manual lines are same-cloud references, not GT. Ground retained by default; optional global ground removal is approximate. No occlusion or matching. Support scans correlated and not independent uncertainty estimates.",
                remove_ground=a.remove_ground,arms={})
    for cfg in settings:
        name=f"cell_{cfg.cell_m:g}m";arm=a.out/name;arm.mkdir()
        record=dict(settings=asdict(cfg),frames={});result["arms"][name]=record
        for f in frames:
            row=rows[f];src=a.data/f"frame_{f:06d}";folder=arm/f"frame_{f:06d}";folder.mkdir()
            cp=src/"cloud.npz";ip=src/"image.png"
            if sha(cp)!=row["cloud_sha256"] or sha(ip)!=row["image_sha256"]:raise ValueError("M0 archive changed")
            with np.load(cp,allow_pickle=False) as z:
                xyz=z["xyz"];scans=z["scan_frames"]
            img=cv2.imread(str(ip))
            if img is None:raise ValueError("Unreadable image")
            # Preserve original archive indices even when optional ground is removed.
            keep=np.arange(len(xyz));ground=None
            if a.remove_ground:
                from spade_calib.lidar.ground_planes import extract_lidar_ground
                candidate,ground=extract_lidar_ground(xyz)
                mask=abs(xyz@candidate.plane.normal+candidate.plane.offset_m)>.14
                keep=keep[mask];ground["removed_points"]=int((~mask).sum())
            start=perf_counter()
            raw,merged,stats=extract_lines(xyz[keep],scans[keep],cfg)
            elapsed=perf_counter()-start
            # The raw and merged objects may share plane dictionaries.
            # Serialize with remapped copies so indices are always M0 original IDs.
            def remap(group):
                return [dict(l,planes=tuple(dict(p,ids=keep[p["ids"]]) for p in l["planes"])) for l in group]
            save_lines(folder,"raw_lines",remap(raw));save_lines(folder,"merged_lines",remap(merged))
            labels=[r for r in ann["records"] if int(r["frame"])==f and r.get("manual_identity_confirmed")]
            manual=[dict(xyz=np.asarray(r["xyz_lidar"],float)) for r in labels]
            checks=[dict(id=r["id"],raw=compare(ref["xyz"],raw),merged=compare(ref["xyz"],merged)) for r,ref in zip(labels,manual)]
            overlay=img.copy();visible=draw(overlay,merged,X,K,(255,255,0),labels=True)
            reference=img.copy();draw(reference,manual,X,K,(255,0,255))
            combined=np.vstack((img,overlay,reference))
            if not cv2.imwrite(str(folder/"projection.png"),combined):raise OSError("Cannot save projection")
            save_ply(folder/"cloud_lines.ply",xyz,merged,manual)
            info=dict(**stats,extraction_seconds=elapsed,visible_merged_lines=visible,ground=ground,
                      manual_comparison=checks,cloud_sha256=sha(cp),image_sha256=sha(ip))
            record["frames"][str(f)]=info
            write(folder/"summary.json",info);write(a.out/"summary.json",result)
            print(f"{name} {f:06d}: {len(raw)} raw -> {len(merged)} merged, {visible} projected, {elapsed:.1f}s",flush=True)
    return 0


if __name__=="__main__":raise SystemExit(main())
