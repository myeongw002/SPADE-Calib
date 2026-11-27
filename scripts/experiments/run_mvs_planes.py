#!/usr/bin/env python3
"""MVS2 ground and finite plane extraction on metric MVS PLY."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from spade_calib.camera.mvs_planes import Settings,sample_cloud,experiment,normal_rank


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ply",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--voxel",type=float,default=.10)
    p.add_argument("--max-range",type=float,default=65.)
    p.add_argument("--max-points",type=int,default=40000)
    p.add_argument("--distance",type=float,default=.12)
    p.add_argument("--normal-deg",type=float,default=20.)
    p.add_argument("--connectivity",type=float,default=.65)
    p.add_argument("--min-points",type=int,default=100)
    p.add_argument("--min-extent",type=float,default=.5)
    p.add_argument("--max-planes",type=int,default=12)
    p.add_argument("--iterations",type=int,default=512)
    a=p.parse_args()
    import open3d as o3d
    if a.out.exists() and any(a.out.iterdir()):raise FileExistsError("Use fresh output")
    if not a.ply.is_file():raise FileNotFoundError(a.ply)
    pc=o3d.io.read_point_cloud(str(a.ply))
    if not pc.has_normals():raise ValueError("Expected MVS-provided point normals")
    cfg=Settings(voxel_m=a.voxel,max_range_m=a.max_range,max_points=a.max_points,
                 distance_m=a.distance,normal_deg=a.normal_deg,connectivity_m=a.connectivity,
                 min_points=a.min_points,min_extent_m=a.min_extent,max_planes=a.max_planes,
                 iterations=a.iterations)
    p,n,source,diag=sample_cloud(np.asarray(pc.points),np.asarray(pc.normals),cfg)
    a.out.mkdir(parents=True,exist_ok=True)
    print(f"Sampled {len(p)} points; extracting ground and plane candidates",flush=True)
    start=time.monotonic()
    ground,arms,remaining,info=experiment(p,n,cfg)
    def save_cloud(path,ids,colors=None):
        c=o3d.geometry.PointCloud();c.points=o3d.utility.Vector3dVector(p[ids]);c.normals=o3d.utility.Vector3dVector(n[ids])
        if colors is not None:c.colors=o3d.utility.Vector3dVector(colors)
        elif pc.has_colors():c.colors=o3d.utility.Vector3dVector(np.asarray(pc.colors)[source[ids]])
        if not o3d.io.write_point_cloud(str(path),c):raise OSError(f"Cannot write {path}")
    def write(path,d):path.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n")
    save_cloud(a.out/"sampled.ply",np.arange(len(p)))
    save_cloud(a.out/"nonground_input.ply",remaining)
    np.savez_compressed(a.out/"sampled.npz",xyz_m=p,normals=n,source_indices=source,remaining_indices=remaining)
    report=dict(status="ok" if arms else "ground_failed",input_ply=str(a.ply.resolve()),
                convention="anchor_camera_metric_x_right_y_down_z_forward",settings=asdict(cfg),
                sampling=diag,ground_diagnostics=info,ground={},arms={},elapsed_seconds=time.monotonic()-start,
                caveat="Normal rank is an equal-plane-weight translation-only proxy. No physical ground truth, plane correspondence or extrinsic optimization. Thresholds are pilot heuristics.")
    for name,result in ground.items():
        if result is None:report["ground"][name]={"status":"failed"};continue
        ids,row=result;save_cloud(a.out/f"{name}.ply",ids)
        np.save(a.out/f"{name}_indices.npy",ids)
        report["ground"][name]={"status":"ok",**row}
    rng=np.random.default_rng(5);palette=rng.uniform(.2,1,(cfg.max_planes,3))
    for name,planes in arms.items():
        folder=a.out/name;folder.mkdir()
        rgb=np.full((len(p),3),.25);rgb[np.setdiff1d(np.arange(len(p)),remaining)]=[.15,.65,.15]
        rows=[];indices={}
        for i,(ids,row) in enumerate(planes):
            key=f"plane_{i:02d}";save_cloud(folder/f"{key}.ply",ids)
            rgb[ids]=palette[i]
            rows.append({"patch_id":i,**row});indices[key]=ids
        save_cloud(folder/"colored_planes.ply",np.arange(len(p)),rgb)
        np.savez_compressed(folder/"indices.npz",**indices)
        write(folder/"planes.json",rows)
        combined=[ground["G1"][1]]+rows
        report["arms"][name]=dict(planes=rows,plane_count=len(rows),support_count=sum(r["support_count"] for r in rows),
            translation_proxy_nonground=normal_rank(rows),translation_proxy_with_ground=normal_rank(combined))
    write(a.out/"summary.json",report)
    print(json.dumps({k:{"plane_count":v["plane_count"],"translation_proxy_with_ground":v["translation_proxy_with_ground"]} for k,v in report["arms"].items()},indent=2))
    return 0 if arms else 1


if __name__=="__main__":raise SystemExit(main())
