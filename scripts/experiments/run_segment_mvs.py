#!/usr/bin/env python3
"""MVS1: densify archived SfM scale segments without changing camera poses."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from scipy.spatial.transform import Rotation
from spade_calib.io.colmap import load_registered_images,resolve_colmap_images
from spade_calib.io.kitti import frame_id_from_name
from spade_calib.camera.segment_mvs import read_text_model,subset_text_model,anchor_metric


def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n")


def executable(name):
    path=shutil.which(name)
    if path is None:raise FileNotFoundError(f"Executable not found: {name}")
    return str(Path(path).resolve())


def run(args,cwd,log,commands):
    args=list(map(str,args));print(" ".join(args),flush=True)
    record=dict(argv=args,cwd=str(cwd),log=str(log));commands.append(record)
    start=time.monotonic()
    with log.open("w") as f:
        proc=subprocess.run(args,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
    record.update(returncode=proc.returncode,seconds=time.monotonic()-start)
    if proc.returncode:
        raise RuntimeError(f"Command failed ({proc.returncode}); see {log}")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=Path("outputs/segment07_trial1"))
    p.add_argument("--image-dir",type=Path,required=True,help="Original COLMAP image root (names must match)")
    p.add_argument("--sfm",type=Path,help="Override relocated original COLMAP sparse model")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--segment-id",type=int,nargs="+")
    p.add_argument("--contains-frame",type=int,nargs="+")
    p.add_argument("--image-stride",type=int,default=1)
    p.add_argument("--backend",choices=["openmvs","colmap","both"],default="openmvs")
    p.add_argument("--colmap",default="colmap")
    p.add_argument("--interface",default="InterfaceCOLMAP")
    p.add_argument("--densify",default="DensifyPointCloud")
    p.add_argument("--max-image-size",type=int,default=1600)
    p.add_argument("--resolution-level",type=int,default=0)
    p.add_argument("--number-views",type=int,default=8)
    p.add_argument("--fusion-views",type=int,default=3)
    p.add_argument("--cuda-device",type=int,help="OpenMVS --cuda-device; omit to use installed default")
    p.add_argument("--threads",type=int,default=0,help="OpenMVS max threads; 0 auto")
    p.add_argument("--prepare-only",action="store_true",help="Export per-segment sparse text models only")
    a=p.parse_args(argv)
    if min(a.image_stride,a.max_image_size,a.number_views)<1 or a.resolution_level<0 or a.fusion_views<2 or a.threads<0:
        p.error("Invalid settings")
    a.out=a.out.resolve();a.image_dir=a.image_dir.resolve();a.run=a.run.resolve()
    if a.out.exists() and (not a.out.is_dir() or any(a.out.iterdir())):raise FileExistsError("Use fresh output")
    cfg=read(a.run/"config.json");he=read(a.run/"segmented_handeye.json")["fit"]
    archived=read(a.run/"summary.json")
    old={int(x["segment_id"]):x for x in archived["segments"]}
    source=resolve_colmap_images(a.sfm or Path(cfg["sfm"])).parent
    records=load_registered_images(source)
    if not a.image_dir.is_dir():raise FileNotFoundError(a.image_dir)
    segments=[]
    for s in he["segments"]:
        k=int(s["segment_id"]);names=s["image_names"]
        frames=[frame_id_from_name(n) for n in names]
        if a.segment_id and k not in a.segment_id:continue
        if a.contains_frame and not any(min(frames)<=f<=max(frames) for f in a.contains_frame):continue
        anchor=names[(len(names)-1)//2];scale=float(s["sfm_scale_to_metric"])
        if k in old and (frame_id_from_name(anchor)!=old[k]["anchor_frame"] or not np.isclose(scale,old[k]["scale"],rtol=1e-9,atol=0)):
            raise ValueError(f"Archived anchor/scale mismatch: {k}")
        selected=[n for i,n in enumerate(names) if i%a.image_stride==0 or n==anchor or i==len(names)-1]
        if len(selected)<max(2,a.fusion_views if not a.prepare_only else 2):
            raise ValueError(f"Too few images: {k}")
        for n in selected:
            if n not in records or not (a.image_dir/n).is_file():
                raise FileNotFoundError(f"Missing registered image/file: {n}; check --image-dir")
        segments.append((k,anchor,scale,selected,len(names)))
    if not segments or (a.segment_id and not set(a.segment_id)<=set(x[0] for x in segments)):
        raise ValueError("No selected segment or unknown segment ID")
    colmap=executable(a.colmap)
    if not a.prepare_only:
        try:import open3d as o3d
        except ImportError as exc:raise SystemExit("Install open3d in this venv for PLY conversion") from exc
        if a.backend in ("openmvs","both"):
            interface=executable(a.interface);densify=executable(a.densify)
    a.out.mkdir(parents=True,exist_ok=True)
    summary=dict(status="running",warning="MVS only; original global SfM gauge and fixed archived HE segment scales. "
                 "No pose BA, GT or LiDAR matching. Dense surface accuracy and calibration not evaluated. "
                 "Image stride/resolution and fusion settings are pilot choices.",
                 source_model=str(source),image_dir=str(a.image_dir),run=str(a.run),
                 settings=dict(backend=a.backend,image_stride=a.image_stride,max_image_size=a.max_image_size,
                               resolution_level=a.resolution_level,number_views=a.number_views,
                               fusion_views=a.fusion_views,cuda_device=a.cuda_device,threads=a.threads,
                               prepare_only=a.prepare_only),commands=[],segments={})
    full=a.out/"global_text";full.mkdir()
    run([colmap,"model_converter","--input_path",source,"--output_path",full,"--output_type","TXT"],
        a.out,a.out/"model_converter.log",summary["commands"])
    model=read_text_model(full);write(a.out/"summary.json",summary)
    failed=False
    for k,anchor,scale,names,total in segments:
        folder=a.out/f"segment_{k:02d}";folder.mkdir()
        rec=records[anchor];q=np.asarray(rec.qvec_wxyz);R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        t=np.asarray(rec.tvec,float)
        info=dict(status="running",anchor_image=anchor,anchor_frame=frame_id_from_name(anchor),
                  sfm_scale_to_metric=scale,anchor_R_camera_world=R.tolist(),anchor_t_camera_world=t.tolist(),
                  original_image_count=total,selected_images=names,backends={},commands=[])
        summary["segments"][str(k)]=info
        try:
            info["sparse"]=subset_text_model(model,names,folder/"sparse")
            if not a.prepare_only:
                dense=folder/"dense";dense.mkdir()
                run([colmap,"image_undistorter","--image_path",a.image_dir,
                     "--input_path",folder/"sparse","--output_path",dense,
                     "--output_type","COLMAP","--max_image_size",a.max_image_size],
                    folder,folder/"undistort.log",info["commands"])
                # Undistortion may change intrinsics/image dimensions; poses must not change.
                und=load_registered_images(dense/"sparse")
                if set(und)!=set(names):raise ValueError("Undistorted image membership mismatch")
                for n in names:
                    b=records[n];c=und[n]
                    qb=np.array(b.qvec_wxyz);qc=np.array(c.qvec_wxyz)
                    if min(np.linalg.norm(qb-qc),np.linalg.norm(qb+qc))>1e-6 or not np.allclose(b.tvec,c.tvec,atol=1e-7,rtol=1e-9):
                        raise ValueError(f"Undistortion changed pose: {n}")
                for backend in (["openmvs","colmap"] if a.backend=="both" else [a.backend]):
                    work=folder/backend;work.mkdir()
                    result=dict(status="running");info["backends"][backend]=result
                    try:
                        if backend=="openmvs":
                            scene=work/"scene.mvs";cloud=work/"scene_dense.ply"
                            run([interface,"-i",dense,"-o",scene,"--image-folder",dense/"images","-w",work],
                                work,work/"interface.log",info["commands"])
                            cmd=[densify,"-i",scene,"-o",work/"scene_dense.mvs","-w",work,
                                 "--resolution-level",a.resolution_level,"--number-views",a.number_views,
                                 "--number-views-fuse",a.fusion_views,"--max-threads",a.threads]
                            if a.cuda_device is not None:cmd+=["--cuda-device",a.cuda_device]
                            run(cmd,work,work/"densify.log",info["commands"])
                        else:
                            cloud=work/"fused.ply"
                            run([colmap,"patch_match_stereo","--workspace_path",dense,
                                 "--workspace_format","COLMAP","--PatchMatchStereo.geom_consistency","true"],
                                work,work/"patch_match.log",info["commands"])
                            run([colmap,"stereo_fusion","--workspace_path",dense,"--workspace_format","COLMAP",
                                 "--input_type","geometric","--output_path",cloud,
                                 "--StereoFusion.min_num_pixels",a.fusion_views],
                                work,work/"fusion.log",info["commands"])
                        if not cloud.is_file():raise FileNotFoundError(f"No expected dense PLY: {cloud}")
                        pc=o3d.io.read_point_cloud(str(cloud));xyz=np.asarray(pc.points)
                        if not len(xyz):raise ValueError("Empty dense cloud")
                        metric=anchor_metric(xyz,R,t,scale)
                        pc.points=o3d.utility.Vector3dVector(metric)
                        if pc.has_normals():pc.normals=o3d.utility.Vector3dVector(np.asarray(pc.normals)@R.T)
                        outply=work/"cloud_anchor_camera_m.ply"
                        if not o3d.io.write_point_cloud(str(outply),pc):raise OSError(f"Could not write {outply}")
                        result.update(status="ok",point_count=len(metric),
                                      anchor_range_m_p10_p50_p90=np.percentile(np.linalg.norm(metric,axis=1),[10,50,90]).tolist(),
                                      raw_ply=str(cloud),metric_ply=str(outply),has_normals=pc.has_normals(),has_colors=pc.has_colors())
                    except Exception as exc:
                        failed=True;result.update(status="failed",error=str(exc));print(f"{k} {backend}: {exc}",flush=True)
            info["status"]="prepared" if a.prepare_only else ("ok" if all(v["status"]=="ok" for v in info["backends"].values()) else "failed")
        except Exception as exc:
            failed=True;info.update(status="failed",error=str(exc));print(f"Segment {k}: {exc}",flush=True)
        write(folder/"report.json",info);write(a.out/"summary.json",summary)
    summary["status"]="failed_or_partial" if failed else ("prepared" if a.prepare_only else "ok")
    write(a.out/"summary.json",summary)
    return int(failed)


if __name__=="__main__":raise SystemExit(main())
