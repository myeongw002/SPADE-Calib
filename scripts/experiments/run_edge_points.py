#!/usr/bin/env python3
"""EP0: extract independent RGB, DA2 and LiDAR edge points; no matching."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from spade_calib.calibration.edge_points import dense_edges, sparse_projection, sparse_edges
from spade_calib.evaluation.kitti import kitti_gt_camera_lidar, parse_kitti_odom_calib
from spade_calib.selection.scale_segmentation import validate_extrinsic


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n")

def image(p):
    im=cv2.imread(str(p))
    if im is None:raise ValueError(f"Cannot read {p}")
    return im

def save(p,im):
    if not cv2.imwrite(str(p),im):raise OSError(f"Cannot write {p}")

def points_map(shape,uv):
    out=np.zeros(shape,np.uint8)
    q=np.floor(uv).astype(int)
    if len(q):out[q[:,1],q[:,0]]=255
    return out

def color_depth(D,lo,hi):
    valid=np.isfinite(D);u=np.zeros(D.shape,np.uint8)
    u[valid]=np.rint(np.clip((D[valid]-lo)/max(hi-lo,1e-8),0,1)*255).astype(np.uint8)
    out=cv2.applyColorMap(u,cv2.COLORMAP_TURBO);out[~valid]=0
    return out

def overlay(im,uv,color,radius=1):
    # Draw in source image coordinates; visualization only.
    for p in np.floor(uv).astype(int):
        cv2.circle(im,tuple(p),radius,color,-1)
    return im

def direction_map(shape,uv,normal,*,axial=False):
    hsv=np.zeros((*shape,3),np.uint8)
    q=np.floor(uv).astype(int)
    theta=np.arctan2(normal[:,1],normal[:,0])
    hue=np.floor((theta%(np.pi if axial else 2*np.pi))/(np.pi if axial else 2*np.pi)*180).astype(np.uint8)
    if len(q):
        hsv[q[:,1],q[:,0],0]=hue
        hsv[q[:,1],q[:,0],1:]=255
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

def panel(im,label):
    x=cv2.copyMakeBorder(im,25,0,0,0,cv2.BORDER_CONSTANT)
    cv2.putText(x,label,(5,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    return x

def percentiles(a):
    return np.percentile(a,[10,50,90]).tolist() if len(a) else None

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=Path("outputs/segment07_trial1"))
    p.add_argument("--ground",type=Path,default=Path("outputs/ground07_B4c"))
    p.add_argument("--camera",type=Path,default=Path("outputs/edge07_E1_depth"))
    p.add_argument("--cloud-data",type=Path,default=Path("outputs/lines07_M0_ghost"))
    p.add_argument("--gt-calib",type=Path,help="Separate diagnostic projection only")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--frames",type=int,nargs="+")
    p.add_argument("--radius",type=float,default=12.)
    p.add_argument("--max-depth",type=float,default=40.)
    p.add_argument("--jump-abs",type=float,default=1.)
    p.add_argument("--jump-rel",type=float,default=.1)
    p.add_argument("--camera-jump",type=float,default=.02)
    p.add_argument("--camera-crease",type=float,default=.02)
    p.add_argument("--non-ground",action="store_true",help="Restrict LiDAR foreground/back support to >.25m above archived segment ground; z-buffer unchanged")
    a=p.parse_args(argv)
    vals=[a.radius,a.max_depth,a.jump_abs,a.jump_rel,a.camera_jump,a.camera_crease]
    if not np.isfinite(vals).all() or min(vals)<=0 or a.max_depth<=.5:
        p.error("Invalid extraction settings")
    if a.out.exists() and (not a.out.is_dir() or any(a.out.iterdir())):
        raise FileExistsError("Use a fresh output directory")
    run=read(a.run/"summary.json");cfg=read(a.run/"config.json")
    gd=read(a.ground/"summary.json");cam=read(a.camera/"summary.json")
    manifest=read(a.cloud_data/"manifest.json")
    fit=gd["experiments"]["B4c_w0.1"]
    if not fit["success"] or fit["refit_scales"] or fit["normal_enabled"]:
        raise ValueError("Expected successful fixed-scale offset-only B4-c")
    he=read(a.run/"segmented_handeye.json")["fit"]["T_camera_lidar"]
    if not np.allclose(he,gd["baseline"]["T_camera_lidar"],atol=1e-9,rtol=0):
        raise ValueError("B4-c/HE provenance mismatch")
    X=validate_extrinsic(np.asarray(fit["final"]["T_camera_lidar"],float))
    K=np.asarray(cam["K"],float);camera=int(cam["camera_index"])
    if (K.shape!=(3,3) or not np.isfinite(K).all() or not np.allclose(K[2],[0,0,1])
        or K[0,0]<=0 or K[1,1]<=0 or camera!=int(cfg["camera_index"])
        or camera!=int(manifest["camera_index"]) or not np.allclose(K,manifest["K"])
        or Path(manifest["run"]).name!=a.run.name):
        raise ValueError("Camera/run mismatch")
    if cam["depth_mode"]=="disabled":
        raise ValueError("Require original raw DA2 output, not RGB-only E1")
    poses={"B4c":X}
    if a.gt_calib:
        cal=parse_kitti_odom_calib(a.gt_calib)
        if not np.allclose(cal[f"P{camera}"][:,:3],K):
            raise ValueError("Diagnostic GT intrinsics mismatch")
        poses["GT_oracle"]=kitti_gt_camera_lidar(a.gt_calib,camera)
    clouds={int(r["frame"]):r for r in manifest["frames"]}
    anchors={int(r["anchor_frame"]):int(r["segment_id"]) for r in run["segments"]}
    planes={int(r["segment_id"]):r["lidar"] for r in gd["ground_diagnostics"] if r["accepted"]}
    frames=sorted(set(a.frames or map(int,cam["frames"])))
    if not frames or any(f not in clouds or str(f) not in cam["frames"] for f in frames):
        raise ValueError("Missing E1 or M0 frame")
    if a.non_ground and any(f not in anchors or anchors[f] not in planes for f in frames):
        raise ValueError("Non-ground arm requires accepted anchor ground planes")
    for f in frames:
        for path in (a.camera/f"frame_{f:06d}"/"original.png",
                     a.camera/f"frame_{f:06d}"/"relative_depth.npy",
                     a.cloud_data/f"frame_{f:06d}"/"image.png",
                     a.cloud_data/f"frame_{f:06d}"/"cloud.npz"):
            if not path.is_file():raise FileNotFoundError(path)
    a.out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    summary=dict(warning="Independent edge proposals, no matching, fitting or physical labels. "
                 "Camera jump/crease are relative inverse-depth proxies, not exclusive types. "
                 "LiDAR visibility is approximate; holes unknown; bracket width is not a certified boundary interval. "
                 "Repeated foreground IDs are directional hypotheses, not independent observations. "
                 "GT projection optional diagnostic only; no GT-guided B4c selection.",
                 sources=dict(run=str(a.run),ground=str(a.ground),camera=str(a.camera),
                              cloud=str(a.cloud_data),run_sha256=sha(a.run/"summary.json"),
                              ground_sha256=sha(a.ground/"summary.json"),
                              camera_sha256=sha(a.camera/"summary.json"),
                              manifest_sha256=sha(a.cloud_data/"manifest.json"),
                              depth_mode=cam["depth_mode"]),
                 settings=dict(radius_px=a.radius,max_depth_m=a.max_depth,
                               jump_abs_m=a.jump_abs,jump_rel=a.jump_rel,
                               same_abs_m=.4,same_rel=.05,sectors=8,
                               camera_jump_min=a.camera_jump,camera_crease_min=a.camera_crease,
                               camera_sigma_px=1.2,profile_offsets_px=[3,9],
                               canny=[50,150],non_ground=a.non_ground,ground_margin_m=.25),
                 K=K.tolist(),transforms={k:v.tolist() for k,v in poses.items()},frames={})
    for f in frames:
        src=a.camera/f"frame_{f:06d}";base=a.cloud_data/f"frame_{f:06d}"
        cr=clouds[f];er=cam["frames"][str(f)];cp=base/"cloud.npz";dp=src/"relative_depth.npy"
        if sha(cp)!=cr["cloud_sha256"] or sha(base/"image.png")!=cr["image_sha256"] or sha(dp)!=er["depth_sha256"]:
            raise ValueError(f"Input hash mismatch: {f}")
        im=image(src/"original.png")
        if not np.array_equal(im,image(base/"image.png")) or list(im.shape[:2])!=er["shape"]:
            raise ValueError(f"Image mismatch: {f}")
        D=np.load(dp,allow_pickle=False)
        shape=im.shape[:2]
        if D.shape!=shape:raise ValueError("Depth resolution mismatch")
        with np.load(cp,allow_pickle=False) as z:
            xyz=np.asarray(z["xyz"],float);scans=np.asarray(z["scan_frames"])
        if len(xyz)!=cr["points"] or scans.shape!=(len(xyz),):
            raise ValueError("Cloud point/scan count mismatch")
        fg=np.ones(len(xyz),bool)
        if a.non_ground:
            pl=planes[anchors[f]]
            n=np.asarray(pl["normal"],float);d=float(pl["offset_m"])
            if not np.isclose(np.linalg.norm(n),1.) or n[2]<=0 or d<=0:
                raise ValueError("Invalid archived ground")
            fg=xyz@n+d>.25
        folder=a.out/f"frame_{f:06d}";folder.mkdir()
        rgb=cv2.Canny(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),50,150,L2gradient=True)
        edges=dense_edges(D,K,jump_min=a.camera_jump,crease_min=a.camera_crease)
        lo,hi=np.percentile(D,[5,95]);dvis=color_depth(D,lo,hi)
        save(folder/"original.png",im);save(folder/"rgb_canny.png",rgb)
        save(folder/"da2_depth.png",dvis)
        info=dict(cloud_sha256=sha(cp),depth_sha256=sha(dp),points=len(xyz),
                  rgb_edge_pixels=int(np.count_nonzero(rgb)),camera={},lidar={})
        panels=[panel(im,"Original RGB"),panel(dvis,"DA2 relative inverse depth"),
                panel(cv2.cvtColor(rgb,cv2.COLOR_GRAY2BGR),"RGB Canny (reference)")]
        for name,e in edges.items():
            np.savez_compressed(folder/f"camera_{name}.npz",**e)
            save(folder/f"camera_{name}.png",points_map(shape,e["uv"]))
            save(folder/f"camera_{name}_direction.png",direction_map(shape,e["uv"],e["normal_uv"],axial=name=="crease"))
            ov=overlay(im.copy(),e["uv"],(0,255,0) if name=="jump" else (255,0,255))
            save(folder/f"camera_{name}_overlay.png",ov)
            panels.append(panel(ov,f"DA2 {name} proxy | {len(e['uv'])} pixels"))
            info["camera"][name]=dict(count=len(e["uv"]),score_p10_p50_p90=percentiles(e["score"]))
        for name,T in poses.items():
            q,depth,index=sparse_projection(xyz,T,K,shape,max_depth=a.max_depth)
            e=sparse_edges(q,xyz,K,radius=a.radius,jump_abs=a.jump_abs,jump_rel=a.jump_rel,foreground_mask=fg)
            e["scan_frames"]=scans[e["point_indices"]]
            e["background_scan_frames"]=scans[e["background_indices"]]
            e["support_scan_frames"]=scans[e["support_indices"]]
            np.savez_compressed(folder/f"{name}_sparse_depth.npz",depth_m=depth,index_map=index,valid=np.isfinite(depth))
            np.savez_compressed(folder/f"{name}_edges.npz",**e)
            vis=color_depth(depth,.5,a.max_depth)
            save(folder/f"{name}_sparse_depth.png",vis)
            save(folder/f"{name}_edge_map.png",points_map(shape,e["uv"]))
            # Direction image may show only last hypothesis per pixel; NPZ retains all.
            save(folder/f"{name}_direction.png",direction_map(shape,e["uv"],e["normal_uv"]))
            ov=overlay(im.copy(),e["uv"],(0,165,255))
            save(folder/f"{name}_edge_overlay.png",ov)
            both=overlay(overlay(im.copy(),edges["jump"]["uv"],(0,255,0)),e["uv"],(0,165,255))
            save(folder/f"{name}_jump_overlay.png",both)
            panels.extend([panel(vis,f"{name}: sparse optical Z"),
                           panel(ov,f"{name}: LiDAR foreground edge points"),
                           panel(both,f"{name}: green DA2 / orange LiDAR (not matches)")])
            info["lidar"][name]=dict(visible_pixels=len(q["ids"]),hypotheses=len(e["uv"]),
                 unique_foreground_points=len(np.unique(e["point_indices"])),
                 bracket_width_px_p10_p50_p90=percentiles(e["bracket_width_px"]),
                 depth_jump_m_p10_p50_p90=percentiles(e["depth_jump_m"]))
        save(folder/"comparison.png",np.vstack(panels))
        summary["frames"][str(f)]=info
        write(folder/"summary.json",info)
        print(f"{f:06d}: camera jump={info['camera']['jump']['count']}, "
              f"crease={info['camera']['crease']['count']}, "
              f"B4c LiDAR unique={info['lidar']['B4c']['unique_foreground_points']}",flush=True)
    write(a.out/"summary.json",summary)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
