#!/usr/bin/env python3
"""M0 manual physical 2D–3D line correspondences, no GT-guided matching."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from run_segment_ground import write_json
from spade_calib.io.kitti import accumulate_kitti_rigid_window
from spade_calib.evaluation.kitti import parse_kitti_odom_calib,kitti_gt_camera_lidar,evaluate_camera_lidar
from spade_calib.evaluation.segment_projection import project_xyz,render_overlay,_numbered_file
from spade_calib.selection.scale_segmentation import validate_extrinsic
from spade_calib.calibration.crossmodal_lines import (fit_wall,finite_intersection,image_line,
    sample_line,project_line,fit_lines,metrics,information,perturb,residual)
from spade_calib.calibration.scene_consistency import voxel_return_indices


def fresh(path):
    if path.exists() and any(path.iterdir()): raise FileExistsError(f"Use fresh output: {path}")
    path.mkdir(parents=True,exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_image(path,image):
    if not cv2.imwrite(str(path),image): raise OSError(f"Cannot save {path}")


def prepare(a):
    cfg=json.loads((a.run/"config.json").read_text())
    archive=json.loads((a.run/"summary.json").read_text())
    X=validate_extrinsic(np.array(json.loads((a.run/"segmented_handeye.json").read_text())["fit"]["T_camera_lidar"]))
    if a.initial_summary:
        doc=json.loads(a.initial_summary.read_text())
        arm=doc["experiments"][a.initial_arm]
        if not arm["success"]: raise ValueError("Initial arm was not successful")
        if not np.allclose(doc["baseline"]["T_camera_lidar"],X,atol=1e-9):
            raise ValueError("Initial summary does not match archived HE")
        X=validate_extrinsic(np.asarray(arm["final"]["T_camera_lidar"]))
    camera=int(cfg["camera_index"])
    K=parse_kitti_odom_calib(a.intrinsics_calib,require_tr=False)[f"P{camera}"][:,:3]
    if a.window<1 or a.window%2==0 or a.voxel<=0 or a.max_range<=1:
        raise ValueError("Invalid window/voxel/range")
    times=Path(cfg["sequence"])/"times.txt"
    first=int(cfg["glim_first_frame"]);last=first+len(np.loadtxt(cfg["glim"],ndmin=2))-1
    frames=sorted(set(a.frames or [int(r["anchor_frame"]) for r in archive["segments"]]))
    fresh(a.out)
    manifest=dict(run=str(a.run),camera_index=camera,K=K.tolist(),T_camera_lidar=X.tolist(),
                  initial_summary=str(a.initial_summary) if a.initial_summary else None,
                  initial_arm=a.initial_arm if a.initial_summary else "HE",
                  window=a.window,voxel_m=a.voxel,frames=[],
                  warning="GLIM rigid local accumulation, no deskew; initial overlay not a label, GT not loaded.")
    for f in frames:
        if not first<=f<=last: raise ValueError(f"Frame outside GLIM range: {f}")
        ids=list(range(max(first,f-a.window//2),min(last,f+a.window//2)+1))
        c=accumulate_kitti_rigid_window(cfg["lidar_dir"],cfg["glim"],times,ids,
            target_frame_id=f,glim_first_frame=first,max_range_m=a.max_range)
        idx=voxel_return_indices(c.xyz,a.voxel)
        xyz=c.xyz[idx];scans=np.asarray(ids)[c.scan_id[idx]]
        folder=a.out/f"frame_{f:06d}";folder.mkdir()
        np.savez_compressed(folder/"cloud.npz",xyz=xyz,scan_frames=scans)
        path=_numbered_file(a.image_dir,f,(".png",".jpg",".jpeg"))
        img=cv2.imread(str(path))
        if img is None: raise ValueError(f"Cannot read {path}")
        save_image(folder/"image.png",img)
        _,uv,z=project_xyz(xyz,X,K,img.shape[1],img.shape[0],max_depth_m=a.max_range)
        overlay,_=render_overlay(img,uv,z,a.max_range)
        save_image(folder/"initial_overlay.png",overlay)
        manifest["frames"].append(dict(frame=f,source_frames=ids,points=len(xyz),
            image_shape=list(img.shape[:2]),cloud_sha256=sha(folder/"cloud.npz"),
            image_sha256=sha(folder/"image.png")))
        print(f"Frame {f:06d}: {len(xyz)} points from {len(ids)} scans",flush=True)
    write_json(a.out/"manifest.json",manifest)


def load_data(folder):
    path=folder/"manifest.json"
    manifest=json.loads(path.read_text());K=np.asarray(manifest["K"],float)
    X=validate_extrinsic(np.asarray(manifest["T_camera_lidar"],float))
    return manifest,X,K,sha(path)


def frame_data(folder,manifest,f):
    row=next((r for r in manifest["frames"] if r["frame"]==f),None)
    if row is None: raise ValueError(f"Frame {f} not prepared")
    base=folder/f"frame_{f:06d}"
    for name,key in (("cloud.npz","cloud_sha256"),("image.png","image_sha256")):
        if sha(base/name)!=row[key]: raise ValueError(f"Prepared file changed: {base/name}")
    with np.load(base/"cloud.npz",allow_pickle=False) as z:
        xyz=z["xyz"];scans=z["scan_frames"]
    img=cv2.imread(str(base/"image.png"))
    if img is None: raise ValueError("Prepared image unreadable")
    return xyz,scans,img


def draw(img,xyz,uv,X,K):
    out=img.copy()
    a=np.rint(uv).astype(int);cv2.line(out,tuple(a[0]),tuple(a[1]),(0,255,0),2)
    p,z=project_line(xyz,X,K)
    if np.all(z>.5) and np.max(abs(p))<100000:
        p=np.rint(p).astype(int);cv2.line(out,tuple(p[0]),tuple(p[1]),(0,0,255),2)
    return out


def annotate(a):
    # Matplotlib interactive backend required; never change raw image labels
    # by snapping to a projected LiDAR line.
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as Polygon
    manifest,X,K,digest=load_data(a.data)
    xyz,scans,img=frame_data(a.data,manifest,a.frame)
    if a.z_min>=a.z_max or a.near<=0: raise ValueError("Invalid annotation ROI settings")
    # Use height-filtered LiDAR-only top view for both wall selections.
    visible=(xyz[:,2]>=a.z_min)&(xyz[:,2]<=a.z_max)
    p=xyz[visible]
    if len(p)<60: raise ValueError("Too few points after height filter")
    selected=[];walls=[];polygons=[]
    for i in range(2):
        fig,ax=plt.subplots(figsize=(10,8))
        sc=ax.scatter(p[:,0],p[:,1],c=p[:,2],s=2,cmap="viridis")
        ax.set_aspect("equal");ax.set_xlabel("LiDAR forward X [m]");ax.set_ylabel("LiDAR left Y [m]")
        ax.set_title(f"Wall {i+1}: polygon vertices, Enter to finish; zoom before selecting")
        if selected:
            q=selected[0];ax.scatter(q[:,0],q[:,1],s=3,c="red")
        fig.colorbar(sc,ax=ax,label="LiDAR Z [m]")
        plt.show(block=False)
        vertices=np.asarray(plt.ginput(n=-1,timeout=0,show_clicks=True,
                                       mouse_add=1,mouse_pop=3,mouse_stop=2))
        plt.close(fig)
        if len(vertices)<3: raise ValueError("Need at least 3 polygon vertices")
        inside=visible.copy();inside[visible]=Polygon(np.vstack((vertices,vertices[0]))).contains_points(p[:,:2])
        roi=xyz[inside]
        n,d,keep,report=fit_wall(roi,seed=a.seed+i,threshold=a.plane_threshold,min_inliers=a.min_inliers)
        q=roi[keep];report["scan_count"]=len(np.unique(scans[inside][keep]))
        print(f"Wall {i+1}: {report}",flush=True)
        selected.append(q);walls.append((n,d,report));polygons.append(vertices.tolist())
    endpoints,geometry=finite_intersection(walls[0][0],walls[0][1],selected[0],
        walls[1][0],walls[1][1],selected[1],near=a.near,min_length=a.min_length)
    # Show approximate initial projection on a SEPARATE panel, label raw image.
    fig,axes=plt.subplots(2,1,figsize=(15,8))
    preview=img.copy();pr,z=project_line(endpoints,X,K)
    if np.any(z<.5) or np.max(abs(pr))>100000:
        plt.close(fig);raise ValueError("Candidate not in front of camera at initializer")
    xy=np.rint(pr).astype(int);cv2.line(preview,tuple(xy[0]),tuple(xy[1]),(0,0,255),2)
    for ax,im in zip(axes,(img,preview)):
        ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_axis_off()
    axes[0].set_title("CLICK TWO POINTS ON REAL EDGE HERE (raw image)")
    axes[1].set_title("Initial projection: identity hint only, NOT a pixel label")
    plt.tight_layout();plt.show(block=False)
    # ginput returns data coordinates from whichever axes are clicked;
    # use explicit event handler to reject all clicks outside raw panel.
    clicks=[]
    def click(event):
        if event.inaxes is axes[0] and event.button==1 and event.xdata is not None:
            clicks.append([event.xdata,event.ydata])
            axes[0].plot(event.xdata,event.ydata,"g+");fig.canvas.draw_idle()
            if len(clicks)==2: fig.canvas.stop_event_loop()
    cid=fig.canvas.mpl_connect("button_press_event",click)
    closeid=fig.canvas.mpl_connect("close_event",lambda event:fig.canvas.stop_event_loop())
    fig.canvas.start_event_loop(0)
    fig.canvas.mpl_disconnect(cid);fig.canvas.mpl_disconnect(closeid);plt.close(fig)
    if len(clicks)!=2: raise ValueError("Annotation cancelled")
    uv=np.asarray(clicks,float);image_line(uv)
    if np.any(uv<0) or np.any(uv>=np.array([img.shape[1],img.shape[0]])):
        raise ValueError("Image annotation outside image")
    fig,ax=plt.subplots(figsize=(14,5))
    ax.imshow(cv2.cvtColor(draw(img,endpoints,uv,X,K),cv2.COLOR_BGR2RGB))
    ax.set_title("Green: manual image edge; red: initial LiDAR line. Close to confirm in terminal.")
    ax.set_axis_off();plt.show(block=True);plt.close(fig)
    if input("Same physical static corner, visible in both sensors? Save [y/N]: ").strip().lower()!="y":
        print("Discarded");return
    path=a.data/"annotations.json"
    doc=json.loads(path.read_text()) if path.exists() else dict(manifest_sha256=digest,records=[])
    if doc["manifest_sha256"]!=digest: raise ValueError("Annotation manifest mismatch")
    if any(r["frame"]==a.frame and r["split"]!=a.split for r in doc["records"]):
        raise ValueError("Keep all annotations from a frame in one split")
    if any(r["feature"]==a.feature and r["split"]!=a.split for r in doc["records"]):
        raise ValueError("Keep same physical feature in one split")
    if any(r["frame"]==a.frame and r["feature"]==a.feature for r in doc["records"]):
        raise ValueError("Feature already annotated at this frame")
    rid=f"line_{len(doc['records']):04d}"
    rec=dict(id=rid,frame=a.frame,feature=a.feature,split=a.split,xyz_lidar=endpoints.tolist(),
             uv_image=uv.tolist(),geometry=geometry,polygons_xy=polygons,
             settings=dict(z_min=a.z_min,z_max=a.z_max,plane_threshold=a.plane_threshold,
                           min_inliers=a.min_inliers,near=a.near,min_length=a.min_length,seed=a.seed),
             planes=[dict(normal=n.tolist(),offset_m=d,**r) for n,d,r in walls],
             manual_identity_confirmed=True)
    support=a.data/"line_support";support.mkdir(exist_ok=True)
    np.savez_compressed(support/f"{rid}.npz",plane1=selected[0],plane2=selected[1])
    save_image(support/f"{rid}.png",draw(img,endpoints,uv,X,K))
    doc["records"].append(rec)
    tmp=path.with_suffix(".tmp");write_json(tmp,doc);tmp.replace(path)
    print(f"Saved {rid}, total={len(doc['records'])}")


def fit(a):
    manifest,X,K,digest=load_data(a.data)
    doc=json.loads((a.data/"annotations.json").read_text())
    if doc["manifest_sha256"]!=digest: raise ValueError("Annotation manifest mismatch")
    records=doc["records"]
    ids=set();splits={};features={}
    for r in records:
        if r["id"] in ids or not r["manual_identity_confirmed"] or r["split"] not in ("train","test"):
            raise ValueError("Invalid manual correspondence record")
        ids.add(r["id"])
        for store,key in ((splits,r["frame"]),(features,r["feature"])):
            if key in store and store[key]!=r["split"]: raise ValueError("Train/test leakage")
            store[key]=r["split"]
        sample_line(r["xyz_lidar"]);image_line(r["uv_image"])
    train=[r for r in records if r["split"]=="train"];test=[r for r in records if r["split"]=="test"]
    if len(train)<3 or len(set(r["frame"] for r in train))<2:
        raise ValueError("Need >=3 training correspondences across >=2 frames; aim for 10–20 diverse lines")
    for f in splits: frame_data(a.data,manifest,f)
    if any(np.any(project_line(sample_line(r["xyz_lidar"]),X,K)[1]<.5) for r in records):
        raise ValueError("A fixed line is behind/too near camera at initialization")
    fresh(a.out)
    gt=kitti_gt_camera_lidar(a.gt_calib,manifest["camera_index"]) if a.gt_calib else None
    report=dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                manifest_sha256=digest,annotation_sha256=sha(a.data/"annotations.json"),
                train_ids=[r["id"] for r in train],test_ids=[r["id"] for r in test],
                initial=dict(T_camera_lidar=X.tolist(),train=metrics(X,train,K),
                             test=metrics(X,test,K),information=information(X,train,K,a.sigma_px)),
                experiments={},warning="Manual physical matches; infinite image-line distance, finite support/overlap diagnostic; not automatic calibration. Fixed GLIM, no HE/ground refit.")
    if gt is not None:
        report["initial"]["gt_evaluation"]=evaluate_camera_lidar(X,gt)
        report["GT_diagnostic"]=dict(train=metrics(gt,train,K),test=metrics(gt,test,K),
                                      information=information(gt,train,K,a.sigma_px))
        # GT-centered X-axis scan is evaluation only, no selected initializer.
        report["GT_diagnostic"]["x_sweep"]=[dict(delta_m=float(dx),**metrics(
            perturb(gt,[0,0,0,dx,0,0]),train,K)) for dx in np.linspace(-.5,.5,21)]
    rng=np.random.default_rng(a.seed)
    for mode in ("x_only","translation","full"):
        for i in range(a.restarts+1):
            start=None if i==0 else rng.uniform(-.5,.5,6)*np.r_[np.full(3,np.deg2rad(a.max_rotation_deg)),np.full(3,a.max_translation_m)]
            Y,r=fit_lines(train,X,K,mode=mode,sigma_px=a.sigma_px,max_rotation_deg=a.max_rotation_deg,
                         max_translation_m=a.max_translation_m,start=start)
            name=f"{mode}_{i}";r["T_camera_lidar"]=Y.tolist();r["test"]=metrics(Y,test,K)
            r["start_delta"]=start.tolist() if start is not None else [0.]*6
            if gt is not None: r["gt_evaluation"]=evaluate_camera_lidar(Y,gt)
            report["experiments"][name]=r
            write_json(a.out/"summary.json",report)
            print(name,"cost",r["cost"],"rank",r["information"]["effective_rank"],"success",r["success"],flush=True)
    # Select restart by training loss only, not GT/test.
    report["best_by_train_cost"]={}
    for mode in ("x_only","translation","full"):
        valid=[k for k,r in report["experiments"].items() if k.startswith(mode+"_") and r["success"]]
        if valid: report["best_by_train_cost"][mode]=min(valid,key=lambda k:report["experiments"][k]["cost"])
    for mode,name in report["best_by_train_cost"].items():
        Y=np.asarray(report["experiments"][name]["T_camera_lidar"])
        for r in records:
            img=cv2.imread(str(a.data/f"frame_{r['frame']:06d}"/"image.png"))
            before=draw(img,np.asarray(r["xyz_lidar"]),np.asarray(r["uv_image"]),X,K)
            after=draw(img,np.asarray(r["xyz_lidar"]),np.asarray(r["uv_image"]),Y,K)
            save_image(a.out/f"{mode}_{r['id']}.png",np.vstack((before,after)))
    write_json(a.out/"summary.json",report)


def main():
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest="command",required=True)
    q=sp.add_parser("prepare")
    q.add_argument("--run",type=Path,required=True);q.add_argument("--out",type=Path,required=True)
    q.add_argument("--image-dir",type=Path,required=True);q.add_argument("--intrinsics-calib",type=Path,required=True)
    q.add_argument("--initial-summary",type=Path);q.add_argument("--initial-arm",default="H2_joint_scene")
    q.add_argument("--frames",nargs="+",type=int);q.add_argument("--window",type=int,default=21)
    q.add_argument("--voxel",type=float,default=.05);q.add_argument("--max-range",type=float,default=40.)
    q.set_defaults(func=prepare)
    q=sp.add_parser("annotate");q.add_argument("--data",type=Path,required=True)
    q.add_argument("--frame",type=int,required=True);q.add_argument("--feature",required=True)
    q.add_argument("--split",choices=["train","test"],default="train")
    q.add_argument("--z-min",type=float,default=-1.);q.add_argument("--z-max",type=float,default=4.5)
    q.add_argument("--plane-threshold",type=float,default=.08);q.add_argument("--min-inliers",type=int,default=30)
    q.add_argument("--near",type=float,default=.7);q.add_argument("--min-length",type=float,default=.75)
    q.add_argument("--seed",type=int,default=17);q.set_defaults(func=annotate)
    q=sp.add_parser("fit");q.add_argument("--data",type=Path,required=True);q.add_argument("--out",type=Path,required=True)
    q.add_argument("--gt-calib",type=Path);q.add_argument("--sigma-px",type=float,default=2.)
    q.add_argument("--max-rotation-deg",type=float,default=5.);q.add_argument("--max-translation-m",type=float,default=1.)
    q.add_argument("--restarts",type=int,default=3);q.add_argument("--seed",type=int,default=17)
    q.set_defaults(func=fit)
    a=p.parse_args()
    if getattr(a,"restarts",0)<0: p.error("restarts must be nonnegative")
    a.func(a)


if __name__=="__main__":
    main()
