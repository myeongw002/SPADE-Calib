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
    sample_line,project_line,fit_lines,metrics,information,perturb,residual,
    lidar_bev_xy,polygon_roi_mask)
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


def select_wall_roi(p, image_ids, uv, z, img, wall, previous_mask=None):
    """Linked BEV + projected image ROI editor. No GT or pixel snapping."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    if "agg"==plt.get_backend().lower() or "inline" in plt.get_backend().lower():
        raise RuntimeError("Interactive Matplotlib backend required (e.g. MPLBACKEND=TkAgg)")
    bev=lidar_bev_xy(p)
    fig=plt.figure(figsize=(17,8))
    grid=fig.add_gridspec(2,2,width_ratios=[1,1.6])
    axb=fig.add_subplot(grid[:,0]);axi=fig.add_subplot(grid[0,1]);axr=fig.add_subplot(grid[1,1])
    # Same metric color for each point in both views; matching IDs highlighted.
    dist=np.linalg.norm(p,axis=1);norm=Normalize(0,max(float(np.percentile(dist,99)),1.))
    axb.scatter(bev[:,0],bev[:,1],c=dist,cmap="turbo",norm=norm,s=3,alpha=.8)
    axb.scatter([0],[0],marker="^",c="black",s=85,zorder=5)
    axb.set_aspect("equal");axb.set_xlabel("Right = -LiDAR Y [m]")
    axb.set_ylabel("Forward = LiDAR X [m]");axb.set_title("BEV: vehicle forward UP, right RIGHT")
    axb.axhline(0,c=".5",lw=.6);axb.axvline(0,c=".5",lw=.6)
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    for ax in (axi,axr):
        ax.imshow(rgb);ax.set_xlim(-.5,img.shape[1]-.5);ax.set_ylim(img.shape[0]-.5,-.5)
        ax.set_axis_off()
    # Draw far-to-near; all depth-valid points remain selectable, including
    # potentially occluded points. A second BEV ROI can reject them.
    order=np.argsort(z)[::-1]
    axi.scatter(uv[order,0],uv[order,1],c=dist[image_ids[order]],cmap="turbo",
                norm=norm,s=4,alpha=.75)
    axi.set_title("Image + initial LiDAR projection: draw ROI here or in BEV")
    axr.set_title("Raw image (reference only)")
    if previous_mask is not None:
        b=bev[previous_mask];q=uv[previous_mask[image_ids]]
        axb.scatter(b[:,0],b[:,1],s=9,c="magenta",alpha=.65)
        axi.scatter(q[:,0],q[:,1],s=10,c="magenta",alpha=.65)
    sb=axb.scatter([],[],s=13,facecolors="none",edgecolors="lime",linewidths=.8,zorder=10)
    si=axi.scatter([],[],s=16,facecolors="none",edgecolors="lime",linewidths=.9,zorder=10)
    sr=axr.scatter([],[],s=9,c="lime",alpha=.7,zorder=10)
    artists={key:ax.plot([],[],"w-o",lw=1.5,ms=4,markerfacecolor="lime",
                          markeredgecolor="black",zorder=15)[0]
             for key,ax in (("bev",axb),("image",axi))}
    polygons={"bev":[],"image":[]}
    state={"confirmed":False}
    title=fig.suptitle("")
    axes={axb:"bev",axi:"image"}
    fig.subplots_adjust(left=.07,right=.98,top=.86,bottom=.08,wspace=.17,hspace=.25)

    def refresh():
        complete={k:v for k,v in polygons.items() if len(v)>=3}
        mask=polygon_roi_mask(p,image_ids,uv,complete) if complete else np.zeros(len(p),bool)
        sb.set_offsets(bev[mask]);sel=uv[mask[image_ids]]
        si.set_offsets(sel);sr.set_offsets(sel)
        for key,artist in artists.items():
            v=np.asarray(polygons[key],float).reshape(-1,2)
            if len(v)>=3: v=np.vstack((v,v[0]))
            artist.set_data(v[:,0],v[:,1])
        title.set_text(f"Wall {wall} | BEV vertices {len(polygons['bev'])} | image vertices {len(polygons['image'])} | selected {mask.sum()} points\n"
                       "Left click: vertices (either panel; both = intersection) | Right click: undo in panel | R: reset | Enter: accept | Esc: cancel\n"
                       "Green rings: linked selected points | Magenta: first wall | Pan/zoom toolbar OFF before drawing")
        fig.canvas.draw_idle()
        return mask

    def on_click(event):
        key=axes.get(event.inaxes)
        if key is None or event.xdata is None or event.ydata is None: return
        toolbar=getattr(fig.canvas,"toolbar",None)
        if toolbar is not None and getattr(toolbar,"mode",""): return
        if event.button==1:
            polygons[key].append([float(event.xdata),float(event.ydata)])
        elif event.button==3 and polygons[key]:
            polygons[key].pop()
        else: return
        refresh()

    def on_key(event):
        if event.key in ("enter","return"):
            if not any(len(v)>=3 for v in polygons.values()) or any(0<len(v)<3 for v in polygons.values()):
                print("Draw at least one polygon with >=3 vertices; finish or undo partial polygons.",flush=True)
                return
            state["confirmed"]=True;fig.canvas.stop_event_loop()
        elif event.key=="r":
            for v in polygons.values(): v.clear()
            refresh()
        elif event.key=="escape":
            fig.canvas.stop_event_loop()

    cids=[fig.canvas.mpl_connect("button_press_event",on_click),
          fig.canvas.mpl_connect("key_press_event",on_key),
          fig.canvas.mpl_connect("close_event",lambda event:fig.canvas.stop_event_loop())]
    refresh()
    plt.show(block=False)
    try: fig.canvas.start_event_loop(0)
    finally:
        for cid in cids: fig.canvas.mpl_disconnect(cid)
        plt.close(fig)
    if not state["confirmed"]: raise ValueError("Wall annotation cancelled")
    chosen={k:v for k,v in polygons.items() if len(v)>=3}
    return polygon_roi_mask(p,image_ids,uv,chosen),chosen


def annotate(a):
    # ROI display coordinates only; plane fitting always uses original LiDAR XYZ.
    import matplotlib.pyplot as plt
    manifest,X,K,digest=load_data(a.data)
    xyz,scans,img=frame_data(a.data,manifest,a.frame)
    if a.z_min>=a.z_max or a.near<=0 or not 0<a.depth_min<a.depth_max:
        raise ValueError("Invalid annotation ROI settings")
    visible=(xyz[:,2]>=a.z_min)&(xyz[:,2]<=a.z_max)
    p=xyz[visible];source_scans=scans[visible]
    if len(p)<60: raise ValueError("Too few points after height filter")
    image_ids,uv,z=project_xyz(p,X,K,img.shape[1],img.shape[0],
                             min_depth_m=a.depth_min,max_depth_m=a.depth_max)
    if not len(image_ids):
        raise ValueError("No projected points: inspect initializer, frame and camera index")
    selected=[];walls=[];polygons=[];roi_selections=[];previous=None
    for i in range(2):
        inside,roi_views=select_wall_roi(p,image_ids,uv,z,img,i+1,previous)
        roi=p[inside]
        n,d,keep,report=fit_wall(roi,seed=a.seed+i,threshold=a.plane_threshold,min_inliers=a.min_inliers)
        q=roi[keep];report["scan_count"]=len(np.unique(source_scans[inside][keep]))
        previous=np.zeros(len(p),bool);previous[np.flatnonzero(inside)[keep]]=True
        print(f"Wall {i+1}: {report}",flush=True)
        selected.append(q);walls.append((n,d,report))
        # Preserve original LiDAR XY polygon field for BEV selections; image
        # selections have no equivalent planar LiDAR XY polygon.
        v=np.asarray(roi_views["bev"],float) if "bev" in roi_views else None
        polygons.append(np.column_stack((v[:,1],-v[:,0])).tolist() if v is not None else None)
        roi_selections.append(dict(polygons=roi_views,roi_points=int(inside.sum()),
                                   bev_convention="horizontal=-Y, vertical=X"))
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
        toolbar=getattr(fig.canvas,"toolbar",None)
        if toolbar is not None and getattr(toolbar,"mode",""): return
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
             uv_image=uv.tolist(),geometry=geometry,polygons_xy=polygons,roi_selections=roi_selections,
             settings=dict(z_min=a.z_min,z_max=a.z_max,plane_threshold=a.plane_threshold,
                           min_inliers=a.min_inliers,near=a.near,min_length=a.min_length,seed=a.seed,
                           depth_min=a.depth_min,depth_max=a.depth_max),
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
    q.add_argument("--depth-min",type=float,default=.5);q.add_argument("--depth-max",type=float,default=65.,help="Optical depth gate for image ROI projection only")
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
