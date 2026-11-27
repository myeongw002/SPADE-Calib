#!/usr/bin/env python3
"""FV1: prepare blinded RGB boundary traces and raw64 LiDAR candidate audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_edge_points import read,write,sha,image,save
from spade_calib.io.kitti_bin import load_kitti_bin_scan
from spade_calib.evaluation.kitti import parse_kitti_odom_calib,kitti_gt_camera_lidar
from spade_calib.calibration.edge_points import sparse_projection,sparse_edges
from spade_calib.calibration.feature_reference import select_tiles,angular_coordinates,local_evidence,blinded_samples

ARMS=["MV1_target","MV1_all","MV1_A","MV1_C_HV","Surface_HV"]


def range_plot(path,local,c,az_radius,el_radius):
    fig,axs=plt.subplots(1,3,figsize=(11,3.8),layout="constrained")
    sc=axs[0].scatter(local[:,0],local[:,1],c=local[:,2],s=13,cmap="viridis",vmin=0,vmax=40)
    axs[0].scatter([0],[0],marker="+",c="magenta",s=110,linewidths=2)
    axs[0].set(xlim=(-az_radius,az_radius),ylim=(-el_radius,el_radius),
               xlabel="Relative azimuth (deg)",ylabel="Relative elevation (deg)",title="Raw64 observed rays")
    fig.colorbar(sc,ax=axs[0],label="Range (m)")
    # Both plots show all nearby returns, with the orthogonal angle as color.
    for ax,axis,other,label,radius in ((axs[1],0,1,"azimuth",az_radius),(axs[2],1,0,"elevation",el_radius)):
        d=local[:,other];sc=ax.scatter(local[:,axis],local[:,2],c=d,s=14,cmap="coolwarm",
                                        vmin=-max(az_radius,el_radius),vmax=max(az_radius,el_radius))
        ax.scatter([0],[c[2]],marker="+",c="magenta",s=110,linewidths=2)
        ax.set(xlim=(-radius,radius),xlabel=f"Relative {label} (deg)",ylabel="Observed range (m)",
               title="All nearby rays (color: other angle)")
        fig.colorbar(sc,ax=ax,label="Other angle (deg)")
    fig.suptitle(f"Candidate range {c[2]:.2f} m | raw64 observed samples {len(local)} | magenta = candidate")
    fig.savefig(path,dpi=135);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lidar64-dir",type=Path,required=True,help="Original KITTI odometry velodyne directory, NOT synthetic16")
    p.add_argument("--gt-calib",type=Path,required=True)
    p.add_argument("--camera-index",type=int,default=2)
    p.add_argument("--mv8",type=Path,default=Path("outputs/edgepoints07_MV8"))
    p.add_argument("--ep0",type=Path,default=Path("outputs/edgepoints07_EP0"))
    p.add_argument("--synthetic-dir",type=Path,help="Override MV8 original synthetic16 input")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--frames",nargs="+",type=int)
    p.add_argument("--arms",nargs="+",default=ARMS)
    p.add_argument("--tiles-per-stratum",type=int,default=2)
    p.add_argument("--samples-per-arm",type=int,default=3)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--angular-radius",type=float,default=2.)
    a=p.parse_args(argv)
    if (a.seed<0 or min(a.tiles_per_stratum,a.samples_per_arm)<1
        or not np.isfinite(a.angular_radius) or not 0<a.angular_radius<20
        or len(set(a.arms))!=len(a.arms)):
        p.error("Invalid sampling settings")
    if a.out.exists() and (not a.out.is_dir() or any(a.out.iterdir())):
        raise FileExistsError("Use fresh output directory")
    mv=read(a.mv8/"summary.json");ep=read(a.ep0/"summary.json")
    if mv["sources"]["ep0_summary_sha256"]!=sha(a.ep0/"summary.json"):
        raise ValueError("MV8/EP0 provenance mismatch")
    K=np.asarray(ep["K"],float);X=kitti_gt_camera_lidar(a.gt_calib,a.camera_index)
    cal=parse_kitti_odom_calib(a.gt_calib)
    if (not np.allclose(K,cal[f"P{a.camera_index}"][:,:3]) or not np.allclose(K,mv["K"])
        or not np.allclose(X,ep["transforms"]["GT_oracle"],atol=1e-8,rtol=0)):
        raise ValueError("Reference camera/calibration mismatch")
    synthetic=a.synthetic_dir or Path(mv["sources"]["lidar_dir"])
    if synthetic.resolve()==a.lidar64_dir.resolve():raise ValueError("Need original64, not synthetic16")
    frames=sorted(set(a.frames or map(int,mv["frames"])))
    if not frames or any(str(f) not in mv["frames"] or str(f) not in ep["frames"] for f in frames):
        raise ValueError("Missing MV8/EP0 frame")
    a.out.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    manifest=dict(warning="Proposals, not ground truth. Original64 shares measurements with synthetic16; "
                  "GT projection and sparse visibility approximate; no DA2/ELSED shown in manual reference. "
                  "No automatic precision or recall before adjudication. Lidar audit is anchor-view, "
                  "common-camera-FOV population only, not source-view validity. Uncertain is not negative.",
                  sources=dict(mv8=str(a.mv8),ep0=str(a.ep0),original64=str(a.lidar64_dir),
                               synthetic16=str(synthetic),calib=str(a.gt_calib),
                               mv8_sha256=sha(a.mv8/"summary.json"),ep0_sha256=sha(a.ep0/"summary.json"),
                               calib_sha256=sha(a.gt_calib)),
                  settings=dict(seed=a.seed,tiles_per_stratum=a.tiles_per_stratum,
                                samples_per_arm=a.samples_per_arm,angular_radius_deg=a.angular_radius,
                                camera_index=a.camera_index,arms=a.arms,range_reference_m=100.,
                                camera_reference_max_depth_m=80.,audit_max_depth_m=40.,
                                tile_width=224,tile_height=124),
                  K=K.tolist(),T_camera_lidar=X.tolist(),frames={},camera_tasks=[],lidar_tasks=[])
    public=[]
    for f in frames:
        folder=a.out/f"frame_{f:06d}";folder.mkdir()
        rawpath=a.lidar64_dir/f"{f:06d}.bin";synpath=synthetic/f"{f:06d}.bin"
        raw=load_kitti_bin_scan(rawpath,scan_id=0,timestamp_ns=0,max_range_m=100.)
        syn=load_kitti_bin_scan(synpath,scan_id=0,timestamp_ns=0,max_range_m=100.)
        if len(raw.xyz)<1.5*len(syn.xyz) or not len(syn.xyz):
            raise ValueError(f"Original scan not substantially denser: {f}")
        dist=cKDTree(raw.xyz).query(syn.xyz)[0]
        overlap=float((dist<=.001).mean())
        if overlap<.9:raise ValueError(f"Synthetic16 not a matching subset of original64: {f}, {overlap:.3f}")
        base=a.ep0/f"frame_{f:06d}";src=a.mv8/f"frame_{f:06d}"
        im=image(base/"original.png");h,w=im.shape[:2]
        proj,depth,index=sparse_projection(raw.xyz,X,K,(h,w),max_depth=80.)
        # Existing sparse-depth method is used ONLY to stratify tile selection.
        # It is not a reference label or a criterion for accepting a trace.
        e=sparse_edges(proj,raw.xyz,K)
        np.savez_compressed(folder/"raw64_reference.npz",xyz_lidar=raw.xyz,
            source_return_index=raw.source_return_index,uv=proj["uv"],z_m=proj["z"],
            projected_indices=proj["ids"],depth_m=depth,index_map=index,
            **{f"proposal_{k}":v for k,v in e.items()})
        ref=im.copy();colors=cv2.applyColorMap(np.clip(np.rint(255*proj["z"]/40),0,255).astype(np.uint8).reshape(-1,1),cv2.COLORMAP_TURBO).reshape(-1,3)
        for xy,col in zip(np.floor(proj["uv"]).astype(int),colors):
            cv2.circle(ref,tuple(xy),1,tuple(map(int,col)),-1)
        save(folder/"rgb.png",im);save(folder/"raw64_depth_overlay.png",ref)
        tilelist=select_tiles((h,w),e["uv"],per_stratum=a.tiles_per_stratum,seed=a.seed+f)
        for j,t in enumerate(tilelist):
            ident=f"C{f:06d}_{j:02d}";x,y,x1,y1=t["box"]
            rgb=folder/f"{ident}_rgb.png";dep=folder/f"{ident}_depth.png"
            save(rgb,im[y:y1,x:x1]);save(dep,ref[y:y1,x:x1])
            manifest["camera_tasks"].append(dict(id=ident,frame=f,**t))
            public.append(dict(id=ident,kind="camera",rgb=rgb.relative_to(a.out).as_posix(),
                               depth=dep.relative_to(a.out).as_posix(),width=x1-x,height=y1-y))
        with np.load(src/"source_features.npz",allow_pickle=False) as z:
            xyz=np.asarray(z["xyz_lidar"],float);scan_frames=z["scan_frames"];source_return=z["source_return_index"]
        q=xyz@X[:3,:3].T+X[:3,3];z=q[:,2]
        uv=np.full((len(q),2),np.nan);positive=z>.5
        uv[positive]=(q[positive]@K.T)[:,:2]/z[positive,None]
        eligible=(z>.5)&(z<=40)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        populations={}
        for arm in a.arms:
            with np.load(src/f"{arm}_source_ids.npz",allow_pickle=False) as v:ids=v["feature_indices"]
            if (ids.ndim!=1 or ids.dtype.kind not in "iu" or np.any((ids<0)|(ids>=len(xyz)))
                or len(ids)!=mv["frames"][str(f)]["source_candidates"][arm]):
                raise ValueError(f"Bad candidate IDs: {f} {arm}")
            populations[arm]=ids[eligible[ids]]
        selected,sampling=blinded_samples(populations,per_arm=a.samples_per_arm,seed=a.seed+100000+f)
        angular=angular_coordinates(raw.xyz);ca=angular_coordinates(xyz)
        for j,(i,arm_draws) in enumerate(selected):
            ident=f"L{f:06d}_{j:03d}";c=ca[i]
            rid,loc=local_evidence(angular,c,az_radius=a.angular_radius,el_radius=a.angular_radius)
            path=folder/f"{ident}.png";range_plot(path,loc,c,a.angular_radius,a.angular_radius)
            np.savez_compressed(folder/f"{ident}_evidence.npz",candidate_xyz=xyz[i],candidate_angular=c,
                                raw64_xyz=raw.xyz[rid],raw64_indices=raw.source_return_index[rid],
                                relative_angles_range=loc)
            manifest["lidar_tasks"].append(dict(id=ident,frame=f,feature_index=i,source_frame=int(scan_frames[i]),
                source_return_index=int(source_return[i]),sampled_by=arm_draws,
                memberships=[arm for arm,ids in populations.items() if i in ids],
                candidate_xyz=xyz[i].tolist(),gt_uv=uv[i].tolist(),raw64_neighbors=len(rid)))
            public.append(dict(id=ident,kind="lidar",plot=path.relative_to(a.out).as_posix()))
        manifest["frames"][str(f)]=dict(original_points=len(raw.xyz),synthetic_points=len(syn.xyz),
            exact_subset_fraction_1mm=overlap,raw64_sha256=sha(rawpath),synthetic_sha256=sha(synpath),
            projected_points=len(proj["uv"]),raw64_jump_proposals=len(e["uv"]),tile_count=len(tilelist),
            lidar_review_count=len(selected),sampling=sampling)
        print(f"{f:06d}: original64={len(raw.xyz)} synthetic16={len(syn.xyz)} tiles={len(tilelist)} lidar audit={len(selected)}",flush=True)
    # Shuffle task order independently of arm membership. Camera and LiDAR
    # separate groups; HTML does not contain memberships or sampling strata.
    rng=np.random.default_rng(a.seed+999)
    tasks=[]
    for kind in ("camera","lidar"):
        group=[t for t in public if t["kind"]==kind]
        tasks.extend([group[i] for i in rng.permutation(len(group))])
    write(a.out/"manifest.json",manifest)
    template=Path(__file__).with_name("fv1_review.html").read_text(encoding="utf-8")
    bundle=dict(reference_id=sha(a.out/"manifest.json"),tasks=tasks)
    payload=json.dumps(bundle,ensure_ascii=True).replace("<",r"\u003c")
    (a.out/"review.html").write_text(template.replace("__FV1_DATA__",payload),encoding="utf-8")
    write(a.out/"summary.json",dict(reference_id=bundle["reference_id"],frames=manifest["frames"],
          camera_tasks=len(manifest["camera_tasks"]),lidar_tasks=len(manifest["lidar_tasks"]),
          status="unreviewed",note="No physical ground truth or metrics before manual review."))
    return 0


if __name__=="__main__":raise SystemExit(main())
