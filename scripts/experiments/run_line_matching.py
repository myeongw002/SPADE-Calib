#!/usr/bin/env python3
"""S2 matching pilot: frozen M0 lines and archived ELSED, then human review."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from spade_calib.calibration.line_matching import transform_line,clip_projected_line,rank_candidates,reject_conflicts
from spade_calib.lidar.trajectory import GlimTrajectory
from spade_calib.io.kitti import read_kitti_times_ns
from spade_calib.selection.scale_segmentation import validate_extrinsic


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,obj):
    p.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")


def read(p):
    return json.loads(p.read_text())


def draw_line(img,uv,color,width=2):
    p=np.rint(np.asarray(uv)).astype(int)
    cv2.line(img,tuple(p[0]),tuple(p[1]),color,width,cv2.LINE_AA)


def archived_views(reference, run, manifest, X):
    """Use exact H2 view IDs, never approximate frame offsets."""
    prior=read(reference/"summary.json")
    archive=read(run/"summary.json")
    name=manifest["initial_arm"]
    if (name not in ("H2_joint_scene","H2_joint_scene_w5")
            or not prior["experiments"][name]["success"]
            or not np.allclose(prior["experiments"][name]["final"]["T_camera_lidar"],X,atol=1e-9)
            or Path(prior["settings"]["run"]).name!=run.name
            or prior["settings"]["window"]!=5 or prior["settings"]["scene_kind"]!="point"):
        raise ValueError("BA reference does not match M0 initializer")
    views={};hashes={}
    for r in archive["segments"]:
        k=int(r["segment_id"]);anchor=int(r["anchor_frame"])
        p=reference/name/f"segment_{k:02d}.npz"
        with np.load(p,allow_pickle=False) as z:
            f=z["frames"];R=z["camera_R_anchor"];t=z["camera_t_anchor_ref_metric"]
        if (f.ndim!=1 or f.dtype.kind not in "iu" or np.any(np.diff(f)<=0)
                or anchor not in f or R.shape!=(len(f),3,3) or t.shape!=(len(f),3)
                or not np.isfinite(R).all() or not np.isfinite(t).all()):
            raise ValueError(f"Invalid BA frame archive: {k}")
        j=int(np.flatnonzero(f==anchor)[0])
        if not np.allclose(R[j],np.eye(3),atol=1e-7) or not np.allclose(t[j],0,atol=1e-7):
            raise ValueError("Invalid BA anchor gauge")
        if anchor in views:raise ValueError("Duplicate BA anchor")
        views[anchor]=[int(i) for i in f]
        hashes[str(k)]=sha(p)
    return views,dict(arm=name,summary_sha256=sha(reference/"summary.json"),segment_sha256=hashes)


def prepare(a):
    m=a.data/"manifest.json";ap=a.data/"annotations.json";ep=a.edge_data/"summary.json"
    manifest=read(m);ann=read(ap);edge=read(ep)
    if ann["manifest_sha256"]!=sha(m) or edge["manifest_sha256"]!=sha(m) or edge["annotations_sha256"]!=sha(ap) or edge["camera_index"]!=manifest["camera_index"]:
        raise ValueError("M0 / ELSED provenance mismatch")
    run=a.run or Path(manifest["run"])
    if run.name!=Path(manifest["run"]).name: raise ValueError("M0 run mismatch")
    cfg=read(run/"config.json")
    if cfg["camera_index"]!=manifest["camera_index"]: raise ValueError("Camera mismatch")
    X=validate_extrinsic(np.asarray(manifest["T_camera_lidar"],float))
    K=np.asarray(manifest["K"],float)
    if K.shape!=(3,3) or not np.isfinite(K).all() or not np.allclose(K[2],[0,0,1]):
        raise ValueError("Invalid rectified K")
    ba_views=None;ba_provenance=None
    if a.reference_ba:
        ba_views,ba_provenance=archived_views(a.reference_ba,run,manifest,X)
        for rec in ann["records"]:
            anchor=int(rec["frame"])
            if anchor not in ba_views:raise ValueError("M0 anchor missing from BA")
            path=a.data/f"frame_{anchor:06d}"/"image.png"
            row=next(r for r in manifest["frames"] if int(r["frame"])==anchor)
            im=cv2.imread(str(path));raw=cv2.imread(str(a.image_dir/f"{anchor:06d}.png"))
            if sha(path)!=row["image_sha256"] or im is None or raw is None or not np.array_equal(im,raw):
                raise ValueError("Source sequence mismatch at M0 anchor")
        if not any(r["split"]=="train" and len(ba_views[int(r["frame"])])>1 for r in ann["records"]):
            raise ValueError("No train M0 line with archived nonanchor BA views")
        import pyelsed
    gp=Path(cfg["glim"]);times=read_kitti_times_ns(Path(cfg["sequence"])/"times.txt")
    glim=GlimTrajectory.from_array(np.loadtxt(gp,ndmin=2))
    first=int(cfg["glim_first_frame"]);R=Rotation.from_quat(glim.quaternions_xyzw).as_matrix();t=glim.positions
    if first<0 or first+len(R)>len(times): raise ValueError("GLIM range mismatch")
    if a.out.exists() and any(a.out.iterdir()): raise FileExistsError("Use fresh output")
    a.out.mkdir(parents=True,exist_ok=True)
    settings=dict(max_distance=a.max_distance,max_angle=a.max_angle,min_overlap=a.min_overlap,
                  min_length=a.min_length,ambiguity_gap=a.ambiguity_gap,top_k=a.top_k)
    out=dict(manifest_sha256=sha(m),annotations_sha256=sha(ap),edge_summary_sha256=sha(ep),
        glim_sha256=sha(gp),run=str(run),T_camera_lidar=X.tolist(),K=K.tolist(),settings=settings,
        warning="Fixed M0 initializer; manual ROI 3D lines. No GT. Matching uses no manual image label. No occlusion model; near/image clipping only. Multiple proposals are not proof of physical identity. Review before optimization.",
        ba_support=ba_provenance,observations=[])
    loaded={}
    def load_frame(f):
        if f in loaded: return loaded[f]
        folder=a.edge_data/f"frame_{f:06d}"
        source=a.image_dir/f"{f:06d}.png"
        if str(f) not in edge["frames"]:
            if ba_views is None:raise ValueError("Missing ELSED archive")
            img=cv2.imread(str(source))
            if img is None:raise ValueError(f"Missing source image: {f}")
            seg,_=pyelsed.detect(np.ascontiguousarray(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)))
            seg=np.asarray(seg,float).reshape(-1,4)
            if not np.isfinite(seg).all():raise ValueError("Invalid ELSED detection")
            p=a.out/"detected_segments"/f"{f:06d}.npz"
            p.parent.mkdir(exist_ok=True)
            np.savez_compressed(p,segments=seg)
            loaded[f]=(img,seg,sha(p));return loaded[f]
        info=edge["frames"][str(f)]
        # Old pilot did not hash segment arrays. Record their current hash
        # and verify image pixels against original KITTI source.
        img=cv2.imread(str(folder/"original.png"))
        raw=cv2.imread(str(source))
        if img is None or raw is None or not np.array_equal(img,raw) or list(img.shape[:2])!=info["shape"]:
            raise ValueError(f"Image provenance mismatch: {f}")
        if f in {int(r["frame"]) for r in manifest["frames"]}:
            p=a.data/f"frame_{f:06d}"/"image.png"
            row=next(r for r in manifest["frames"] if int(r["frame"])==f)
            if sha(p)!=row["image_sha256"] or not np.array_equal(cv2.imread(str(p)),img):
                raise ValueError("Prepared M0 image mismatch")
        p=folder/"elsed_segments.npz"
        with np.load(p,allow_pickle=False) as z: seg=np.asarray(z["segments"],float)
        if seg.ndim!=2 or seg.shape[1]!=4 or not np.isfinite(seg).all() or len(seg)!=info["elsed"]["segments"]:
            raise ValueError(f"Invalid ELSED segment archive: {f}")
        loaded[f]=(img,seg,sha(p));return loaded[f]
    ids=set()
    for rec in ann["records"]:
        rid=rec["id"];anchor=int(rec["frame"])
        if rid in ids or not rec.get("manual_identity_confirmed"): raise ValueError("Invalid M0 record")
        ids.add(rid)
        xyz=np.asarray(rec["xyz_lidar"],float)
        if xyz.shape!=(2,3) or not np.isfinite(xyz).all() or np.linalg.norm(xyz[1]-xyz[0])<.1:
            raise ValueError("Invalid M0 3D line")
        if not first<=anchor<first+len(R): raise ValueError("Anchor outside GLIM")
        target_frames=(ba_views[anchor] if ba_views is not None else
                       [anchor+offset for offset in sorted(set(a.offsets)|{0})])
        print(f"{rid}: split={rec['split']} anchor={anchor} selected frames={target_frames}",flush=True)
        for f in target_frames:
            row=dict(id=f"{rid}_f{f:06d}",source_id=rid,feature=rec["feature"],split=rec["split"],
                     anchor_frame=anchor,frame=f,xyz_lidar=xyz.tolist(),projected_uv=None,
                     status="missing_frame",candidates=[],gated_count=0,selected_index=None,cost_gap=None)
            out["observations"].append(row)
            if not first<=f<first+len(R) or (ba_views is None and str(f) not in edge["frames"]): continue
            q=transform_line(xyz,R[anchor-first],t[anchor-first],R[f-first],t[f-first],X)
            if ba_views is not None and f!=anchor:
                shape=next(r["image_shape"] for r in manifest["frames"] if int(r["frame"])==anchor)
                if clip_projected_line(q,K,shape,min_length=a.min_length) is None:
                    row["status"]="not_visible_or_short";continue
            img,seg,digest=load_frame(f)
            row["segments_sha256"]=digest
            p=clip_projected_line(q,K,img.shape[:2],min_length=a.min_length)
            if p is None: row["status"]="not_visible_or_short";continue
            row["projected_uv"]=p.tolist()
            row.update(rank_candidates(p,seg,**settings))
    reject_conflicts(out["observations"])
    for r in out["observations"]:
        if r["frame"] not in loaded: continue
        img,seg,_=loaded[r["frame"]]
        panel=img.copy()
        if r["projected_uv"] is not None: draw_line(panel,r["projected_uv"],(0,0,255),2)
        for j,c in enumerate(r["candidates"]):
            # Candidate IDs, not physical truth. Green only auto-proposed.
            color=(0,255,0) if c["segment_index"]==r["selected_index"] else (0,220,255)
            draw_line(panel,c["uv"],color,2)
            mid=np.rint(np.mean(c["uv"],axis=0)).astype(int)
            cv2.putText(panel,str(c["segment_index"]),tuple(mid),(cv2.FONT_HERSHEY_SIMPLEX),.5,color,1,cv2.LINE_AA)
        label=f"{r['id']} | {r['status']} | red=projection green=proposed yellow=other"
        panel=cv2.copyMakeBorder(panel,30,0,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
        cv2.putText(panel,label,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        if not cv2.imwrite(str(a.out/(r["id"]+".png")),np.vstack((img,panel))):
            raise OSError("Cannot write review image")
    counts={}
    for r in out["observations"]: counts[r["status"]]=counts.get(r["status"],0)+1
    out["counts"]=counts
    write(a.out/"matches.json",out)
    # Explicit review decisions required; auto-proposed index is NOT accepted.
    with (a.out/"review.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["id","anchor_frame","frame","status","proposed_index","candidate_indices","decision","chosen_index"])
        for r in out["observations"]:
            w.writerow([r["id"],r["anchor_frame"],r["frame"],r["status"],r["selected_index"],
                        " ".join(str(c["segment_index"]) for c in r["candidates"]),"",""])
    print(counts,flush=True)


def finalize(a):
    p=a.data/"matches.json";doc=read(p);rows={r["id"]:r for r in doc["observations"]}
    with (a.data/"review.csv").open(newline="") as f: reviewed=list(csv.DictReader(f))
    if len(reviewed)!=len(rows) or {r["id"] for r in reviewed}!=set(rows):
        raise ValueError("Review IDs must match all proposals exactly")
    if a.out.exists(): raise FileExistsError(a.out)
    records=[];counts={};used=set()
    for v in reviewed:
        r=rows[v["id"]];decision=v["decision"].strip().lower()
        if decision not in ("accept","reject","skip"): raise ValueError(f"Explicit accept/reject/skip required: {r['id']}")
        counts[decision]=counts.get(decision,0)+1
        if decision!="accept": continue
        raw=v["chosen_index"].strip()
        idx=int(raw) if raw else r["selected_index"]
        c=next((c for c in r["candidates"] if c["segment_index"]==idx),None)
        if c is None: raise ValueError(f"Accepted index not a saved gated candidate: {r['id']}")
        key=(r["frame"],idx)
        if key in used: raise ValueError("Same image segment accepted for multiple 3D lines")
        used.add(key)
        records.append(dict(id=r["id"],source_id=r["source_id"],feature=r["feature"],split=r["split"],
            anchor_frame=r["anchor_frame"],frame=r["frame"],xyz_lidar=r["xyz_lidar"],uv_image=c["uv"],
            segment_index=idx,manual_identity_confirmed=True))
    groups={}
    for r in records: groups.setdefault(r["source_id"],[]).append(r["frame"])
    if doc.get("ba_support") and not any(r["split"]=="train" and r["frame"]!=r["anchor_frame"] for r in records):
        raise ValueError("No accepted nonanchor train view. Review actual BA frames before finalizing.")
    write(a.out,dict(ba_support=doc.get("ba_support"),matches_sha256=sha(p),review_sha256=sha(a.data/"review.csv"),
          manifest_sha256=doc["manifest_sha256"],annotations_sha256=doc["annotations_sha256"],
          glim_sha256=doc["glim_sha256"],K=doc["K"],T_camera_lidar=doc["T_camera_lidar"],
          counts=counts,accepted_frames_by_source=groups,records=records,
          warning="Human-reviewed fixed associations. xyz_lidar is in anchor_frame, not observation frame. Not an M0/M1 input; S2 backend must transform explicitly."))
    print(counts,groups,flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="mode",required=True)
    q=sub.add_parser("prepare")
    q.add_argument("--data",type=Path,required=True);q.add_argument("--edge-data",type=Path,required=True)
    q.add_argument("--image-dir",type=Path,required=True);q.add_argument("--run",type=Path)
    q.add_argument("--out",type=Path,required=True)
    q.add_argument("--reference-ba",type=Path,help="Use actual H2 archived frame support; detect missing ELSED views")
    q.add_argument("--offsets",type=int,nargs="+",default=[-10,-5,0,5,10])
    q.add_argument("--max-distance",type=float,default=20.)
    q.add_argument("--max-angle",type=float,default=10.)
    q.add_argument("--min-overlap",type=float,default=.3)
    q.add_argument("--min-length",type=float,default=15.)
    q.add_argument("--ambiguity-gap",type=float,default=.1)
    q.add_argument("--top-k",type=int,default=5)
    q.set_defaults(fn=prepare)
    q=sub.add_parser("finalize")
    q.add_argument("--data",type=Path,required=True);q.add_argument("--out",type=Path,required=True)
    q.set_defaults(fn=finalize)
    a=p.parse_args(argv);a.fn(a);return 0


if __name__=="__main__":
    raise SystemExit(main())
