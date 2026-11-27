#!/usr/bin/env python3
"""Compare Canny and official ELSED on M0 annotations and adjacent frames."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from spade_calib.calibration.edge_detector_pilot import raster_segments,line_metrics,overlay


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,img):
    if not cv2.imwrite(str(path),img): raise OSError(f"Cannot write {path}")


def detect_timed(fn,repeat):
    fn() # warm up
    times=[]
    for _ in range(repeat):
        t=perf_counter();result=fn();times.append((perf_counter()-t)*1000)
    return result,float(np.median(times))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True,help="M0 prepared directory with annotations.json")
    p.add_argument("--image-dir",type=Path,help="Original rectified KITTI image directory for adjacent views")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--offsets",type=int,nargs="+",default=[-10,-5,0,5,10])
    p.add_argument("--frames",type=int,nargs="+",help="Optional subset of annotated anchor frames")
    p.add_argument("--canny-low",type=float,default=50.)
    p.add_argument("--canny-high",type=float,default=150.)
    p.add_argument("--repeat",type=int,default=10)
    a=p.parse_args(argv)
    if not 0<=a.canny_low<a.canny_high or a.repeat<1: p.error("Invalid Canny thresholds/repeat")
    if any(a.offsets) and a.image_dir is None: p.error("Adjacent frames require --image-dir; use --offsets 0 for anchors only")
    try:
        import pyelsed
    except ImportError as exc:
        raise SystemExit("Install official ELSED: python -m pip install git+https://github.com/iago-suarez/ELSED.git") from exc
    m=a.data/"manifest.json";ap=a.data/"annotations.json"
    manifest=json.loads(m.read_text());doc=json.loads(ap.read_text())
    if doc["manifest_sha256"]!=sha(m): raise ValueError("Annotation manifest mismatch")
    rows={int(r["frame"]):r for r in manifest["frames"]}
    labels={}
    for r in doc["records"]:
        if not r.get("manual_identity_confirmed"): raise ValueError("Unconfirmed physical correspondence")
        f=int(r["frame"])
        if a.frames is None or f in a.frames:
            labels.setdefault(f,[]).append(r)
    if not labels or (a.frames is not None and set(a.frames)!=set(labels)):
        raise ValueError("Requested frames need existing M0 annotations")
    if a.out.exists() and any(a.out.iterdir()): raise FileExistsError("Use fresh output")
    a.out.mkdir(parents=True,exist_ok=True)
    # One thread for OpenCV; ELSED internal threading is not controlled.
    cv2.setNumThreads(1)
    packages={}
    for name in metadata.packages_distributions().get("pyelsed",[]):
        dist=metadata.distribution(name)
        packages[name]=dict(version=dist.version,direct_url=dist.read_text("direct_url.json"))
    result=dict(manifest_sha256=sha(m),annotations_sha256=sha(ap),camera_index=manifest["camera_index"],
        settings=dict(canny_low=a.canny_low,canny_high=a.canny_high,L2gradient=True,
                      offsets=sorted(set(a.offsets)|{0}),repeat=a.repeat,opencv_threads=cv2.getNumThreads(),
                      tolerances_px=[1,2,3,5],band_px=10),
        environment=dict(python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__,
                         elsed_module=str(pyelsed.__file__),elsed_distributions=packages),
        warning="Manual M0 lines are sparse selected examples, not exhaustive ground truth. Coverage uses nearest raster pixels and can be inflated by nearby clutter. No global precision, temporal repeatability or calibration accuracy claim. Neighbor views are unlabeled visual checks only.",
        frames={},missing_frames=[])
    frames=sorted({f+d for f in labels for d in set(a.offsets)|{0}})
    for f in frames:
        records=labels.get(f,[])
        if f in rows and records:
            path=a.data/f"frame_{f:06d}"/"image.png"
            if sha(path)!=rows[f]["image_sha256"]: raise ValueError(f"Prepared image changed: {path}")
            # Verify external sequence identity at each labeled anchor.
            if a.image_dir is not None:
                src=a.image_dir/f"{f:06d}.png"
                src_img=cv2.imread(str(src))
                ref_img=cv2.imread(str(path))
                if src_img is None or ref_img is None or not np.array_equal(src_img,ref_img):
                    raise ValueError(f"Original image directory does not match M0 anchor {f}")
        else:
            path=a.image_dir/f"{f:06d}.png"
            if not path.exists():
                result["missing_frames"].append(f);continue
        img=cv2.imread(str(path))
        if img is None: raise ValueError(f"Cannot read {path}")
        gray=np.ascontiguousarray(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
        canny,tc=detect_timed(lambda:cv2.Canny(gray,a.canny_low,a.canny_high,L2gradient=True),a.repeat)
        (segments,scores),te=detect_timed(lambda:pyelsed.detect(gray),a.repeat)
        segments=np.asarray(segments,float).reshape(-1,4)
        scores=np.asarray(scores).reshape(-1)
        if len(scores)!=len(segments) or not np.isfinite(scores).all(): raise ValueError("Unexpected ELSED scores")
        elsed=raster_segments(segments,gray.shape)
        folder=a.out/f"frame_{f:06d}";folder.mkdir()
        save(folder/"original.png",img)
        save(folder/"canny.png",canny);save(folder/"elsed.png",elsed)
        uv=[r["uv_image"] for r in records]
        co=overlay(img,canny,uv);eo=overlay(img,elsed,uv)
        save(folder/"canny_overlay.png",co);save(folder/"elsed_overlay.png",eo)
        panels=[]
        for name,panel in (("Original",img),("Canny",co),("ELSED",eo)):
            b=cv2.copyMakeBorder(panel,28,0,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
            cv2.putText(b,name,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
            panels.append(b)
        save(folder/"comparison.png",np.vstack(panels))
        np.savez_compressed(folder/"elsed_segments.npz",segments=segments,scores=scores)
        info=dict(image_sha256=sha(path),shape=list(gray.shape),anchors=[g for g in labels if f-g in set(a.offsets)|{0}],
            canny=dict(edge_pixels=int(np.count_nonzero(canny)),detect_median_ms=tc),
            elsed=dict(edge_pixels=int(np.count_nonzero(elsed)),segments=len(segments),detect_median_ms=te),
            annotations=[dict(id=r["id"],feature=r["feature"],split=r["split"],
                uv_image=r["uv_image"],canny=line_metrics(canny,r["uv_image"]),
                elsed=line_metrics(elsed,r["uv_image"])) for r in records])
        result["frames"][str(f)]=info
        print(f"{f:06d}: Canny {tc:.2f} ms, ELSED {te:.2f} ms, {len(segments)} segments, labels={len(records)}",flush=True)
        (a.out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    all_labels=[r for frame in result["frames"].values() for r in frame["annotations"]]
    result["macro_coverage"]={name:{t:float(np.mean([r[name]["thresholds"][t]["coverage"] for r in all_labels]))
                                     for t in ("1.0","2.0","3.0","5.0")} for name in ("canny","elsed")}
    result["labeled_lines"]=len(all_labels)
    (a.out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
