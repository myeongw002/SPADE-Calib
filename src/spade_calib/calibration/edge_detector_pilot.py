"""Detector-only pilot: common raster evaluation against sparse manual lines.

No matching, no LiDAR projection, no claim of global edge precision.
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy.spatial import cKDTree


def raster_segments(segments, shape):
    s=np.asarray(segments,float).reshape(-1,4)
    if not np.isfinite(s).all(): raise ValueError("Nonfinite segment")
    h,w=shape
    out=np.zeros((h,w),np.uint8)
    for a in s:
        # Fixed-point drawing preserves subpixel endpoints before rasterization.
        p=np.rint(a*16).astype(int)
        cv2.line(out,tuple(p[:2]),tuple(p[2:]),255,1,cv2.LINE_8,shift=4)
    return out


def line_metrics(mask,uv,tolerances=(1.,2.,3.,5.),band_px=10.):
    """Sample manual finite line every <=1 px; nearest detected raster pixel.

    Nearby clutter can inflate coverage. Band support is not precision:
    unannotated real boundaries are not negatives.
    """
    mask=np.asarray(mask)!=0
    uv=np.asarray(uv,float)
    if mask.ndim!=2 or uv.shape!=(2,2) or not np.isfinite(uv).all():
        raise ValueError("Invalid raster or annotation")
    if np.any(uv<0) or np.any(uv>=np.array(mask.shape[::-1])):
        raise ValueError("Annotation outside image")
    length=float(np.linalg.norm(uv[1]-uv[0]))
    if length<15 or not np.isfinite(tolerances).all() or any(t<=0 for t in tolerances) or band_px<max(tolerances):
        raise ValueError("Invalid line or tolerances")
    n=int(np.ceil(length))+1
    q=np.linspace(uv[0],uv[1],n)
    yy,xx=np.nonzero(mask)
    points=np.column_stack((xx,yy))
    if len(points):
        distances=cKDTree(points).query(q)[0]
        v=uv[1]-uv[0]
        alpha=np.clip((points-uv[0])@v/(v@v),0,1)
        near=np.linalg.norm(points-uv[0]-alpha[:,None]*v,axis=1)
        band_count=int(np.count_nonzero(near<=band_px))
    else:
        distances=np.full(n,np.inf);band_count=0
    result={}
    for t in tolerances:
        good=distances<=t
        bad=np.flatnonzero(~good)
        runs=np.split(bad,np.flatnonzero(np.diff(bad)>1)+1) if len(bad) else []
        result[str(float(t))]=dict(coverage=float(good.mean()),
            longest_uncovered_px=float(min(length,max((len(r) for r in runs),default=0)*length/(n-1))))
    finite=bool(len(points))
    return dict(length_px=length,samples=n,nearest_median_px=float(np.median(distances)) if finite else None,
                nearest_p90_px=float(np.percentile(distances,90)) if finite else None,
                nearest_rmse_px=float(np.sqrt(np.mean(distances**2))) if finite else None,
                band_edge_pixels=band_count,thresholds=result)


def overlay(img,mask,labels=()):
    out=img.copy()
    # Green detections, magenta manual segment. Preserve original dimensions.
    out[mask!=0]=[0,255,0]
    for uv in labels:
        p=np.rint(np.asarray(uv)).astype(int)
        cv2.line(out,tuple(p[0]),tuple(p[1]),(255,0,255),1,cv2.LINE_AA)
    return out
