"""L2 conditional scan-wise stability of frozen plane assignments.

Refit using only archived inliers; no correspondence discovery. Scan blocks
are resampled jointly for both planes. This is repeatability, not covariance
or independent ground truth.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import numpy as np


@dataclass(frozen=True)
class StabilitySettings:
    min_points: int=15
    min_scans: int=2
    min_spread_m: float=.10
    max_thickness_ratio: float=.30
    min_angle_deg: float=30.
    near_m: float=.7
    min_overlap_fraction: float=.30
    min_near_points: int=5
    samples: int=21
    bootstrap: int=50
    seed: int=17

    def validate(self):
        if not (self.min_points>=3 and self.min_scans>=1 and self.min_spread_m>0
                and 0<self.max_thickness_ratio<1 and 0<self.min_angle_deg<90
                and self.near_m>0 and 0<self.min_overlap_fraction<=1
                and self.min_near_points>=2 and self.samples>=3 and self.bootstrap>=1):
            raise ValueError("Invalid stability settings")


def fit_subset(p,s,weights,cfg):
    """Integer scan multiplicity weights; unique support gates before weighting."""
    keep=weights>0;p=p[keep];s=s[keep];w=weights[keep].astype(float)
    stats=dict(points=len(p),scans=len(np.unique(s)))
    if len(p)<cfg.min_points or stats["scans"]<cfg.min_scans:
        return None,"insufficient_points_or_scans",stats
    mu=np.average(p,axis=0,weights=w)
    q=p-mu
    cov=(q*w[:,None]).T@q/w.sum()
    e,V=np.linalg.eigh(cov);e=np.maximum(e,0)
    spread=float(np.sqrt(e[1]));ratio=float(np.sqrt(e[0]/max(e[1],1e-30)))
    stats.update(spread_m=spread,thickness_ratio=ratio,rmse_m=float(np.sqrt(e[0])))
    if spread<cfg.min_spread_m:return None,"insufficient_2d_extent",stats
    if ratio>cfg.max_thickness_ratio:return None,"nonplanar_subset",stats
    n=V[:,0]
    return (n,-float(n@mu),p), "ok", stats


def refit(reference,planes,scans,weights,cfg):
    a,b=np.asarray(reference,float);L=np.linalg.norm(b-a);v=(b-a)/L
    fitted=[];stats=[];reasons=[]
    for p,s,w in zip(planes,scans,weights):
        f,reason,st=fit_subset(p,s,w,cfg);stats.append(st);reasons.append(reason)
        fitted.append(f)
    result=dict(plane_stats=stats,plane_status=reasons,status="ok")
    if any(f is None for f in fitted):
        result["status"]="nonplanar_subset" if "nonplanar_subset" in reasons else "insufficient_support"
        return result,None
    (n,d,p),(m,e,q)=fitted
    u=np.cross(n,m);norm=np.linalg.norm(u)
    if norm<np.sin(np.deg2rad(cfg.min_angle_deg)):
        result["status"]="degenerate_intersection";return result,None
    u/=norm
    if u@v<0:u=-u
    # Work in a local origin for numerical stability.
    mid=(a+b)/2
    center=mid+np.linalg.solve(np.stack((n,m,u)),[-(d+n@mid),-(e+m@mid),0.])
    dot=float(u@v)
    angle=float(np.rad2deg(np.arccos(np.clip(dot,0,1))))
    result["angle_to_reference_deg"]=angle
    if dot<1e-6:
        result["status"]="degenerate_intersection";return result,None
    # Measure support in fixed reference-axis stations; require both planes
    # to support a shared interval, not merely infinite line extrapolation.
    spans=[]
    for points in (p,q):
        delta=points-center;t=delta@u
        near=np.linalg.norm(delta-t[:,None]*u,axis=1)<=cfg.near_m
        if near.sum()<cfg.min_near_points:
            result["status"]="insufficient_near_support";return result,None
        spans.append(np.quantile((points[near]-a)@v,[.05,.95]))
    lo=max(0.,*(float(s[0]) for s in spans));hi=min(float(L),*(float(s[1]) for s in spans))
    result["support_interval_m"]=[lo,hi] if hi>lo else None
    result["overlap_fraction"]=max(0.,hi-lo)/L
    if hi-lo<cfg.min_overlap_fraction*L:
        result["status"]="insufficient_overlap";return result,None
    # Represent refit infinite line at fixed reference endpoints.
    # Full interval errors deliberately NOT reported; supported span only.
    line=center+((np.array([0.,L])-(center-a)@v)/dot)[:,None]*u
    result["refit_xyz_at_reference_stations"]=line.tolist()
    result["normal_angle_deg"]=float(np.rad2deg(np.arccos(np.clip(abs(n@m),0,1))))
    return result,line


def compare(reference,line1,line2,interval,samples=21):
    a,b=np.asarray(reference,float);L=np.linalg.norm(b-a)
    lo,hi=interval
    t=np.linspace(lo,hi,samples)/L
    q=np.asarray(line1);p=np.asarray(line2)
    delta=(q[0]-p[0])+t[:,None]*((q[1]-q[0])-(p[1]-p[0]))
    dist=np.linalg.norm(delta,axis=1)
    u=q[1]-q[0];v=p[1]-p[0]
    angle=np.rad2deg(np.arccos(np.clip(abs(u@v)/(np.linalg.norm(u)*np.linalg.norm(v)),0,1)))
    return dict(rmse_m=float(np.sqrt(np.mean(dist**2))),max_m=float(dist.max()),
                angle_deg=float(angle),interval_m=[float(lo),float(hi)],interval_length_m=float(hi-lo))


def evaluate(reference,planes,scans,source_frames,cfg=StabilitySettings(),*,seed=None):
    cfg.validate()
    reference=np.asarray(reference,float)
    if reference.shape!=(2,3) or not np.isfinite(reference).all() or np.linalg.norm(np.diff(reference,axis=0))<=0:
        raise ValueError("Invalid reference")
    planes=[np.asarray(p,float) for p in planes];scans=[np.asarray(s) for s in scans]
    frames=np.asarray(source_frames)
    if (len(planes)!=2 or len(scans)!=2 or frames.ndim!=1 or frames.dtype.kind not in "iu"
            or len(frames)<2 or np.any(np.diff(frames)<=0)):
        raise ValueError("Invalid scan frames")
    for p,s in zip(planes,scans):
        if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or s.shape!=(len(p),) or s.dtype.kind not in "iu" or not np.isin(s,frames).all():
            raise ValueError("Invalid plane points/scan IDs")
    idx=[np.searchsorted(frames,s) for s in scans]
    L=float(np.linalg.norm(np.diff(reference,axis=0)))
    def run(w):
        return refit(reference,planes,scans,[w[i] for i in idx],cfg)
    def pair(mask):
        A,l1=run(mask.astype(int));B,l2=run((~mask).astype(int))
        r=dict(A=A,B=B,status="ok",comparison=None)
        if l1 is None or l2 is None:
            r["status"]="subset_failed";return r
        lo=max(A["support_interval_m"][0],B["support_interval_m"][0])
        hi=min(A["support_interval_m"][1],B["support_interval_m"][1])
        if hi-lo<cfg.min_overlap_fraction*L:
            r["status"]="insufficient_common_overlap";return r
        r["comparison"]=compare(reference,l1,l2,(lo,hi),cfg.samples)
        return r
    odd=pair(np.arange(len(frames))%2==0)
    temporal=pair(np.arange(len(frames))<len(frames)//2)
    rng=np.random.default_rng(cfg.seed if seed is None else seed)
    trials=[];counts=Counter();valid=[]
    for k in range(cfg.bootstrap):
        w=np.bincount(rng.integers(0,len(frames),len(frames)),minlength=len(frames))
        r,line=run(w)
        counts[r["status"]]+=1
        item=dict(trial=k,status=r["status"],unique_drawn_scans=int(np.count_nonzero(w)),
                  plane_status=r["plane_status"],plane_stats=r["plane_stats"])
        if line is not None:
            # Fixed central 30% (or configured fraction) for all bootstrap
            # replicates, preventing variable support span from hiding error.
            lo=L*(1-cfg.min_overlap_fraction)/2;hi=L-lo
            span=r["support_interval_m"]
            if span[0]<=lo+1e-10 and span[1]>=hi-1e-10:
                item["comparison"]=compare(reference,line,reference,(lo,hi),cfg.samples)
                valid.append(item["comparison"])
            else:
                counts["ok"]-=1;counts["insufficient_fixed_overlap"]+=1
                item["status"]="insufficient_fixed_overlap"
        trials.append(item)
    summary=dict(trials=cfg.bootstrap,valid=len(valid),valid_fraction=len(valid)/cfg.bootstrap,
                 statuses=dict(counts),fixed_interval_fraction=cfg.min_overlap_fraction)
    for key in ("rmse_m","max_m","angle_deg"):
        summary[key+"_median"]=float(np.median([x[key] for x in valid])) if valid else None
        summary[key+"_p90"]=float(np.quantile([x[key] for x in valid],.9)) if valid else None
    return dict(odd_even=odd,temporal=temporal,bootstrap=summary,bootstrap_trials=trials,
                original_plane_scans=[len(np.unique(s)) for s in scans])
