"""L1: point-wise two-plane support on archived L0 lines, no refit."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def sample_support(xyz,plane1,plane2,step=.005):
    p=np.asarray(xyz,float)
    a,b=np.asarray(plane1,float),np.asarray(plane2,float)
    if p.shape!=(2,3) or not np.isfinite(p).all() or not np.isfinite(step) or step<=0:
        raise ValueError("Invalid segment or sample step")
    for q in (a,b):
        if q.ndim!=2 or q.shape[1]!=3 or len(q)==0 or not np.isfinite(q).all():
            raise ValueError("Invalid plane support")
    length=float(np.linalg.norm(p[1]-p[0]))
    if length<=0:raise ValueError("Degenerate segment")
    # Include both endpoints; actual spacing <= requested step.
    n=int(np.ceil(length/step))+1
    t=np.linspace(0,length,n)
    q=p[0]+t[:,None]*(p[1]-p[0])/length
    distances=np.column_stack((cKDTree(a).query(q)[0],cKDTree(b).query(q)[0]))
    return t,q,distances


def support_metrics(t,q,distances,near,far,min_span=.30):
    t=np.asarray(t,float);q=np.asarray(q,float);d=np.asarray(distances,float)
    if (t.ndim!=1 or len(t)<2 or q.shape!=(len(t),3) or d.shape!=(len(t),2)
            or not np.isfinite(t).all() or not np.isfinite(q).all() or not np.isfinite(d).all()
            or np.any(d<0) or t[0]!=0 or np.any(np.diff(t)<=0)
            or not np.isfinite([near,far,min_span]).all() or not 0<near<=far or min_span<=0):
        raise ValueError("Invalid support metrics input")
    ok=(d.min(axis=1)<near)&(d.max(axis=1)<far)
    ids=np.flatnonzero(ok)
    runs=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1) if len(ids) else []
    # Strict contiguous spans: never bridge an unsupported sample.
    spans=[dict(start_m=float(t[r[0]]),end_m=float(t[r[-1]]),
                length_m=float(t[r[-1]]-t[r[0]]),xyz_lidar=q[[r[0],r[-1]]].tolist())
           for r in runs]
    bad=np.flatnonzero(~ok)
    badruns=np.split(bad,np.flatnonzero(np.diff(bad)>1)+1) if len(bad) else []
    gap=max((min(t[-1],t[min(r[-1]+1,len(t)-1)])-t[max(r[0]-1,0)] for r in badruns),default=0.)
    # Length coverage uses intervals whose two endpoints pass, rather than count.
    supported=float(np.sum(np.diff(t)*(ok[:-1]&ok[1:])))
    kept=[s for s in spans if s["length_m"]+1e-12>=min_span]
    return dict(samples=len(t),supported_samples=int(ok.sum()),sample_coverage=float(ok.mean()),
                length_coverage=supported/float(t[-1]),supported_length_m=supported,
                longest_supported_span_m=max((s["length_m"] for s in spans),default=0.),
                max_unsupported_gap_upper_m=float(gap),
                median_near_distance_m=float(np.median(d.min(axis=1))),
                median_far_distance_m=float(np.median(d.max(axis=1))),
                p90_far_distance_m=float(np.quantile(d.max(axis=1),.9)),
                original_like_sample_count_pass=bool(ok.sum()>10),
                accepted_spans=kept,accepted=bool(kept))


def common_interval(ref,lines):
    """Intersection of finite projected spans on the fixed reference axis."""
    a,b=np.asarray(ref,float);length=np.linalg.norm(b-a)
    if length<=0:raise ValueError("Degenerate reference")
    v=(b-a)/length
    lo,hi=0.,float(length)
    for line in lines:
        q=np.asarray(line,float)
        t=(q-a)@v
        lo=max(lo,float(t.min()));hi=min(hi,float(t.max()))
    return (lo,hi) if hi>lo else None


def interval_error(ref,line,interval,n=101):
    """Perpendicular reference-axis cross-section distance on same stations.

    A point on candidate at each reference station is obtained by solving
    projection along the reference direction. This compares actual geometry
    on a fixed interval, not candidate endpoint-dependent distances.
    """
    a,b=np.asarray(ref,float);v=(b-a)/np.linalg.norm(b-a)
    c,e=np.asarray(line,float);w=e-c
    dot=w@v
    if abs(dot)<1e-10:raise ValueError("Candidate perpendicular to reference")
    lo,hi=interval
    t=np.linspace(lo,hi,n)
    alpha=(t-(c-a)@v)/dot
    if np.any(alpha< -1e-8) or np.any(alpha>1+1e-8):
        raise ValueError("Interval outside finite candidate")
    diff=c+alpha[:,None]*w-(a+t[:,None]*v)
    distance=np.linalg.norm(diff,axis=1)
    return dict(mean_m=float(distance.mean()),rmse_m=float(np.sqrt(np.mean(distance**2))),
                max_m=float(distance.max()))


def manual_matches(ref,lines,angle_max=10.,distance_max=.30):
    """Diagnostic matches only. No manual reference used for support selection."""
    a,b=np.asarray(ref,float);L=np.linalg.norm(b-a);v=(b-a)/L
    candidates=[];intervals=[]
    for i,line in enumerate(lines):
        q=np.asarray(line,float);w=q[1]-q[0];length=np.linalg.norm(w)
        if length<=0:continue
        angle=float(np.rad2deg(np.arccos(np.clip(abs(v@w/length),0,1))))
        if angle>angle_max:continue
        interval=common_interval(ref,[q])
        if interval is None:continue
        err=interval_error(ref,q,interval)
        if err["mean_m"]>distance_max:continue
        span=interval[1]-interval[0]
        candidates.append(dict(index=i,angle_deg=angle,interval_m=list(interval),
                               overlap_fraction=span/L,**err))
        intervals.append(interval)
    union=0.;end=0.
    for lo,hi in sorted(intervals):
        union+=max(0.,hi-max(lo,end));end=max(end,hi)
    # Choose candidates with >=30% reference span for individual comparison.
    eligible=[r for r in candidates if r["overlap_fraction"]>=.30]
    eligible.sort(key=lambda r:(r["mean_m"],-r["overlap_fraction"],r["index"]))
    return dict(candidate_count=len(candidates),union_overlap_fraction=min(1.,union/L),
                individual_30pct_count=len(eligible),best=eligible[0] if eligible else None)


def paired_merge_error(ref,raw,merged):
    r=manual_matches(ref,raw)["best"];m=manual_matches(ref,merged)["best"]
    if r is None or m is None:return None
    a,b=raw[r["index"]],merged[m["index"]]
    interval=common_interval(ref,[a,b])
    if interval is None:return None
    return dict(raw_index=r["index"],merged_index=m["index"],interval_m=list(interval),
                length_m=interval[1]-interval[0],raw=interval_error(ref,a,interval),
                merged=interval_error(ref,b,interval),
                warning="Best independently selected raw/merged candidates; not traced merge ancestry.")
