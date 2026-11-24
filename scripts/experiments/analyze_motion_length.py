#!/usr/bin/env python3
"""Diagnose within-segment motion length error from trajectory_oracle outputs.

GT is diagnostic only. No trajectory refinement or extrinsic optimization.
Blocked CV tests low-order log-scale models, not a new calibration method.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def norm(x):
    return np.linalg.norm(x, axis=-1)


def metrics(e):
    e = np.asarray(e)
    if not len(e):
        return None
    return dict(count=len(e), mean=float(np.mean(e)), median=float(np.median(e)),
                mae=float(np.mean(abs(e))), rmse=float(np.sqrt(np.mean(e**2))),
                p90_abs=float(np.percentile(abs(e),90)))


def corr(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12:
        return None
    return float(np.corrcoef(a,b)[0,1])


def fit_log_scale(t,y,degree):
    D=np.polynomial.polynomial.polyvander(t,degree)
    x=np.linalg.lstsq(D,y,rcond=None)[0]
    f=least_squares(lambda a:D@a-y,x,loss="soft_l1",f_scale=.02)
    if not f.success:
        raise RuntimeError(f.message)
    return f.x


def blocked_predictions(frames, groups, time, target, folds=5):
    """Each test block is contiguous within segment; purge interval overlaps.

    No train motion interval may intersect ANY test interval. Predictions
    of boundary blocks involve extrapolation; report separately.
    """
    predictions={d:np.full(len(time),np.nan) for d in (0,1)}
    extrap=np.zeros(len(time),dtype=bool)
    fold_ids=np.full(len(time),-1,dtype=int)
    records=[]
    for k in np.unique(groups):
        ids=np.flatnonzero((groups==k)&np.isfinite(target))
        if len(ids)<12:
            continue
        ids=ids[np.argsort(frames[ids,0])]
        center=time[ids].mean()
        span=max(np.ptp(time[ids]),1e-9)
        t=(time-center)/span
        for j,test in enumerate(np.array_split(ids,folds)):
            if not len(test):
                continue
            lo,hi=frames[test,0].min(),frames[test,1].max()
            train=ids[(frames[ids,1]<lo)|(frames[ids,0]>hi)]
            if len(train)<8 or np.ptp(t[train])<1e-9:
                continue
            extrap[test]=(time[test]<time[train].min())|(time[test]>time[train].max())
            fold_ids[test]=j
            for d in (0,1):
                a=fit_log_scale(t[train],target[train],d)
                predictions[d][test]=np.polynomial.polynomial.polyval(t[test],a)
            records.append(dict(segment=int(k),fold=j,train=len(train),test=len(test),
                                test_start=int(lo),test_end=int(hi)))
    return predictions,extrap,fold_ids,records


def lag_correlations(frames,groups,e,lags):
    """Raw frame-start lag, within segment, segment-median centered."""
    out=[]
    centered=e.copy()
    for k in np.unique(groups):
        m=groups==k
        centered[m]-=np.median(e[m])
    lookup={(int(k),int(i)):j for j,(k,i) in enumerate(zip(groups,frames[:,0]))}
    for lag in lags:
        i=[];j=[]
        for n,(k,f) in enumerate(zip(groups,frames[:,0])):
            other=lookup.get((int(k),int(f)+lag))
            if other is not None:
                i.append(n);j.append(other)
        i,j=np.asarray(i,dtype=int),np.asarray(j,dtype=int)
        # Adjacent overlapping pairs inflate apparent correlation.
        nonoverlap=frames[i,1]<frames[j,0]
        out.append(dict(lag_frames=lag,count=len(i),correlation=corr(centered[i],centered[j]),
                        nonoverlap_count=int(nonoverlap.sum()),
                        nonoverlap_correlation=corr(centered[i[nonoverlap]],centered[j[nonoverlap]])))
    return out


def analyze(data,timestamps=None,min_length=.5,folds=5):
    frames=np.asarray(data["frame_ids"])
    g=np.asarray(data["segment_ids"])
    s=np.asarray(data["reference_scales"])
    X=np.asarray(data["X_he"])
    A,B=np.asarray(data["ta"]),np.asarray(data["tb"])
    G=np.asarray(data["tb_gt"])
    Rb,Rgt=np.asarray(data["Rb"]),np.asarray(data["Rb_gt"])
    n,gt=norm(B),norm(G)
    if frames.shape!=(len(g),2) or np.any(frames[:,1]<=frames[:,0]) or np.any(np.diff(frames[:,0])<=0):
        raise ValueError("Invalid ordered frame pair identities")
    if g.dtype.kind not in "iu" or np.any(g<0) or np.any(g>=len(s)) or np.any(s<=0):
        raise ValueError("Invalid segment scales")
    if any(not np.isfinite(x).all() for x in (A,B,G,Rb,Rgt,s,X)):
        raise ValueError("Nonfinite motion")
    # Metric camera translation predicted by the existing HE equation.
    # GLIM proxy uses archived HE, never GT X. It is not independent of HE.
    proxy=A@X[:3,:3].T-(Rb-np.eye(3))@X[:3,3]
    pn=norm(proxy)
    if timestamps is not None:
        stamps=np.asarray(timestamps)
        if stamps.ndim!=1 or np.any(frames<0) or frames.max()>=len(stamps) or not np.isfinite(stamps).all() or np.any(np.diff(stamps)<=0):
            raise ValueError("Invalid times.txt")
        start,end=stamps[frames[:,0]],stamps[frames[:,1]]
        time_unit="seconds"
    else:
        start,end=frames[:,0].astype(float),frames[:,1].astype(float)
        time_unit="raw_frames"
    time=(start+end)/2
    valid=(gt>=min_length)&(n*s[g]>=min_length)&(pn>=min_length)
    if valid.sum()<12:
        raise ValueError("Insufficient nonstationary pairs")
    excluded=int((~valid).sum())
    frames,g,time,start,end,A,B,G,proxy,Rb,Rgt,n,gt,pn=[x[valid] for x in
        (frames,g,time,start,end,A,B,G,proxy,Rb,Rgt,n,gt,pn)]
    ref=s[g]
    length=ref*n
    # Radial/tangential decomposition in GT camera-start coordinate axes.
    u=G/gt[:,None]
    vec=ref[:,None]*B-G
    radial=np.sum(vec*u,axis=1)
    tangent=norm(vec-radial[:,None]*u)
    angle=np.rad2deg(np.arccos(np.clip(np.sum(B*G,axis=1)/(n*gt),-1,1)))
    oracle=np.log(gt/n)
    proxy_log=np.log(pn/n)
    relative_log=np.log(length/gt)
    rv=np.rad2deg(Rotation.from_matrix(Rgt).as_rotvec())
    rot_err=np.rad2deg(Rotation.from_matrix(np.swapaxes(Rgt,1,2)@Rb).magnitude())
    speed=gt/(end-start)
    report=dict(diagnostic_only=True,time_unit=time_unit,count=len(g),excluded=excluded,
                min_length_m=min_length,folds=folds,
                archived_length_error_m=metrics(length-gt),
                archived_relative_error_percent=metrics(100*(length/gt-1)),
                direction_error_deg=metrics(angle),radial_error_m=metrics(radial),
                tangential_error_m=metrics(tangent),segments=[],cv={})
    for k in np.unique(g):
        m=g==k
        report["segments"].append(dict(segment=int(k),count=int(m.sum()),reference_scale=float(s[k]),
            oracle_scale_median=float(np.median(np.exp(oracle[m]))),
            oracle_scale_p10_p90=np.percentile(np.exp(oracle[m]),[10,90]).tolist(),
            length_error_m=metrics((length-gt)[m]),direction_error_deg=metrics(angle[m]),
            radial_error_m=metrics(radial[m]),tangential_error_m=metrics(tangent[m]),
            log_scale_vs_time=corr(time[m],oracle[m]),
            log_error_vs_speed=corr(relative_log[m],speed[m]),
            log_error_vs_turn=corr(relative_log[m],norm(rv[m])),
            log_error_vs_signed_camera_y_rotation=corr(relative_log[m],rv[m,1])))
    def center(x):
        return x-np.array([np.median(x[g==k]) for k in g])
    report["within_segment_correlations"]=dict(
        log_error_vs_speed=corr(center(relative_log),center(speed)),
        log_error_vs_turn=corr(center(relative_log),center(norm(rv))),
        log_error_vs_signed_camera_y_rotation=corr(center(relative_log),center(rv[:,1])),
        log_error_vs_direction_error=corr(center(relative_log),center(angle)))
    report["lag_correlations"]=lag_correlations(frames,g,relative_log,[1,5,10,11,20,30,50])
    all_preds={}
    for label,target in (("GT_oracle",oracle),("GLIM_HE_proxy",proxy_log)):
        pred,extra,fold,records=blocked_predictions(frames,g,time,target,folds)
        common=np.isfinite(pred[0])&np.isfinite(pred[1])
        r=dict(coverage=float(common.mean()),folds=records)
        for name,p in (("constant",pred[0]),("linear",pred[1]),("archived_scale",np.log(ref))):
            all_preds[label+"_"+name]=p
            er=np.exp(p)*n-gt
            r[name]={a:metrics(er[m]) for a,m in
                     (("all",common),("interpolation",common&~extra),("extrapolation",common&extra))}
        report["cv"][label]=r
        all_preds[label+"_fold"]=fold
    report["warning"]="GT fits are supervised oracle diagnostics. GLIM proxy and archived scale use HE fitted on full data, so CV is conditional, not independent GT-free calibration validation. Pair norm ratio is effective scale, not proof of physical scale drift. Motion pairs overlap; correlations are descriptive."
    columns=dict(frame_i=frames[:,0],frame_j=frames[:,1],segment=g,time=time,
                 gt_length_m=gt,sfm_length=n,archived_scale=ref,metric_length_m=length,
                 oracle_scale=np.exp(oracle),proxy_scale=np.exp(proxy_log),
                 length_error_m=length-gt,relative_error_percent=100*(length/gt-1),
                 direction_error_deg=angle,radial_error_m=radial,tangential_error_m=tangent,
                 speed_m_per_time_unit=speed,gt_turn_deg=norm(rv),gt_camera_y_rotation_deg=rv[:,1],
                 rotation_error_deg=rot_err,**all_preds)
    return report,columns


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True,help="trajectory_oracleXX directory")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--times",type=Path,help="KITTI times.txt; otherwise raw frame units")
    p.add_argument("--min-length",type=float,default=.5)
    p.add_argument("--folds",type=int,default=5)
    args=p.parse_args()
    if args.folds<2 or not np.isfinite(args.min_length) or args.min_length<=0:
        raise ValueError("Invalid folds or minimum length")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use fresh output directory")
    with np.load(args.run/"motion_pairs.npz",allow_pickle=False) as data:
        report,c=analyze(data,np.loadtxt(args.times) if args.times else None,args.min_length,args.folds)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    with (args.out/"motion_errors.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(c);w.writerows(zip(*c.values()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(3,1,figsize=(13,10),sharex=True)
    t=c["time"]
    for name in ("oracle_scale","archived_scale","proxy_scale"):
        ax[0].plot(t,c[name],label=name,lw=1)
    ax[0].set_ylabel("m / SfM unit");ax[0].legend()
    ax[1].plot(t,c["length_error_m"],label="length error")
    ax[1].plot(t,c["radial_error_m"],label="radial")
    ax[1].plot(t,c["tangential_error_m"],label="tangential")
    ax[1].set_ylabel("m");ax[1].legend()
    ax[2].plot(t,c["direction_error_deg"],label="direction")
    ax[2].plot(t,c["rotation_error_deg"],label="rotation")
    ax[2].set_ylabel("degrees");ax[2].set_xlabel(report["time_unit"]);ax[2].legend()
    for x in ax:
        for i in np.flatnonzero(np.diff(c["segment"])!=0)+1:
            x.axvline(t[i],color="gray",alpha=.4)
        x.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.out/"motion_diagnostics.png",dpi=150);plt.close(fig)
    print(json.dumps({k:report[k] for k in ("count","archived_length_error_m","direction_error_deg")},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
