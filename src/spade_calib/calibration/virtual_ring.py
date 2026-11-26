"""Target-view angular sampling of actual accumulated returns, not new beams."""
from __future__ import annotations
import numpy as np


def virtual_rings(xyz,beams,ranks,*,model="offset",max_error_deg=.20,margin_deg=.04):
    """Assign to nearest eligible virtual beam, reject off-beam/ambiguous returns.

    Model and units match recover_rings; unlike source recovery compare only
    requested virtual beams. No point motion or interpolation.
    """
    p=np.asarray(xyz,float);b=np.asarray(beams,float);rr=np.asarray(ranks)
    if (p.ndim!=2 or p.shape[1]!=3 or b.shape!=(64,4)
        or not np.isfinite(p).all() or not np.isfinite(b).all()
        or rr.ndim!=1 or rr.dtype.kind not in "iu" or len(rr)<2
        or len(np.unique(rr))!=len(rr) or np.any((rr<0)|(rr>=64))
        or model not in ("offset","angle") or not np.isfinite([max_error_deg,margin_deg]).all()
        or max_error_deg<=0 or margin_deg<0):
        raise ValueError("Invalid virtual beam inputs")
    rr=np.sort(rr);bb=b[rr];a=np.deg2rad(bb[:,1]);v=bb[:,2];h=bb[:,3]
    best=np.empty(len(p),int);err=np.empty(len(p));gap=np.empty(len(p))
    for start in range(0,len(p),8192):
        sl=slice(start,start+8192);q=p[sl]
        rho=np.hypot(q[:,0],q[:,1])[:,None];z=q[:,2,None]
        if model=="angle":
            e=abs(np.rad2deg(np.arctan2(z,rho))-bb[:,1])
        else:
            radial=np.sqrt(np.maximum(rho*rho-h*h,0))
            e=abs(np.rad2deg(np.arctan2(z-v/np.cos(a),radial))-bb[:,1])
            e=np.where(rho>abs(h),e,1e6)
        order=np.argsort(e,axis=1,kind="stable")[:,:2]
        first,second=np.take_along_axis(e,order,axis=1).T
        best[sl]=rr[order[:,0]];err[sl]=first;gap[sl]=second-first
    valid=(err<=max_error_deg)&(gap>=margin_deg)
    return dict(ring=np.where(valid,best,-1),error_deg=err,margin_deg=gap)


def angular_grid(xyz,ring,scan_ids,target,*,bin_deg=.2,policy="nearest"):
    """One actual return per ring/azimuth bin, unknown=-1.

    target: target scan only. nearest: radial z-buffer of all supplied points.
    fill: target observed bins have priority; nearest historical in empty bins.
    No guarantees of visibility between bins or beam-width correction.
    """
    p=np.asarray(xyz,float);r=np.asarray(ring);s=np.asarray(scan_ids)
    if (p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all()
        or r.shape!=(len(p),) or r.dtype.kind not in "iu"
        or s.shape!=(len(p),) or s.dtype.kind not in "iu"
        or np.any((r<-1)|(r>=64)) or np.any(s<0)
        or not np.isfinite(bin_deg) or not .01<=bin_deg<=10
        or policy not in ("target","nearest","fill")):
        raise ValueError("Invalid angular grid inputs")
    n=int(np.ceil(360/bin_deg));step=360/n
    az=np.rad2deg(np.arctan2(p[:,1],p[:,0]))%360
    col=np.minimum((az/step).astype(int),n-1)
    ranges=np.linalg.norm(p,axis=1)
    ok=(r>=0)&(ranges>0)
    if policy=="target":ok &= s==target
    idx=np.flatnonzero(ok)
    cell=r[idx]*n+col[idx]
    priority=(s[idx]!=target).astype(int) if policy=="fill" else np.zeros(len(idx),int)
    order=np.lexsort((idx,ranges[idx],priority,cell))
    _,first=np.unique(cell[order],return_index=True)
    win=order[first]
    out=np.full((64,n),-1,int)
    out.flat[cell[win]]=idx[win]
    return out


def grid_stats(grid,target_grid,ranks,*,gap_deg=.8):
    """Occupancy and cyclic gaps; compare only rows in ranks."""
    g=np.asarray(grid)[ranks];t=np.asarray(target_grid)[ranks]
    if g.shape!=t.shape or g.ndim!=2 or g.shape[1]==0:raise ValueError("Invalid grids")
    a=g>=0;b=t>=0
    gap=[];long=0;empty_rows=0
    for row in a:
        x=np.flatnonzero(row)
        if not len(x):empty_rows+=1;continue
        d=np.diff(np.r_[x,x[0]+len(row)])*360/len(row)
        gap.extend(d.tolist());long+=int((d>gap_deg+1e-9).sum())
    # 360-degree gap of singleton row included; empty rows reported separately.
    return dict(bins=int(a.size),occupied=int(a.sum()),occupancy=float(a.mean()),
                target_occupied=int(b.sum()),added_bins=int((a&~b).sum()),
                lost_bins=int((b&~a).sum()),replaced_target_bins=int((a&b&(g!=t)).sum()),
                empty_rings=empty_rows,long_gap_count=long,
                gap_deg_p50_p90_p99=np.percentile(gap,[50,90,99]).tolist() if gap else None)
