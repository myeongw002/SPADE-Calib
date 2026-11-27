"""Validate native-scan edge candidates against an anchor-view range grid.

No synthetic points and no candidate position replacement. Unknown support
is not equivalent to occlusion or to a verified boundary.
"""
from __future__ import annotations
import numpy as np


def native_visibility_masks(triples,eligible,surface_retained,ghost_keep):
    """Return per-triple masks; baseline already passes ghost near test.

    A/B isolate newly inferred surface occlusion only, so out-of-range or
    out-of-FOV points are not silently treated as hidden. B_ghost additionally
    rejects original far/back free-space conflicts (diagnostic, not truth).
    """
    e=np.asarray(triples);ok=np.asarray(eligible,bool)
    retained=np.asarray(surface_retained,bool);ghost=np.asarray(ghost_keep,bool)
    if (ok.ndim!=1 or retained.shape!=ok.shape or ghost.shape!=ok.shape
        or e.ndim!=2 or e.shape[1]!=3 or e.dtype.kind not in "iu"
        or np.any((e<0)|(e>=len(ok))) or np.any(retained&~ok)):
        raise ValueError("Invalid native visibility inputs")
    removed=ok&~retained
    a=~removed[e[:,0]]
    b=~np.any(removed[e],axis=1)
    bg=b&np.all(ghost[e],axis=1)
    return a,b,bg


def anchor_edge_support(xyz,ids,grid,*,el_min=-26.,el_max=4.,
                        center_tol_m=.2,az_gap_deg=.8,el_gap_deg=.8,
                        jump_abs=1.,jump_rel=.1,same_abs=.4,same_rel=.05):
    """Test native near returns themselves for H/V jump in the anchor grid.

    Require an occupied same cell with range agreement to the candidate.
    Then use nearest observed cell in each direction, no extrapolation.
    Compare actual angles/ranges relative to the *candidate*, not the grid
    winner. Both sides must be observed; azimuth periodic, elevation not.
    Returns same-cell support, H and V pass flags, and actual far/back IDs
    (-1 when unsupported). Unknown is excluded from strict C but retained
    by A/B unless explicitly contradicted.
    """
    p=np.asarray(xyz,float);i=np.asarray(ids);g=np.asarray(grid)
    vals=[center_tol_m,az_gap_deg,el_gap_deg,jump_abs,jump_rel,same_abs,same_rel]
    if (p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all()
        or i.ndim!=1 or i.dtype.kind not in "iu" or np.any((i<0)|(i>=len(p)))
        or g.ndim!=2 or min(g.shape)<1 or g.dtype.kind not in "iu"
        or np.any((g<-1)|(g>=len(p))) or not np.isfinite(vals+[el_min,el_max]).all()
        or min(vals)<=0 or max(az_gap_deg,el_gap_deg)>=90
        or not -90<=el_min<el_max<=90):
        raise ValueError("Invalid anchor edge inputs")
    ne,na=g.shape;da=360/na;de=(el_max-el_min)/ne
    rho=np.hypot(p[:,0],p[:,1]);r=np.linalg.norm(p,axis=1)
    az=np.rad2deg(np.arctan2(p[:,1],p[:,0]))%360
    el=np.rad2deg(np.arctan2(p[:,2],rho))
    valid=(rho[i]>1e-8)&(el[i]>=el_min)&(el[i]<el_max)
    row=np.clip(np.floor((el[i]-el_min)/de).astype(int),0,ne-1)
    col=np.clip(np.floor(az[i]/da).astype(int),0,na-1)
    same=g[row,col];safe=np.maximum(same,0)
    center=valid&(same>=0)&(abs(r[safe]-r[i])<=center_tol_m) if len(p) else valid
    def side(axis,sign,gap):
        found=np.full(len(i),-1,int)
        limit=min((na-1) if axis=="H" else (ne-1),
                  int(np.ceil(gap/(da if axis=="H" else de)))+1)
        for k in range(1,limit+1):
            rr=row if axis=="H" else row+sign*k
            cc=(col+sign*k)%na if axis=="H" else col
            j=g[np.clip(rr,0,ne-1),cc]
            take=center&(found<0)&(rr>=0)&(rr<ne)&(j>=0)
            found[take]=j[take]
        return found
    def direction(axis,gap):
        lo=side(axis,-1,gap);hi=side(axis,1,gap)
        passed=np.zeros(len(i),bool);far_out=np.full(len(i),-1,int);back_out=far_out.copy()
        both=np.flatnonzero(center&(lo>=0)&(hi>=0))
        if len(both):
            ii=i[both];ll=lo[both];hh=hi[both]
            if axis=="H":
                dl=(az[ii]-az[ll])%360;dh=(az[hh]-az[ii])%360
            else:
                dl=el[ii]-el[ll];dh=el[hh]-el[ii]
            close=(dl>1e-8)&(dh>1e-8)&(dl<=gap)&(dh<=gap)
            for far,back in ((ll,hh),(hh,ll)):
                yes=close&(r[far]-r[ii]>=np.maximum(jump_abs,jump_rel*r[ii]))
                yes &= abs(r[back]-r[ii])<=np.maximum(same_abs,same_rel*r[ii])
                dest=both[yes];passed[dest]=True;far_out[dest]=far[yes];back_out[dest]=back[yes]
        return passed,far_out,back_out
    h,hf,hb=direction("H",az_gap_deg);v,vf,vb=direction("V",el_gap_deg)
    return dict(center_supported=center,h=h,v=v,h_far=hf,h_back=hb,v_far=vf,v_back=vb)
