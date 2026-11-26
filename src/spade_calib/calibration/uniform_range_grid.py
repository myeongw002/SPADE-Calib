"""Uniform spherical angular grid at a single anchor pose."""
from __future__ import annotations
import numpy as np
from .multiview_edge_evidence import ring_edges


def spherical_grid(xyz,eligible,*,az_deg=.2,el_deg=.2,el_min=-26.,el_max=4.):
    """Nearest actual return per angular cell, no interpolation; -1 unknown.
    Geometric elevation from LiDAR origin, not physical beam calibration.
    Bounds are half-open; azimuth is periodic, elevation is not.
    """
    p=np.asarray(xyz,float);ok=np.asarray(eligible,bool)
    if (p.ndim!=2 or p.shape[1]!=3 or ok.shape!=(len(p),)
        or not np.isfinite(p).all() or not np.isfinite([az_deg,el_deg,el_min,el_max]).all()
        or not .05<=az_deg<=5 or not .05<=el_deg<=5 or not -90<=el_min<el_max<=90):
        raise ValueError("Invalid spherical grid settings")
    na=int(np.ceil(360/az_deg));ne=int(np.ceil((el_max-el_min)/el_deg-1e-10))
    da=360/na;de=(el_max-el_min)/ne
    rho=np.hypot(p[:,0],p[:,1]);r=np.linalg.norm(p,axis=1)
    az=np.rad2deg(np.arctan2(p[:,1],p[:,0]))%360
    el=np.rad2deg(np.arctan2(p[:,2],rho))
    ids=np.flatnonzero(ok&(rho>1e-8)&(el>=el_min)&(el<el_max))
    row=np.minimum(np.floor((el[ids]-el_min)/de).astype(int),ne-1)
    col=np.minimum(np.floor(az[ids]/da).astype(int),na-1)
    cell=row*na+col
    order=np.lexsort((ids,r[ids],cell))
    _,first=np.unique(cell[order],return_index=True)
    win=order[first]
    g=np.full((ne,na),-1,int);g.flat[cell[win]]=ids[win]
    return g,dict(az_step_deg=da,el_step_deg=de,eligible_in_fov=len(ids),
                  eligible_outside_fov=int(ok.sum()-len(ids)))


def grid_edges(xyz,grid,*,az_gap_deg=.8,el_gap_deg=.8,jump_abs=1.,
               jump_rel=.1,same_abs=.4,same_rel=.05):
    """Horizontal and vertical near/far/back triples from actual returns.
    Nearest occupied neighbors, actual angle gaps (not bin gaps).
    Vertical neighbors never wrap; missing cells never treated as zero.
    """
    p=np.asarray(xyz,float);g=np.asarray(grid)
    if (p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all()
        or g.ndim!=2 or g.dtype.kind not in "iu" or np.any((g<-1)|(g>=len(p)))
        or not np.isfinite([az_gap_deg,el_gap_deg,jump_abs,jump_rel,same_abs,same_rel]).all()
        or min(az_gap_deg,el_gap_deg,jump_abs,jump_rel,same_abs,same_rel)<=0):
        raise ValueError("Invalid edge inputs")
    rows,cols=np.nonzero(g>=0);ids=g[rows,cols]
    if len(np.unique(ids))!=len(ids):raise ValueError("Repeated grid point ID")
    h=ids[ring_edges(p[ids],rows,az_gap_deg=az_gap_deg,jump_abs=jump_abs,
                     jump_rel=jump_rel,same_abs=same_abs,same_rel=same_rel)]
    el=np.rad2deg(np.arctan2(p[:,2],np.hypot(p[:,0],p[:,1])))
    r=np.linalg.norm(p,axis=1);out=[]
    for col in np.flatnonzero(np.sum(g>=0,axis=0)>=3):
        i=g[:,col];i=i[i>=0]
        center=i[1:-1];lo=i[:-2];hi=i[2:]
        d0=el[center]-el[lo];d1=el[hi]-el[center]
        ok=(d0>1e-8)&(d1>1e-8)&(d0<=el_gap_deg)&(d1<=el_gap_deg)
        for far,back in ((lo,hi),(hi,lo)):
            keep=ok&(r[far]-r[center]>=np.maximum(jump_abs,jump_rel*r[center]))
            keep &= abs(r[back]-r[center])<=np.maximum(same_abs,same_rel*r[center])
            out.append(np.column_stack((center[keep],far[keep],back[keep])))
    v=np.concatenate(out) if out else np.empty((0,3),int)
    return h,v
