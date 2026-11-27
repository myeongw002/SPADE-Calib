"""LiDAR-only anchor visibility proxies. Reconstructed depths never become features."""
from __future__ import annotations
from itertools import combinations
import numpy as np


def ray_triangle_depth(d,a,b,c,*,max_edge_m=.5,min_cos=.15):
    """Vectorized unit-ray / triangle intersections; +inf if unsupported.

    Three original vertices, no extrapolation outside triangle. Reject long
    edges and grazing/degenerate triangles. Rays originate at anchor origin.
    """
    e=b-a;f=c-a
    normal=np.cross(e,f);area=np.linalg.norm(normal,axis=1)
    h=np.cross(d,f);det=np.einsum("ij,ij->i",e,h)
    valid=(area>1e-10)&(abs(det)>=min_cos*area)
    valid &= np.maximum.reduce([np.linalg.norm(e,axis=1),np.linalg.norm(f,axis=1),
                                np.linalg.norm(c-b,axis=1)])<=max_edge_m
    den=np.where(valid,det,1.)
    s=-a;q=np.cross(s,e)
    u=np.einsum("ij,ij->i",s,h)/den
    v=np.einsum("ij,ij->i",d,q)/den
    t=np.einsum("ij,ij->i",f,q)/den
    valid &= (u>=-1e-9)&(v>=-1e-9)&(u+v<=1.+1e-9)&(t>0)&np.isfinite(t)
    return np.where(valid,t,np.inf)


def visibility_depths(xyz,grid,eligible,*,el_min,el_max,splat_deg=.3,
                      patch_radius=1,max_edge_m=.5,min_cos=.15):
    """Estimate occluding range along each actual query ray.

    grid: nearest actual point per anchor spherical cell (MV6).
    Splat uses a rectangular-angle-plane disk, not a true solid-angle disk.
    Surface uses four possible triangles from the four corner cells around
    the query cell. All triangle vertices are real observed grid returns.
    Grid is only a spatial lookup; intersection uses actual XYZ and ray.
    Azimuth wraps; elevation does not. No iterative propagation.
    """
    p=np.asarray(xyz,float);g=np.asarray(grid);ok=np.asarray(eligible,bool)
    if (p.ndim!=2 or p.shape[1]!=3 or ok.shape!=(len(p),) or not np.isfinite(p).all()
        or g.ndim!=2 or min(g.shape)<1 or g.dtype.kind not in "iu" or np.any((g<-1)|(g>=len(p)))
        or not np.isfinite([el_min,el_max,splat_deg,max_edge_m,min_cos]).all()
        or not -90<=el_min<el_max<=90 or not 0<splat_deg<=2
        or not isinstance(patch_radius,(int,np.integer)) or not 1<=patch_radius<=3
        or max_edge_m<=0 or not 0<min_cos<=1):
        raise ValueError("Invalid visibility inputs")
    ne,na=g.shape;de=(el_max-el_min)/ne;da=360/na
    rho=np.hypot(p[:,0],p[:,1]);r=np.linalg.norm(p,axis=1)
    az=np.rad2deg(np.arctan2(p[:,1],p[:,0]))%360
    el=np.rad2deg(np.arctan2(p[:,2],rho))
    ids=np.flatnonzero(ok&(rho>1e-8)&(el>=el_min)&(el<el_max))
    row=np.minimum(np.floor((el[ids]-el_min)/de).astype(int),ne-1)
    col=np.minimum(np.floor(az[ids]/da).astype(int),na-1)
    ds=np.full(len(p),np.inf);dt=np.full(len(p),np.inf)
    # Batch to bound intermediate memory; one read-only support map for both arms.
    re=int(np.ceil(splat_deg/de));ra=int(np.ceil(splat_deg/da))
    corner=[(-patch_radius,-patch_radius),(-patch_radius,patch_radius),
            (patch_radius,-patch_radius),(patch_radius,patch_radius)]
    for start in range(0,len(ids),32768):
        sl=slice(start,start+32768);i=ids[sl];rr=row[sl];cc=col[sl]
        def lookup(dr,dc):
            nr=rr+dr
            return np.where((nr>=0)&(nr<ne),g[np.clip(nr,0,ne-1),(cc+dc)%na],-1)
        best=np.full(len(i),np.inf)
        for dr in range(-re,re+1):
            for dc in range(-ra,ra+1):
                j=lookup(dr,dc);safe=np.maximum(j,0)
                azdiff=(az[safe]-az[i]+180)%360-180
                dist2=azdiff**2+(el[safe]-el[i])**2
                valid=(j>=0)&(dist2<=splat_deg**2+1e-12)
                best=np.minimum(best,np.where(valid,r[safe],np.inf))
        ds[i]=best
        # Complete triangle fan around target cell, without assigning a
        # reconstructed point to the range grid. Test actual ray intersection.
        corners=[lookup(dr,dc) for dr,dc in corner]
        unit=p[i]/r[i,None];best=np.full(len(i),np.inf)
        for ia,ib,ic in combinations(range(4),3):
            ja,jb,jc=(corners[k] for k in (ia,ib,ic))
            good=np.flatnonzero((ja>=0)&(jb>=0)&(jc>=0))
            if not len(good):continue
            d=ray_triangle_depth(unit[good],p[ja[good]],p[jb[good]],p[jc[good]],
                                 max_edge_m=max_edge_m,min_cos=min_cos)
            best[good]=np.minimum(best[good],d)
        dt[i]=best
    return ds,dt


def filter_occluded(xyz,eligible,protected,depth,*,margin_m=.5):
    """Keep unknown depths and protected (direct anchor) measurements."""
    p=np.asarray(xyz,float);ok=np.asarray(eligible,bool);protect=np.asarray(protected,bool)
    d=np.asarray(depth,float)
    if (p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all()
        or any(x.shape!=(len(p),) for x in (ok,protect,d))
        or np.isnan(d).any() or np.any(d<=0) or not np.isfinite(margin_m) or margin_m<0):
        raise ValueError("Invalid occlusion mask inputs")
    r=np.linalg.norm(p,axis=1)
    conflict=ok&np.isfinite(d)&(r>d+margin_m)
    removed=conflict&~protect
    return ok&~removed,dict(eligible=int(ok.sum()),supported=int((ok&np.isfinite(d)).sum()),
                           conflicts=int(conflict.sum()),protected_conflicts=int((conflict&protect).sum()),
                           removed=int(removed.sum()),retained=int((ok&~removed).sum()))
