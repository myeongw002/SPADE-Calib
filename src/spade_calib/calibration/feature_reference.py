"""FV1 reference-review utilities. All labels require human confirmation."""
from __future__ import annotations
import numpy as np


def select_tiles(shape,uv,*,width=224,height=124,per_stratum=2,seed=17):
    """Disjoint image tiles, random draw from evidence-rich and other strata.

    Evidence is raw64 projected jump count, not positive ground truth.
    Both strata are retained; this is not a uniform full-image sample.
    """
    h,w=shape;u=np.asarray(uv,float)
    if (min(h,w,width,height,per_stratum)<=0 or u.ndim!=2 or u.shape[1]!=2
        or not np.isfinite(u).all() or np.any(u<0) or np.any(u>=np.array([w,h]))):
        raise ValueError("Invalid tiles/evidence")
    boxes=[];counts=[]
    for y in range(0,h,height):
        for x in range(0,w,width):
            x1=min(x+width,w);y1=min(y+height,h)
            if x1-x<width/2 or y1-y<height/2:continue
            boxes.append([x,y,x1,y1])
            counts.append(int(((u[:,0]>=x)&(u[:,0]<x1)&(u[:,1]>=y)&(u[:,1]<y1)).sum()))
    if len(boxes)<2:raise ValueError("Need at least two eligible tiles")
    counts=np.asarray(counts)
    order=np.argsort(counts,kind="stable")
    strata={"rich":order[len(order)//2:],"other":order[:len(order)//2]}
    rng=np.random.default_rng(seed);out=[]
    for name,ids in strata.items():
        for i in sorted(rng.choice(ids,min(per_stratum,len(ids)),replace=False)):
            out.append(dict(box=boxes[i],stratum=name,evidence=int(counts[i]),
                            population=len(ids),sample=min(per_stratum,len(ids))))
    return out


def angular_coordinates(xyz):
    p=np.asarray(xyz,float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():
        raise ValueError("Invalid XYZ")
    return np.column_stack((np.rad2deg(np.arctan2(p[:,1],p[:,0])),
        np.rad2deg(np.arctan2(p[:,2],np.hypot(p[:,0],p[:,1]))),
        np.linalg.norm(p,axis=1)))


def local_evidence(angular,candidate,*,az_radius=2.,el_radius=2.):
    """Return observed raw64 neighbors relative to a candidate ray.

    Azimuth wraps; radial range remains absolute. No interpolation.
    """
    a=np.asarray(angular,float);c=np.asarray(candidate,float)
    if (a.ndim!=2 or a.shape[1]!=3 or c.shape!=(3,) or not np.isfinite(a).all()
        or not np.isfinite(c).all() or min(az_radius,el_radius)<=0
        or max(az_radius,el_radius)>=90):
        raise ValueError("Invalid local evidence")
    d=a.copy();d[:,0]=(d[:,0]-c[0]+180)%360-180;d[:,1]-=c[1]
    ids=np.flatnonzero((abs(d[:,0])<=az_radius)&(abs(d[:,1])<=el_radius))
    return ids,d[ids]


def blinded_samples(arms,*,per_arm=3,seed=17):
    """Independent uniform sampling in each eligible arm, then deduplicate.

    Membership is retained privately; do not pool as an unbiased estimate.
    """
    if per_arm<1:raise ValueError("Invalid sample size")
    rng=np.random.default_rng(seed);selected={};meta={}
    for name,ids in sorted(arms.items()):
        a=np.asarray(ids)
        if a.ndim!=1 or a.dtype.kind not in "iu" or np.any(a<0):
            raise ValueError("Invalid candidate IDs")
        a=np.unique(a);n=min(len(a),per_arm)
        choice=rng.choice(a,n,replace=False)
        meta[name]=dict(population=len(a),sample=n)
        for i in choice:selected.setdefault(int(i),[]).append(name)
    ids=np.asarray(list(selected),int);rng.shuffle(ids)
    return [(int(i),selected[int(i)]) for i in ids],meta
