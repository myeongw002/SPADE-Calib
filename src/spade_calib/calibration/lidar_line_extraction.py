"""L0: local plane-pair intersections from frozen accumulated LiDAR clouds.

All thresholds are geometric heuristics, not calibrated uncertainty.
No camera, ground truth or manual labels enter extraction.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class Settings:
    cell_m: float=2.
    halo_fraction: float=.25
    threshold_m: float=.08
    trials: int=256
    min_inliers: int=30
    min_spread_m: float=.10
    max_thickness_ratio: float=.30
    max_planes: int=3
    min_angle_deg: float=30.
    near_m: float=.7
    max_edge_gap_m: float=.25
    min_near: int=10
    min_scans: int=2
    min_length_m: float=.75
    max_support_gap_m: float=.5
    merge_angle_deg: float=8.
    merge_distance_m: float=.15
    merge_gap_m: float=.25
    merge_plane_angle_deg: float=12.
    max_range_m: float=60.
    seed: int=17

    def validate(self):
        if not (self.cell_m>0 and 0<=self.halo_fraction<=.5 and self.threshold_m>0
                and self.trials>=1 and self.min_inliers>=3 and self.min_spread_m>0
                and 0<self.max_thickness_ratio<1 and self.max_planes>=2
                and 0<self.min_angle_deg<90 and 0<self.max_edge_gap_m<=self.near_m
                and self.min_near>=3 and self.min_scans>=1 and self.min_length_m>0
                and self.max_support_gap_m>0 and 0<self.merge_angle_deg<90
                and self.merge_distance_m>0 and self.merge_gap_m>=0
                and 0<self.merge_plane_angle_deg<90 and self.max_range_m>0):
            raise ValueError("Invalid extraction settings")


def fit_plane(p,ids,cfg):
    if len(ids)<cfg.min_inliers: return None
    q=p[ids];mu=q.mean(0)
    _,s,V=np.linalg.svd(q-mu,full_matrices=False)
    spread=s[1]/np.sqrt(len(q))
    if spread<cfg.min_spread_m or s[2]/max(s[1],1e-12)>cfg.max_thickness_ratio:
        return None
    n=V[-1];d=-float(n@mu)
    err=q@n+d
    if np.sqrt(np.mean(err**2))>cfg.threshold_m or np.max(abs(err))>2*cfg.threshold_m:
        return None
    return dict(normal=n,offset=d,ids=np.asarray(ids,int),rmse_m=float(np.sqrt(np.mean(err**2))),
                spread_m=float(spread),thickness_ratio=float(s[2]/s[1]))


def ransac_plane(p,ids,cfg,rng):
    q=p[ids];N=len(q)
    if N<cfg.min_inliers:return None
    # Sample triples in batches; no Nxtrials allocation for large cells.
    hypotheses=[]
    for _ in range(0,cfg.trials,64):
        ntrial=min(64,cfg.trials-_)
        tri=rng.integers(0,N,size=(ntrial,3))
        a,b,c=q[tri[:,0]],q[tri[:,1]],q[tri[:,2]]
        n=np.cross(b-a,c-a);length=np.linalg.norm(n,axis=1)
        ok=length>1e-8
        if not ok.any():continue
        n=n[ok]/length[ok,None];a=a[ok]
        d=-np.einsum("ij,ij->i",a,n)
        err=abs(np.einsum("ij,kj->ik",q,n)+d)
        good=err<=cfg.threshold_m
        count=good.sum(axis=0)
        loss=np.sum(np.where(good,err,0),axis=0)
        for j in np.lexsort((loss,-count))[:6]:
            if count[j]>=cfg.min_inliers:
                hypotheses.append((int(count[j]),float(loss[j]),n[j],d[j]))
    hypotheses.sort(key=lambda a:(-a[0],a[1]))
    for _,__,n,d in hypotheses[:12]:
        inliers=ids[abs(q@n+d)<=cfg.threshold_m]
        plane=None
        for step in range(3):
            plane=fit_plane(p,inliers,cfg)
            if plane is None:break
            if step<2:
                inliers=ids[abs(q@plane["normal"]+plane["offset"])<=cfg.threshold_m]
        if plane is not None:return plane
    return None


def intersection(p,scans,a,b,cfg):
    n1,n2=a["normal"],b["normal"]
    v=np.cross(n1,n2);norm=np.linalg.norm(v)
    if norm<np.sin(np.deg2rad(cfg.min_angle_deg)):return None,"parallel"
    v/=norm
    if v[np.argmax(abs(v))]<0:v=-v
    origin=np.linalg.solve(np.stack((n1,n2,v)),[-a["offset"],-b["offset"],0.])
    spans=[];ts=[];support=[]
    for plane in (a,b):
        ids=plane["ids"];q=p[ids]-origin;t=q@v
        distance=np.linalg.norm(q-t[:,None]*v,axis=1)
        near=distance<=cfg.near_m
        if near.sum()<cfg.min_near:return None,"near_support"
        # Reject intersections lying far outside the observed plane patch.
        if np.quantile(distance[near],.15)>cfg.max_edge_gap_m:return None,"edge_gap"
        ids=ids[near];t=t[near]
        if len(np.unique(scans[ids]))<cfg.min_scans:return None,"scan_support"
        spans.append(np.quantile(t,[.05,.95]));ts.append(t)
        support.append(ids)
    lo=max(s[0] for s in spans);hi=min(s[1] for s in spans)
    if hi-lo<cfg.min_length_m:return None,"short_overlap"
    # Require each plane to support the overlapping span without large gaps.
    stats=[]
    for t,ids in zip(ts,support):
        use=(t>=lo)&(t<=hi)
        if use.sum()<cfg.min_near or len(np.unique(scans[ids[use]]))<cfg.min_scans:
            return None,"shared_support"
        gap=np.diff(np.r_[lo,np.sort(t[use]),hi]).max()
        if gap>cfg.max_support_gap_m:return None,"support_gap"
        stats.append(dict(points=int(use.sum()),scans=int(len(np.unique(scans[ids[use]]))),
                          largest_gap_m=float(gap)))
    return dict(xyz=origin+np.array([lo,hi])[:,None]*v,planes=(a,b),support=stats,
                angle_deg=float(np.rad2deg(np.arccos(np.clip(abs(n1@n2),0,1)))),
                horizontal_plane=bool(max(abs(n1[2]),abs(n2[2]))>np.cos(np.deg2rad(25)))),"ok"


def line_relation(ref,other):
    a,b=np.asarray(ref,float);q=np.asarray(other,float)
    L=np.linalg.norm(b-a);v=(b-a)/L
    w=q[1]-q[0];w/=np.linalg.norm(w)
    t=(q-a)@v
    u=(np.stack((a,b))-q[0])@w
    # Compare line positions locally at endpoints, not just the midpoint.
    d1=np.linalg.norm(q-a-t[:,None]*v,axis=1)
    d2=np.linalg.norm(np.stack((a,b))-q[0]-u[:,None]*w,axis=1)
    gap=max(0.,float(t.min()-L),float(-t.max()))
    overlap=max(0.,min(L,float(t.max()))-max(0.,float(t.min())))
    return dict(angle_deg=float(np.rad2deg(np.arccos(np.clip(abs(v@w),0,1)))),
                mean_distance_m=float((d1.mean()+d2.mean())/2),
                max_distance_m=float(max(d1.max(),d2.max())),gap_m=gap,
                overlap_m=overlap,reference_overlap_fraction=overlap/L)


def merge_pair(p,scans,a,b,cfg):
    rel=line_relation(a["xyz"],b["xyz"])
    if (rel["angle_deg"]>cfg.merge_angle_deg or rel["max_distance_m"]>cfg.merge_distance_m
            or rel["gap_m"]>cfg.merge_gap_m):return None
    pa,pb=a["planes"],b["planes"]
    cosine=np.cos(np.deg2rad(cfg.merge_plane_angle_deg))
    # Require both support planes to match, not simply collinear edges.
    for perm in ((0,1),(1,0)):
        if any(abs(pa[i]["normal"]@pb[j]["normal"])<cosine for i,j in enumerate(perm)):
            continue
        fitted=[]
        for i,j in enumerate(perm):
            x,y=pa[i],pb[j]
            if (abs(p[x["ids"]].mean(0)@y["normal"]+y["offset"])>cfg.merge_distance_m
                or abs(p[y["ids"]].mean(0)@x["normal"]+x["offset"])>cfg.merge_distance_m):
                break
            z=fit_plane(p,np.union1d(x["ids"],y["ids"]),cfg)
            if z is None:break
            fitted.append(z)
        if len(fitted)!=2:continue
        merged,_=intersection(p,scans,*fitted,cfg)
        if merged is not None:
            merged["cells"]=a["cells"]+b["cells"]
            return merged
    return None


def extract_lines(points,scan_frames,cfg=Settings()):
    cfg.validate()
    p=np.asarray(points,float);scans=np.asarray(scan_frames)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or scans.shape!=(len(p),) or scans.dtype.kind not in "iu":
        raise ValueError("Expected finite Nx3 points and integer scan IDs")
    ids=np.flatnonzero(np.linalg.norm(p,axis=1)<=cfg.max_range_m)
    if not len(ids):return [],[],dict(input_points=len(p),range_points=0)
    cells=np.unique(np.floor(p[ids]/cfg.cell_m).astype(np.int64),axis=0)
    tree=cKDTree(p[ids]);rng=np.random.default_rng(cfg.seed)
    raw=[];counts=Counter()
    # Halo avoids missing corners that fall on voxel boundaries.
    for cell in cells:
        center=(cell+.5)*cfg.cell_m
        local=ids[np.asarray(tree.query_ball_point(center,cfg.cell_m*(.5+cfg.halo_fraction),p=np.inf),int)]
        if len(local)<2*cfg.min_inliers:continue
        counts["eligible_cells"]+=1
        planes=[];remaining=np.sort(local)
        for _ in range(cfg.max_planes):
            plane=ransac_plane(p,remaining,cfg,rng)
            if plane is None:break
            planes.append(plane)
            remaining=np.setdiff1d(remaining,plane["ids"],assume_unique=True)
            if len(remaining)<cfg.min_inliers:break
        counts["planes"]+=len(planes)
        for i in range(len(planes)):
            for j in range(i+1,len(planes)):
                line,reason=intersection(p,scans,planes[i],planes[j],cfg)
                counts[reason]+=1
                if line is not None:
                    line["cells"]=[cell.tolist()];raw.append(line)
    # Greedy conservative merge with refit, deterministic long-first order.
    # Not a global segmentation or a guarantee of physical identity.
    merged=[]
    for line in sorted(raw,key=lambda r:-np.linalg.norm(np.diff(r["xyz"],axis=0))):
        current=line
        j=0
        while j<len(merged):
            other=merged[j]
            # Cheap bound before line/plane tests.
            radius=(np.linalg.norm(np.diff(current["xyz"],axis=0))+np.linalg.norm(np.diff(other["xyz"],axis=0)))/2
            if np.linalg.norm(current["xyz"].mean(0)-other["xyz"].mean(0))>radius+cfg.merge_gap_m+cfg.merge_distance_m:
                j+=1;continue
            candidate=merge_pair(p,scans,current,other,cfg)
            if candidate is None:j+=1;continue
            current=candidate;merged.pop(j);j=0
        merged.append(current)
    return raw,merged,dict(input_points=len(p),range_points=len(ids),occupied_cells=len(cells),
                            raw_lines=len(raw),merged_lines=len(merged),**counts)
