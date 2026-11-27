"""MVS2: fixed-support, sign-invariant normal-gated plane RANSAC.

No extrinsic, GT, or scale fitting. All coordinates are metric anchor camera.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from spade_calib.correspondence.sfm_lidar_planes import _largest_connected


@dataclass(frozen=True)
class Settings:
    voxel_m: float = .10
    max_range_m: float = 65.
    max_points: int = 40000
    distance_m: float = .12
    normal_deg: float = 20.
    connectivity_m: float = .65
    min_points: int = 100
    min_extent_m: float = .5
    max_planes: int = 12
    iterations: int = 512
    ground_tilt_deg: float = 15.
    nonground_min_tilt_deg: float = 25.
    seed: int = 20261127

    def __post_init__(self):
        if (not np.isfinite([self.voxel_m,self.max_range_m,self.distance_m,self.normal_deg,
              self.connectivity_m,self.min_extent_m,self.ground_tilt_deg,self.nonground_min_tilt_deg]).all()
            or min(self.voxel_m,self.max_range_m,self.distance_m,self.connectivity_m,self.min_extent_m)<=0
            or not 0<self.normal_deg<90 or not 0<self.ground_tilt_deg<self.nonground_min_tilt_deg<90
            or self.min_points<3 or self.max_points<self.min_points or min(self.max_planes,self.iterations)<1):
            raise ValueError("Invalid MVS2 settings")


UP=np.array([0.,-1.,0.])


def sample_cloud(xyz,normals,cfg):
    xyz=np.asarray(xyz,float);n=np.asarray(normals,float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or n.shape!=xyz.shape:
        raise ValueError("Expected aligned Nx3 XYZ and normals")
    mag=np.linalg.norm(n,axis=1)
    valid=np.isfinite(xyz).all(1)&np.isfinite(n).all(1)&(mag>1e-8)
    dist=np.linalg.norm(xyz,axis=1)
    ids=np.flatnonzero(valid&(dist>=1)&(dist<=cfg.max_range_m))
    if len(ids)<cfg.min_points:raise ValueError("Insufficient finite points/normals")
    _,idx=np.unique(np.floor(xyz[ids]/cfg.voxel_m).astype(np.int64),axis=0,return_index=True)
    ids=ids[np.sort(idx)]
    voxel_count=len(ids)
    if len(ids)>cfg.max_points:
        ids=np.sort(np.random.default_rng(cfg.seed).choice(ids,cfg.max_points,replace=False))
    return xyz[ids],n[ids]/mag[ids,None],ids,dict(input_points=len(xyz),valid_normal_points=int(valid.sum()),voxel_points=voxel_count,sampled_points=len(ids))


def fit(p):
    c=p.mean(0);q=p-c
    ev,v=np.linalg.eigh(q.T@q/len(p))
    n=v[:,0];d=-float(n@c)
    if n[np.argmax(abs(n))]<0:n=-n;d=-d
    return n,d,v[:,1:],np.sqrt(np.maximum(ev,0))


def inliers(p,normals,n,d,cfg,aware):
    ok=np.abs(p@n+d)<=cfg.distance_m
    if aware:ok &= np.abs(normals@n)>=np.cos(np.deg2rad(cfg.normal_deg))
    return ok


def direction_ok(n,mode,cfg):
    a=abs(n@UP)
    if mode=="ground":return a>=np.cos(np.deg2rad(cfg.ground_tilt_deg))
    return a<=np.cos(np.deg2rad(cfg.nonground_min_tilt_deg))


def describe(p,normals,ids,n,d,basis,spread):
    pts=p[ids];r=np.abs(pts@n+d)
    ang=np.rad2deg(np.arccos(np.clip(np.abs(normals[ids]@n),0,1)))
    q=(pts-pts.mean(0))@basis
    extent=np.percentile(q,95,axis=0)-np.percentile(q,5,axis=0)
    return dict(normal=n.tolist(),offset_m=float(d),support_count=len(ids),
                centroid_m=pts.mean(0).tolist(),rmse_m=float(np.sqrt(np.mean(r*r))),
                abs_distance_p50_p90_m=np.percentile(r,[50,90]).tolist(),
                normal_error_p50_p90_deg=np.percentile(ang,[50,90]).tolist(),
                inplane_extent_p05_p95_m=extent.tolist(),inplane_rms_m=spread[1:].tolist(),
                area_proxy_m2=float(np.prod(extent)),
                tilt_from_up_deg=float(np.rad2deg(np.arccos(np.clip(abs(n@UP),0,1)))),
                abs_normal_xyz=np.abs(n).tolist())


def extract_one(p,normals,ids,cfg,aware,mode,rng):
    """Triplet hypotheses; identical distance gate, optional normal gate.

    Score in batches; try top distinct support hypotheses. Use largest
    connected finite support and refit, never count disconnected coplanar
    points as one patch. Final supports satisfy the final model's gates.
    """
    if len(ids)<cfg.min_points:return None
    pts=p[ids];ns=normals[ids]
    proposals=[]
    for _ in range(cfg.iterations):
        tri=pts[rng.choice(len(pts),3,replace=False)]
        n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);mag=np.linalg.norm(n)
        if mag<1e-8:continue
        n/=mag
        if not direction_ok(n,mode,cfg):continue
        d=-float(n@tri[0])
        if mode=="ground":
            if n@UP<0:n=-n;d=-d
            if not .35<=d<=3.5:continue
        proposals.append((n,d))
    if not proposals:return None
    N=np.stack([v[0] for v in proposals]);D=np.array([v[1] for v in proposals])
    scores=[]
    for i in range(0,len(N),32):
        n=N[i:i+32];d=D[i:i+32]
        valid=np.abs(pts@n.T+d)<=cfg.distance_m
        if aware:valid &= np.abs(ns@n.T)>=np.cos(np.deg2rad(cfg.normal_deg))
        scores.extend(valid.sum(0).tolist())
    best=None
    tested=[]
    for j in np.argsort(-np.array(scores),kind="stable"):
        if scores[j]<cfg.min_points or len(tested)>=12:break
        n,d=N[j],D[j]
        if any(abs(n@a)>.999 and abs(d-np.sign(n@a)*b)<cfg.distance_m*.5 for a,b in tested):continue
        tested.append((n,d))
        chosen=ids[inliers(pts,ns,n,d,cfg,aware)]
        chosen=chosen[_largest_connected(p[chosen],cfg.connectivity_m)]
        if len(chosen)<cfg.min_points:continue
        # Monotone trimming/refit; reject unstable membership rather than
        # returning a plane whose final support violates its own gate.
        for _ in range(12):
            if len(chosen)<cfg.min_points:break
            n,d,basis,spread=fit(p[chosen])
            keep=inliers(p[chosen],normals[chosen],n,d,cfg,aware)
            if keep.all():
                connected=_largest_connected(p[chosen],cfg.connectivity_m)
                if len(connected)==len(chosen):break
                chosen=chosen[connected]
            else:chosen=chosen[keep]
        else:continue
        if len(chosen)<cfg.min_points or not direction_ok(n,mode,cfg):continue
        if mode=="ground":
            if n@UP<0:n=-n;d=-d
            if not .35<=d<=3.5:continue
        info=describe(p,normals,chosen,n,d,basis,spread)
        if min(info["inplane_extent_p05_p95_m"])<cfg.min_extent_m:continue
        if best is None or len(chosen)>len(best[0]):best=(chosen,info)
    return best


def normal_rank(rows):
    """Equal plane weights; translation-only geometry proxy, not full SE(3)."""
    N=np.asarray([r["normal"] for r in rows],float).reshape(-1,3)
    if not len(N):return dict(plane_count=0,rank=0,singular_values=[0.,0.,0.],min_sv=0.,condition=None,weak_direction_xyz=None)
    _,s,V=np.linalg.svd(N/np.sqrt(len(N)),full_matrices=True)
    s=np.pad(s,(0,3-len(s)))
    rank=int(np.count_nonzero(s>max(1e-10,.05*s[0])))
    return dict(plane_count=len(N),rank=rank,singular_values=s.tolist(),min_sv=float(s[-1]),
                condition=float(s[0]/s[-1]) if rank==3 else None,
                weak_direction_xyz=V[-1].tolist())


def experiment(p,n,cfg):
    roi=np.flatnonzero((p[:,2]>=3)&(p[:,2]<=25)&(abs(p[:,0])<=3)&
                       (p[:,1]>=.35)&(p[:,1]<=3.5))
    ground={}
    for name,aware in (("G0",False),("G1",True)):
        ground[name]=extract_one(p,n,roi,cfg,aware,"ground",np.random.default_rng(cfg.seed))
    # Use a single ground result for both nonground arms. Only remove
    # points compatible with its normal AND distance, over full sample.
    # No full-scene semantic road labeling is claimed.
    g=ground["G1"]
    if g is None:return ground,{},np.arange(len(p)),dict(ground_roi_points=len(roi),reason="G1_failed")
    row=g[1];mask=inliers(p,n,np.array(row["normal"]),row["offset_m"],cfg,True)
    remaining=np.flatnonzero(~mask)
    arms={}
    for name,aware in (("P0",False),("P1",True)):
        rng=np.random.default_rng(cfg.seed+1);ids=remaining.copy();planes=[]
        for _ in range(cfg.max_planes):
            patch=extract_one(p,n,ids,cfg,aware,"nonground",rng)
            if patch is None:break
            planes.append(patch)
            ids=ids[~np.isin(ids,patch[0])]
        arms[name]=planes
    return ground,arms,remaining,dict(ground_roi_points=len(roi),ground_compatible_removed=int(mask.sum()))
