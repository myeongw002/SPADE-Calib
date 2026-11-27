"""M0: independently observed image lines and finite LiDAR plane intersections."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def fit_wall(points, *, seed=17, threshold=.08, trials=768, min_inliers=30):
    """Local RANSAC + orthogonal least squares. LiDAR z is approximate up."""
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<min_inliers or not np.isfinite(p).all():
        raise ValueError("Not enough finite ROI points")
    if threshold<=0 or trials<1 or min_inliers<3:
        raise ValueError("Invalid wall fitting settings")
    rng=np.random.default_rng(seed)
    best=None;score=(-1,-np.inf)
    for _ in range(trials):
        a,b,c=p[rng.choice(len(p),3,replace=False)]
        n=np.cross(b-a,c-a);l=np.linalg.norm(n)
        if l<1e-8: continue
        n/=l
        if abs(n[2])>np.sin(np.deg2rad(25)): continue
        err=abs((p-a)@n);mask=err<=threshold
        key=(int(mask.sum()),-float(err[mask].sum()))
        if key>score: best=mask;score=key
    if best is None or best.sum()<min_inliers:
        raise ValueError("No supported near-vertical plane in ROI")
    for _ in range(3):
        q=p[best];mu=q.mean(0)
        _,s,V=np.linalg.svd(q-mu,full_matrices=False)
        if s[1]/np.sqrt(len(q))<.10:
            raise ValueError("ROI plane has insufficient two-dimensional extent")
        n=V[-1];d=-n@mu
        if abs(n[2])>np.sin(np.deg2rad(25)):
            raise ValueError("Refitted plane not near vertical")
        best=abs(p@n+d)<=threshold
        if best.sum()<min_inliers: raise ValueError("Plane lost support")
    q=p[best];mu=q.mean(0)
    _,s,V=np.linalg.svd(q-mu,full_matrices=False)
    n=V[-1];d=-n@mu
    if abs(n[2])>np.sin(np.deg2rad(25)) or s[1]/np.sqrt(len(q))<.10:
        raise ValueError("Final plane is degenerate")
    return n,float(d),best,dict(inliers=int(best.sum()),roi_points=len(p),
        rmse_m=float(np.sqrt(np.mean((q@n+d)**2))),inlier_fraction=float(best.mean()))


def finite_intersection(n1,d1,p1,n2,d2,p2,*,near=.7,min_length=.75,min_near=10):
    n1=np.asarray(n1,float);n2=np.asarray(n2,float)
    direction=np.cross(n1,n2);a=np.linalg.norm(direction)
    if a<np.sin(np.deg2rad(25)): raise ValueError("Planes nearly parallel")
    direction/=a
    if abs(direction[2])<np.cos(np.deg2rad(25)):
        raise ValueError("Intersection not near vertical")
    if direction[2]<0: direction=-direction
    origin=np.linalg.solve(np.stack([n1,n2,direction]),[-d1,-d2,0.])
    spans=[];counts=[]
    for p in (p1,p2):
        v=np.asarray(p,float)-origin;t=v@direction
        good=np.linalg.norm(v-t[:,None]*direction,axis=1)<=near
        counts.append(int(good.sum()))
        if counts[-1]<min_near: raise ValueError("Insufficient actual plane support near intersection")
        spans.append(np.quantile(t[good],[.05,.95]))
    lo=max(x[0] for x in spans);hi=min(x[1] for x in spans)
    if hi-lo<min_length: raise ValueError("No sufficiently long overlapping plane support")
    return origin+np.array([lo,hi])[:,None]*direction,dict(
        support_near=counts,length_m=float(hi-lo),
        angle_deg=float(np.rad2deg(np.arccos(np.clip(abs(n1@n2),0,1)))))


def image_line(uv):
    uv=np.asarray(uv,float)
    if uv.shape!=(2,2) or not np.isfinite(uv).all() or np.linalg.norm(uv[1]-uv[0])<15:
        raise ValueError("Image line must have two finite endpoints at least 15 pixels apart")
    l=np.cross(np.r_[uv[0],1.],np.r_[uv[1],1.])
    return l/np.linalg.norm(l[:2])


def sample_line(xyz,n=5):
    xyz=np.asarray(xyz,float)
    if xyz.shape!=(2,3) or not np.isfinite(xyz).all() or np.linalg.norm(xyz[1]-xyz[0])<.1:
        raise ValueError("Invalid finite 3D segment")
    return xyz[0]+np.linspace(0,1,n)[:,None]*(xyz[1]-xyz[0])


def project_line(xyz,X,K):
    pc=np.asarray(xyz)@X[:3,:3].T+X[:3,3]
    z=pc[:,2]
    uv=(pc@K.T)[:,:2]/np.maximum(z[:,None],.1)
    return uv,z


def perturb(X,x):
    Y=np.array(X,dtype=float,copy=True)
    Y[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@Y[:3,:3]
    Y[:3,3]+=x[3:]
    return Y


def residual(X,records,K,sigma_px=2.,depth_barrier=100.):
    """Equal line weight, fixed samples, no changing correspondence mask."""
    rows=[]
    for r in records:
        uv,z=project_line(sample_line(r["xyz_lidar"]),X,K)
        d=uv@image_line(r["uv_image"])[:2]+image_line(r["uv_image"])[2]
        rows.append(np.column_stack((d/sigma_px,np.maximum(.5-z,0)*depth_barrier)).ravel())
    return np.concatenate(rows)


def metrics(X,records,K):
    if not records: return None
    errors=[];depth=[];rows=[]
    for r in records:
        uv,z=project_line(sample_line(r["xyz_lidar"]),X,K)
        l=image_line(r["uv_image"]);d=uv@l[:2]+l[2]
        errors.extend(d);depth.extend(z)
        # Finite-segment overlap diagnostic, not a constraint: visible image
        # segment may be shorter than the independently reconstructed line.
        a,b=np.asarray(r["uv_image"],float);v=(b-a)/np.linalg.norm(b-a)
        s=(uv-a)@v;length=np.linalg.norm(b-a)
        overlap=max(0.,min(float(s.max()),length)-max(float(s.min()),0.))
        rows.append(dict(id=r["id"],frame=r["frame"],rmse_px=float(np.sqrt(np.mean(d*d))),
                         min_depth_m=float(z.min()),overlap_px=overlap))
    return dict(lines=len(records),frames=len(set(r["frame"] for r in records)),
                rmse_px=float(np.sqrt(np.mean(np.square(errors)))),
                median_abs_px=float(np.median(abs(np.asarray(errors)))),
                nonpositive_or_near_samples=int(np.count_nonzero(np.asarray(depth)<.5)),
                per_line=rows)


def information(X,records,K,sigma_px=2.):
    """Local line-only GN in 1deg/.1m units; no uncertainty calibration."""
    units=np.r_[np.full(3,np.deg2rad(1.)),np.full(3,.1)]
    def f(x): return residual(perturb(X,x*units),records,K,sigma_px)
    J=np.column_stack([(f(d)-f(-d))/2e-5 for d in np.eye(6)*1e-5])
    _,sv,Vh=np.linalg.svd(J,full_matrices=False)
    rank=int(np.count_nonzero(sv>sv[0]*1e-8)) if len(sv) else 0
    # Translation x conditioned on all other 5 extrinsic variables.
    others=J[:,[0,1,2,4,5]]
    col=J[:,3];rem=col-others@np.linalg.lstsq(others,col,rcond=1e-8)[0]
    return dict(singular_values=sv.tolist(),effective_rank=rank,
                weakest_direction_normalized=Vh[-1].tolist(),
                units_rad_m=units.tolist(),information_matrix=(J.T@J).tolist(),
                tx_conditional_information=float(rem@rem),
                condition_number_J=float(sv[0]/sv[-1]) if rank==6 else None,
                warning="Local unrobust whitened GN; repeated samples/lines correlated, not covariance.")


def fit_lines(records,X0,K,*,mode="full",sigma_px=2.,max_rotation_deg=5.,
              max_translation_m=1.,max_nfev=200,start=None):
    if len(records)<1 or sigma_px<=0 or max_rotation_deg<=0 or max_translation_m<=0:
        raise ValueError("Invalid line fit configuration")
    cols={"x_only":[3],"xz":[3,5],"translation":[3,4,5],"full":list(range(6))}[mode]
    limits=np.r_[np.full(3,np.deg2rad(max_rotation_deg)),np.full(3,max_translation_m)]
    def expand(x):
        v=np.zeros(6);v[cols]=x;return v
    x0=np.zeros(len(cols)) if start is None else np.asarray(start,float)[cols]
    if np.any(abs(x0)>=limits[cols]): raise ValueError("Start outside fixed bounds")
    sol=least_squares(lambda x:residual(perturb(X0,expand(x)),records,K,sigma_px),x0,
                      bounds=(-limits[cols],limits[cols]),loss="soft_l1",x_scale="jac",
                      max_nfev=max_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    X=perturb(X0,expand(sol.x));stats=metrics(X,records,K);info=information(X,records,K,sigma_px)
    return X,dict(solver_converged=bool(sol.success),status=int(sol.status),message=str(sol.message),
        nfev=int(sol.nfev),cost=float(sol.cost),bound_hit=bool(np.any(sol.active_mask)),
        active_components=[cols[i] for i in np.flatnonzero(sol.active_mask)],
        success=bool(sol.success and stats["nonpositive_or_near_samples"]==0),
        train=stats,information=info,
        full_rank=(info["effective_rank"]==6))


def lidar_bev_xy(xyz):
    """Vehicle-oriented display only: horizontal right=-LiDAR Y, vertical forward=X."""
    p=np.asarray(xyz,float)
    if p.ndim!=2 or p.shape[1]!=3:
        raise ValueError("Expected Nx3 LiDAR points")
    return np.column_stack((-p[:,1],p[:,0]))


def polygon_roi_mask(xyz, image_ids, image_uv, polygons):
    """Intersect display-space ROI polygons; retain original LiDAR point identities.

    A BEV polygon is in (-Y, X) display coordinates. Image polygon is in
    original pixel coordinates projected with the frozen initializer.
    Multiple views constrain the SAME original points, never 3D backprojection.
    """
    from matplotlib.path import Path
    p=np.asarray(xyz,float);ids=np.asarray(image_ids);uv=np.asarray(image_uv,float)
    if (p.ndim!=2 or p.shape[1]!=3 or ids.ndim!=1 or ids.dtype.kind not in "iu"
            or uv.shape!=(len(ids),2) or np.any(ids<0) or np.any(ids>=len(p))
            or len(np.unique(ids))!=len(ids)):
        raise ValueError("Invalid linked ROI inputs")
    mask=np.ones(len(p),bool)
    for view,vertices in polygons.items():
        v=np.asarray(vertices,float)
        if v.ndim!=2 or v.shape[1]!=2 or len(v)<3 or not np.isfinite(v).all():
            raise ValueError("Need at least three finite polygon vertices")
        poly=Path(np.vstack((v,v[0])))
        if view=="bev":
            m=poly.contains_points(lidar_bev_xy(p))
        elif view=="image":
            m=np.zeros(len(p),bool);m[ids]=poly.contains_points(uv)
        else:
            raise ValueError("Unknown ROI view")
        mask&=m
    return mask
