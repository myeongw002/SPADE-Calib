"""Fixed-support image-edge distance objective for extrinsic-only refinement."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

MODES={"xz":(3,5),"t":(3,4,5),"se3":(0,1,2,3,4,5)}


def edge_distance_field(uv,shape):
    """EP0 jump coordinates are integer image pixel indices."""
    u=np.asarray(uv,float);h,w=shape
    if (u.ndim!=2 or u.shape[1]!=2 or not len(u) or not np.isfinite(u).all()
        or min(h,w)<2 or not np.allclose(u,np.rint(u),atol=1e-7,rtol=0)
        or np.any(u<0) or np.any(u>=np.array([w,h]))):
        raise ValueError("Expected nonempty integer EP0 edge coordinates")
    xy=np.rint(u).astype(int)
    mask=np.ones((h,w),bool);mask[xy[:,1],xy[:,0]]=False
    return distance_transform_edt(mask)


def transform_from_delta(X0,delta):
    """Rotation increment in camera axes (degrees); additive camera t (m).
    This is a valid rigid transform chart, not a coupled SE(3) exponential.
    """
    X=np.asarray(X0,float).copy();d=np.asarray(delta,float)
    if X.shape!=(4,4) or d.shape!=(6,) or not np.isfinite(d).all():
        raise ValueError("Invalid transform/delta")
    X[:3,:3]=Rotation.from_rotvec(np.deg2rad(d[:3])).as_matrix()@X[:3,:3]
    X[:3,3]+=d[3:]
    return X


def delta_from_transform(X0,X):
    return np.r_[np.rad2deg(Rotation.from_matrix(X[:3,:3]@X0[:3,:3].T).as_rotvec()),
                 X[:3,3]-X0[:3,3]]


def extrinsic_error(X,gt):
    d=X[:3,3]-gt[:3,3]
    return dict(rotation_deg=float(np.rad2deg(Rotation.from_matrix(X[:3,:3]@gt[:3,:3].T).magnitude())),
                translation_m=float(np.linalg.norm(d)),translation_xyz_m=d.tolist())


@dataclass(frozen=True)
class Frame:
    frame: int
    xyz: np.ndarray
    distance: np.ndarray


def residual_frame(frame,X,K,max_depth=40.):
    """One residual per fixed source point, including invalid projections.
    Bilinear distance field at continuous pixel coordinates. Outside image:
    smooth positive overflow penalty plus boundary distance; out-of-depth
    gets an additional metric overflow penalty in pixel units. Never drops
    or re-z-buffers points while fitting.
    """
    q=frame.xyz@X[:3,:3].T+X[:3,3];z=q[:,2]
    zsafe=np.maximum(z,.5)
    uv=(q@K.T)[:,:2]/zsafe[:,None]
    h,w=frame.distance.shape
    clipped=np.clip(uv,[0.,0.],[w-1.,h-1.])
    dist=map_coordinates(frame.distance,[clipped[:,1],clipped[:,0]],
                         order=1,mode="nearest",prefilter=False)
    overflow=np.linalg.norm(uv-clipped,axis=1)
    dz=np.maximum(.5-z,0.)+np.maximum(z-max_depth,0.)
    # Continuous extensions. Heuristic penalties, not physical measurements.
    residual=dist+2.*overflow+25.*(-np.expm1(-overflow/2.))
    residual+=50.*dz+25.*(-np.expm1(-dz/.1))
    valid=(z>=.5)&(z<=max_depth)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    return residual,valid,uv


def robust_values(r,scale):
    """SciPy soft_l1 cost per residual (twice least_squares.cost)."""
    r=np.asarray(r,float)
    return 2.*r*r/(np.hypot(1.,r/scale)+1.)


def residuals(frames,X,K,max_depth):
    return np.concatenate([residual_frame(f,X,K,max_depth)[0] for f in frames])


def evaluate(frames,X,K,max_depth,scale):
    per={};rs=[];vs=[]
    for f in frames:
        r,v,_=residual_frame(f,X,K,max_depth);rs.append(r);vs.append(v)
        per[str(f.frame)]=_metrics(r,v,scale)
    if not rs:return None
    return dict(**_metrics(np.concatenate(rs),np.concatenate(vs),scale),frames=per)


def _metrics(r,v,scale):
    n=len(r)
    if not n:return dict(n=0,valid=0,invalid=0,cost=None,median_px=None,rmse_px=None,within5=0,fraction5=None)
    hits=int(((r<=5)&v).sum())
    return dict(n=n,valid=int(v.sum()),invalid=int((~v).sum()),
                cost=float(robust_values(r,scale).mean()),median_px=float(np.median(r)),
                rmse_px=float(np.sqrt(np.mean(r*r))),within5=hits,fraction5=hits/n)


def fit(frames,X0,K,mode,start,*,max_depth=40.,scale=5.,rot_bound=2.,
        trans_bound=1.,max_nfev=150):
    active=np.asarray(MODES[mode]);start=np.asarray(start,float)
    if not frames or sum(len(f.xyz) for f in frames)<len(active):
        raise ValueError("Insufficient fixed support")
    if (start.shape!=(6,) or not np.isfinite(start).all()
        or not np.isfinite([max_depth,scale,rot_bound,trans_bound]).all()
        or max_depth<=.5 or min(scale,rot_bound,trans_bound)<=0 or max_nfev<1):
        raise ValueError("Invalid fit settings")
    bound=np.r_[np.full(3,rot_bound),np.full(3,trans_bound)][active]
    if np.any(abs(start[active])>bound+1e-10):raise ValueError("Start outside B4c-centered bounds")
    def expand(x):
        d=np.zeros(6);d[active]=x;return d
    def fun(x):
        return residuals(frames,transform_from_delta(X0,expand(x)),K,max_depth)
    # Absolute central difference avoids near-zero parameter relative-step issues.
    step=np.where(active<3,1e-3,1e-4)
    def jac(x):
        return np.column_stack([(fun(x+np.eye(len(x))[j]*h)-fun(x-np.eye(len(x))[j]*h))/(2*h)
                                for j,h in enumerate(step)])
    sol=least_squares(fun,np.clip(start[active],-bound,bound),jac=jac,
                      bounds=(-bound,bound),method="trf",loss="soft_l1",f_scale=scale,
                      x_scale=np.where(active<3,.5,.2),max_nfev=max_nfev,
                      ftol=1e-8,xtol=1e-8,gtol=1e-6)
    d=expand(sol.x);X=transform_from_delta(X0,d)
    # Singular values in declared scaled parameter units, not a guarantee of
    # global observability or correct data associations.
    sv=np.linalg.svd(sol.jac*np.where(active<3,.5,.2),compute_uv=False)
    return X,dict(success=bool(sol.success),status=int(sol.status),message=str(sol.message),
                  nfev=int(sol.nfev),njev=int(sol.njev) if sol.njev is not None else None,
                  optimality=float(sol.optimality),delta_deg_m=d.tolist(),
                  active_mask=sol.active_mask.tolist(),scaled_robust_jacobian_sv=sv.tolist(),
                  scipy_cost=float(sol.cost))
