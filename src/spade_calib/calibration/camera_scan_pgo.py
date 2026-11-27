"""Fixed scan-registration edge pilot with shared camera-LiDAR extrinsic.

All rotations are world-from-sensor. Edge measurement maps scan j to i.
Registration information is a local, normalized point-plane GN proxy, not
a calibrated covariance. No GT in this module.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from spade_calib.lidar.joint_solver import voxel_sample,normals_for_scan
from spade_calib.calibration.ba_diagnostics import marginal_information
from spade_calib.selection.scale_segmentation import validate_extrinsic


def pose(R,t):
    T=np.zeros((*R.shape[:-2],4,4))
    T[...,3,3]=1;T[...,:3,:3]=R;T[...,:3,3]=t
    return T


def inverse(T):
    R=T[...,:3,:3].swapaxes(-1,-2)
    return pose(R,-np.einsum("...ij,...j->...i",R,T[...,:3,3]))


def relative(C,ij):
    return inverse(C[ij[:,0]])@C[ij[:,1]]


def edge_error(pred,obs):
    # Rotation left tangent, translation additive in target coordinates.
    r=Rotation.from_matrix(pred[:,:3,:3]@obs[:,:3,:3].transpose(0,2,1)).as_rotvec()
    return np.column_stack((r,pred[:,:3,3]-obs[:,:3,3]))


@dataclass(frozen=True)
class RegistrationSettings:
    voxel:float=.18
    samples:int=6000
    k_normals:int=24
    distance:float=.7
    plane:float=.25
    normal_cos:float=.80
    sigma:float=.07
    min_matches:int=80
    min_overlap:float=.10
    iterations:int=20
    max_correction_m:float=1.5
    max_correction_deg:float=5.
    rank_ratio:float=1e-3

    def __post_init__(self):
        if min(self.voxel,self.distance,self.plane,self.sigma,self.max_correction_m,self.max_correction_deg,self.rank_ratio)<=0 or not 0<self.min_overlap<=1 or not 0<self.normal_cos<1 or self.samples<self.k_normals or self.k_normals<4 or self.min_matches<6 or self.iterations<1:
            raise ValueError("Invalid registration settings")


@dataclass
class Scan:
    p:np.ndarray
    n:np.ndarray
    tree:cKDTree


def prepare_scan(xyz,cfg,seed):
    p=np.asarray(xyz,float)
    p=p[voxel_sample(p,cfg.voxel)]
    if len(p)>cfg.samples:
        p=p[np.sort(np.random.default_rng(seed).choice(len(p),cfg.samples,replace=False))]
    n,ok=normals_for_scan(p,cfg.k_normals)
    p=p[ok];n=n[ok]
    return Scan(p,n,cKDTree(p))


def register(target,source,init,cfg=RegistrationSettings()):
    """Iterative point-plane ICP, j->i; reject poor support/correction.
    
    No pose prior in the fit. Rank-deficient directions remain unweighted.
    Matching uses both source/target normal gates. Edge factor is computed
    on the final correspondence set in [left rotation, additive t] chart.
    """
    T=validate_extrinsic(init).copy()
    units=np.r_[np.full(3,np.deg2rad(1)),np.full(3,.1)]
    def matches(T):
        R,t=T[:3,:3],T[:3,3]
        p=source.p@R.T
        dist,idx=target.tree.query(p+t,workers=1)
        n=target.n[idx];q=target.p[idx]
        r=np.sum(n*(p+t-q),axis=1)
        ok=(dist<cfg.distance)&(np.abs(r)<cfg.plane)&(np.abs(np.sum((source.n@R.T)*n,axis=1))>cfg.normal_cos)
        return p[ok],n[ok],r[ok]
    if min(len(target.p),len(source.p))<cfg.min_matches:
        return dict(status="insufficient_normals")
    def factors(T):
        p,n,r=matches(T)
        J=np.column_stack((np.cross(p,n),n))
        w=(1+(r/cfg.sigma)**2)**(-.25)
        return r,J,w
    r0,_,_=factors(T);converged=False
    for it in range(cfg.iterations):
        r,J,w=factors(T)
        if len(r)<cfg.min_matches:return dict(status="insufficient_matches",matches=len(r))
        U,s,V=np.linalg.svd(J*units*w[:,None],full_matrices=False)
        good=s>s[0]*cfg.rank_ratio
        if good.sum()<3:return dict(status="degenerate",rank=int(good.sum()))
        d=units*(-(V[good].T/s[good])@(U[:,good].T@(r*w)))
        factor=min(1.,.20/max(np.linalg.norm(d[3:]),1e-12),np.deg2rad(1)/max(np.linalg.norm(d[:3]),1e-12))
        d*=factor
        T[:3,:3]=Rotation.from_rotvec(d[:3]).as_matrix()@T[:3,:3];T[:3,3]+=d[3:]
        dr=np.rad2deg(Rotation.from_matrix(T[:3,:3]@init[:3,:3].T).magnitude())
        dt=np.linalg.norm(T[:3,3]-init[:3,3])
        if dr>cfg.max_correction_deg or dt>cfg.max_correction_m:
            return dict(status="correction_limit",correction_deg=float(dr),correction_m=float(dt))
        if np.linalg.norm(d/units)<1e-3:
            converged=True;break
    r,J,w=factors(T)
    overlap=len(r)/len(source.p)
    if len(r)<cfg.min_matches or overlap<cfg.min_overlap:
        return dict(status="insufficient_final_overlap",matches=len(r),overlap=overlap)
    if not converged:
        return dict(status="iteration_limit",matches=len(r),overlap=overlap)
    U,s,V=np.linalg.svd(J*units*w[:,None],full_matrices=False)
    good=s>s[0]*cfg.rank_ratio
    if good.sum()<3:return dict(status="degenerate_final",rank=int(good.sum()))
    # Fixed 6 rows, zero weak modes. Equal edge mass, not per-return confidence.
    F=(s*good)[:,None]*V/units/(cfg.sigma*np.sqrt(len(r)))
    return dict(status="ok",T=T.tolist(),factor=F.tolist(),rank=int(good.sum()),
                normalized_singular_values=(s/np.sqrt(len(r))).tolist(),matches=len(r),
                overlap=overlap,initial_matches=len(r0),rmse_m=float(np.sqrt(np.mean(r*r))),
                initial_rmse_m=float(np.sqrt(np.mean(r0*r0))) if len(r0) else None,
                iterations=it+1,correction_m=float(dt),correction_deg=float(dr))


@dataclass
class Graph:
    C0:np.ndarray
    segments:np.ndarray
    anchors:np.ndarray
    camera_edges:np.ndarray
    lidar_edges:np.ndarray
    A:np.ndarray
    F:np.ndarray

    def __post_init__(self):
        n=len(self.C0)
        if self.C0.shape!=(n,4,4) or self.segments.shape!=(n,) or len(self.lidar_edges)<1:
            raise ValueError("Empty/invalid graph")
        for e in (self.camera_edges,self.lidar_edges):
            if e.ndim!=2 or e.shape[1]!=2 or np.any(e<0) or np.any(e>=n) or np.any(e[:,0]>=e[:,1]) or np.any(self.segments[e[:,0]]!=self.segments[e[:,1]]):
                raise ValueError("Invalid graph edges")
        if self.A.shape!=(len(self.lidar_edges),4,4) or self.F.shape!=(len(self.A),6,6):
            raise ValueError("Invalid measurement dimensions")
        if len(np.unique(self.segments[self.anchors]))!=len(self.anchors) or len(self.anchors)!=len(np.unique(self.segments)):
            raise ValueError("Require one fixed anchor per segment")
        self.B=relative(self.C0,self.camera_edges)
        self.free=np.setdiff1d(np.arange(n),self.anchors)


def solve(graph,X0,*,pose_mode="translation",free_X=True,weight=1.,sigma_t=.05,
          sigma_r_deg=.5,max_nfev=100,reference=None):
    """Camera relative-pose tether and fixed LiDAR registration edges.

    Camera node anchor fixed; metric scale held fixed in the SfM tether.
    No explicit scale parameter; free translations can change edge lengths.
    No absolute pose priors, no additional GLIM edge in registration arm.
    """
    X0=validate_extrinsic(X0)
    if pose_mode not in ("fixed","translation","full") or weight<=0 or sigma_t<=0 or sigma_r_deg<=0:
        raise ValueError("Invalid graph settings")
    d={"fixed":0,"translation":3,"full":6}[pose_mode]
    nx=6 if free_X else 0;N=len(graph.free);nvar=nx+d*N
    C0=graph.C0;ce=graph.camera_edges;le=graph.lidar_edges
    sigma=np.r_[np.full(3,np.deg2rad(sigma_r_deg)),np.full(3,sigma_t)]
    wc=1/np.sqrt(len(ce));wl=np.sqrt(weight/len(le))
    def unpack(x):
        X=X0.copy();C=C0.copy()
        if free_X:
            X[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@X0[:3,:3];X[:3,3]+=x[3:6]
        if d:
            q=x[nx:].reshape(N,d)
            C[graph.free,:3,3]+=q[:,-3:]
            if d==6:C[graph.free,:3,:3]=Rotation.from_rotvec(q[:,:3]).as_matrix()@C0[graph.free,:3,:3]
        return X,C
    def blocks(x):
        X,C=unpack(x)
        c=edge_error(relative(C,ce),graph.B)
        l=edge_error(relative(C@X,le),graph.A)
        return c,l
    def raw(x):
        c,l=blocks(x)
        return np.r_[(c/sigma*wc).ravel(),(np.einsum("nij,nj->ni",graph.F,l)*wl).ravel()]
    # Robustification before family normalization: componentwise soft-L1
    # with unit threshold in whitened coordinates, equal weight per edge.
    def fun(x):
        c,l=blocks(x)
        def soft(z):return z*np.sqrt(2/(np.sqrt(1+z*z)+1))
        return np.r_[(soft(c/sigma)*wc).ravel(),
                     (soft(np.einsum("nij,nj->ni",graph.F,l))*wl).ravel()]
    x0=np.zeros(nvar)
    pattern=lil_matrix((6*(len(ce)+len(le)),nvar),dtype=int)
    loc={j:nx+i*d for i,j in enumerate(graph.free)}
    for k,(i,j) in enumerate(np.vstack((ce,le))):
        if free_X and k>=len(ce):pattern[6*k:6*k+6,:6]=1
        if d:
            for v in (i,j):
                if v in loc:pattern[6*k:6*k+6,loc[v]:loc[v]+d]=1
    pattern=pattern.tocsr()
    if nvar:
        lower=np.full(nvar,-np.inf);upper=-lower
        if free_X:
            upper[:6]=np.r_[np.full(3,np.deg2rad(5)),np.ones(3)];lower[:6]=-upper[:6]
        if d:
            upper[nx:]=np.tile(np.r_[np.full(3,np.deg2rad(5)) if d==6 else [],np.full(3,2.)],N)
            lower[nx:]=-upper[nx:]
        kw=dict(jac_sparsity=pattern,method="trf",tr_solver="lsmr",x_scale="jac",
                bounds=(lower,upper),ftol=1e-8,xtol=1e-8,gtol=1e-8)
        init_info=None
        if free_X:
            q=least_squares(raw,x0,max_nfev=1,**kw)
            init_info=marginal_information(q.jac,x0[:3])
        sol=least_squares(fun,x0,max_nfev=max_nfev,**kw)
        x=sol.x;X,C=unpack(x)
        info=None
        if free_X:
            q=least_squares(raw,x,max_nfev=1,**kw)
            info=marginal_information(q.jac,x[:3])
        status=dict(success=bool(sol.success),message=str(sol.message),nfev=int(sol.nfev),
                    active_count=int(np.count_nonzero(sol.active_mask)),optimality=float(sol.optimality),
                    initial_information=init_info,final_information=info)
    else:
        x=x0;X,C=unpack(x);status=dict(success=True,nfev=0,active_count=0)
    c,l=blocks(x)
    rms=lambda a:float(np.sqrt(np.mean(np.sum(a*a,axis=1))))
    status.update(cost=.5*float(fun(x)@fun(x)),initial_cost=.5*float(fun(x0)@fun(x0)),
                  camera_t_residual_rmse_m=rms(c[:,3:]),lidar_t_residual_rmse_m=rms(l[:,3:]),
                  camera_r_residual_rmse_deg=float(np.rad2deg(rms(c[:,:3]))),
                  lidar_r_residual_rmse_deg=float(np.rad2deg(rms(l[:,:3]))),
                  pose_shift_rmse_m=rms(C[:,:3,3]-C0[:,:3,3]),T_camera_lidar=X.tolist(),
                  variable_count=nvar)
    return status,C
