"""P1: piecewise log-linear scale on adjacent SfM displacements.

Original rotations, segment boundaries and selected motion pairs stay fixed.
Midpoint quadrature on the actual matched timestamp grid; never multiply a
long-baseline displacement by the scale at its endpoint.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import csr_matrix
from spade_calib.selection.scale_segmentation import validate_extrinsic
from spade_calib.calibration.ba_diagnostics import marginal_information
from spade_calib.handeye.trajectory_oracle import sqrt_weights


class ScaleMotion:
    def __init__(self,samples,pairs,partition):
        self.pairs=pairs
        self.groups=partition.groups(pairs)
        if np.any(self.groups<0):raise ValueError("P1 expects within-segment pairs only")
        self.n=partition.count
        self.R=samples.camera_R_world_sfm
        self.p=samples.camera_t_world_sfm
        self.dp=np.diff(self.p,axis=0)
        self.tau=np.zeros(len(self.dp))
        self.edge_group=np.full(len(self.dp),-1,int)
        self.intervals=partition.intervals
        self.i,self.j=pairs.indices.T
        time=(samples.timestamps_ns-samples.timestamps_ns[0]).astype(float)
        for k,(a,b) in enumerate(self.intervals):
            if b-a<2:raise ValueError("Segment too short")
            self.tau[a:b-1]=(time[a:b-1]+time[a+1:b]-time[a]-time[b-1])/(time[b-1]-time[a])
            self.edge_group[a:b-1]=k
        tb=self.metric_tb(np.zeros(self.n),np.zeros(self.n))
        if not np.allclose(tb,pairs.tb,rtol=1e-8,atol=1e-9):
            raise ValueError("Integrated unit scale does not reproduce original SfM motion")

    def metric_tb(self,a,b):
        a=np.asarray(a,float);b=np.asarray(b,float)
        if a.shape!=(self.n,) or b.shape!=(self.n,) or not np.isfinite(np.r_[a,b]).all():
            raise ValueError("Invalid scale coefficients")
        out=np.empty_like(self.pairs.tb)
        for k,(lo,hi) in enumerate(self.intervals):
            idx=self.groups==k
            w=np.exp(a[k]+b[k]*self.tau[lo:hi-1])
            local=np.vstack((np.zeros((1,3)),np.cumsum(self.dp[lo:hi-1]*w[:,None],axis=0)))
            delta=local[self.j[idx]-lo]-local[self.i[idx]-lo]
            out[idx]=np.einsum("nji,nj->ni",self.R[self.i[idx]],delta)
        return out

    def raw(self,X,a,b):
        p=self.pairs
        R=X[:3,:3];t=X[:3,3]
        left=p.Rb@R;right=R@p.Ra
        rr=Rotation.from_matrix(left.transpose(0,2,1)@right).as_rotvec()
        rt=(p.Rb-np.eye(3))@t+self.metric_tb(a,b)-p.ta@R.T
        return np.column_stack((rr,rt))


def blocked_masks(model):
    """Central half train; outer quarters test. No shared displacement edges."""
    train=np.zeros(model.pairs.count,bool);test=train.copy()
    for k,(a,b) in enumerate(model.intervals):
        lo=a+(b-a)//4;hi=b-(b-a)//4
        g=model.groups==k;i,j=model.i,model.j
        train|=g&(i>=lo)&(j<hi)
        test|=g&((j<lo)|(i>=hi))
    return train,test


def fit(model,X0,scales,he,mask,*,linear=False,slope_bound=.25):
    X0=validate_extrinsic(X0);scales=np.asarray(scales,float)
    n=model.n;mask=np.asarray(mask)
    if (scales.shape!=(n,) or np.any(scales<=0) or not np.isfinite(scales).all()
            or mask.dtype!=np.bool_ or mask.shape!=(model.pairs.count,)
            or any(np.count_nonzero(mask&(model.groups==k))<3 for k in range(n))
            or not np.isfinite(slope_bound) or slope_bound<=0):
        raise ValueError("Invalid P1 support or initialization")
    start=np.r_[np.zeros(6),np.log(scales),np.zeros(n) if linear else []]
    lo=np.r_[np.full(6,-np.inf),np.full(n,-30.),np.full(n,-slope_bound) if linear else []]
    hi=np.r_[np.full(6,np.inf),np.full(n,30.),np.full(n,slope_bound) if linear else []]
    sigma=np.r_[np.full(3,np.deg2rad(he.sigma_relative_rotation_deg)),np.full(3,he.sigma_relative_translation_m)]
    units=np.r_[np.full(3,np.deg2rad(he.reference_rotation_deg)),np.full(3,he.reference_translation_m),
                np.full(n*(2 if linear else 1),he.reference_log_scale)]
    def unpack(x):
        X=X0.copy();X[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@X0[:3,:3];X[:3,3]+=x[3:6]
        return X,x[6:6+n],x[6+n:] if linear else np.zeros(n)
    def fun(x):return (model.raw(*unpack(x))[mask]/sigma).ravel()
    sol=least_squares(fun,start,bounds=(lo,hi),method="trf",loss=he.robust_loss,
                      x_scale="jac",max_nfev=he.max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    X,a,b=unpack(sol.x)
    def info(x):
        # Raw FD derivatives in physical parameter chart; normalize for rank.
        J=np.column_stack([(fun(x+d)-fun(x-d))/2e-5 for d in np.eye(len(x))*1e-5])
        out={}
        for name,w in (("raw",np.ones(len(J))),("frozen_IRLS",sqrt_weights(fun(x),he.robust_loss))):
            M=J*w[:,None];Z=M*units
            sv=np.linalg.svd(Z,compute_uv=False)
            d=dict(rank=int(np.count_nonzero(sv>sv[0]*he.rank_sv_ratio)),singular_values=sv.tolist(),
                   extrinsic=marginal_information(csr_matrix(M),x[:3]))
            if linear:
                A=Z[:,:6+n];U,S,_=np.linalg.svd(A,full_matrices=False)
                U=U[:,S>S[0]*he.rank_sv_ratio]
                B=Z[:,6+n:];E=B-U@(U.T@B)
                s=np.linalg.svd(E,compute_uv=False)
                d["slope_marginal_singular_values"]=s.tolist()
                d["slope_rank_common_floor"]=int(np.count_nonzero(s>sv[0]*he.rank_sv_ratio))
                d["slope_information_fraction_by_segment"]=(np.sum(E*E,axis=0)/np.maximum(np.sum(B*B,axis=0),1e-30)).tolist()
            out[name]=d
        return out
    return dict(success=bool(sol.success),message=str(sol.message),nfev=int(sol.nfev),optimality=float(sol.optimality),
                cost=float(sol.cost),T_camera_lidar=X.tolist(),scales_mid=np.exp(a).tolist(),log_slopes=b.tolist(),
                scale_start=np.exp(a-b).tolist(),scale_end=np.exp(a+b).tolist(),
                active_indices=np.flatnonzero(sol.active_mask).tolist(),variables=len(start),
                same_state_information=info(start),solution_information=info(sol.x),
                warning="Local rank and frozen IRLS diagnostics, not covariance. Pair rows overlap. Slope bounds numerical/model guard, not evidence.")


def evaluate(model,result,mask,gt=None):
    X=np.asarray(result["T_camera_lidar"]);a=np.log(result["scales_mid"]);b=np.asarray(result["log_slopes"])
    r=model.raw(X,a,b)[mask]
    rms=lambda x:float(np.sqrt(np.mean(np.sum(x*x,axis=1)))) if len(x) else None
    out=dict(count=len(r),motion_translation_rmse_m=rms(r[:,3:]),motion_rotation_rmse_rad=rms(r[:,:3]))
    if gt is not None and len(r):
        tb=model.metric_tb(a,b)[mask];target=gt.tb[mask]
        err=tb-target
        norm=np.linalg.norm(target,axis=1);est=np.linalg.norm(tb,axis=1);good=(norm>.1)&(est>.1)
        out.update(camera_translation_rpe_rmse_m=rms(err),
                   camera_length_rmse_m=float(np.sqrt(np.mean((est-norm)**2))),
                   camera_log_length_rmse_pct=float(100*np.sqrt(np.mean(np.log(est[good]/norm[good])**2))) if np.any(good) else None,
                   segments=[dict(segment=k,count=int(np.count_nonzero(mask&(model.groups==k))),
                                  camera_translation_rpe_rmse_m=rms(err[model.groups[mask]==k]))
                             for k in range(model.n)])
    return out
