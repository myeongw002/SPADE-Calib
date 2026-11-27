"""Segment scene-consistency diagnostics with fixed GLIM and SfM poses.

All point correspondences are identical raw LiDAR return IDs. No new
cross-modal measurements or ICP reassociation. Compact square-root moment
factors preserve each center's mean squared spatial error exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from spade_calib.calibration.segmented_ground import _robust, ground_residuals
from spade_calib.handeye.residuals import residuals
from spade_calib.selection.trajectory import MotionPairs
from spade_calib.selection.scale_segmentation import validate_extrinsic


def predicted_relative(Rb,tb,X,scale):
    """X^-1 B(s) X in the segment anchor LiDAR frame."""
    R,t=X[:3,:3],X[:3,3]
    return R.T@Rb@R, (Rb@t+scale*tb-t)@R


def sqrt_factor(H):
    e,V=np.linalg.eigh((H+H.T)*.5)
    if e[0]<-1e-9*max(1.,e[-1]):
        raise ValueError("Non-PSD scene moment")
    return np.sqrt(np.maximum(e,0))[:,None]*V.T


def scene_factors(q,n):
    """q: center-LiDAR points, n: frozen normals in anchor LiDAR axes.

    F @ vec(D) yields a compact residual, D is row-major 3x4 difference
    between predicted and GLIM relative transforms. Squared norm equals
    the mean of squared point distances (or normal projections).
    """
    q=np.asarray(q,dtype=float); n=np.asarray(n,dtype=float)
    if (q.ndim!=2 or q.shape[1]!=3 or n.shape!=q.shape or len(q)<3
            or not np.isfinite(q).all() or not np.isfinite(n).all()
            or not np.allclose(np.linalg.norm(n,axis=1),1.,atol=1e-6)):
        raise ValueError("Invalid point/normal support")
    h=np.column_stack((q,np.ones(len(q))))
    D=np.einsum("ni,nj->nij",n,h).reshape(-1,12)
    return sqrt_factor(np.kron(np.eye(3),h.T@h/len(h))),sqrt_factor(D.T@D/len(D))


def voxel_return_indices(q, voxel):
    """One actual return nearest each occupied voxel's centroid.

    Grid is in the fixed center LiDAR frame. No pre-sampling, no point cap.
    Choosing a measured return rather than a synthetic centroid preserves
    exact (scan frame, return index) identity for both reconstructed maps.
    """
    q=np.asarray(q,dtype=float)
    if q.ndim!=2 or q.shape[1]!=3 or not np.isfinite(q).all() or not np.isfinite(voxel) or voxel<=0:
        raise ValueError("Invalid voxel support")
    if not len(q):
        return np.empty(0,dtype=np.int64)
    _,inv=np.unique(np.floor(q/voxel).astype(np.int64),axis=0,return_inverse=True)
    inv=inv.reshape(-1)
    count=np.bincount(inv)
    means=np.column_stack([np.bincount(inv,weights=q[:,d])/count for d in range(3)])
    dist=np.sum((q-means[inv])**2,axis=1)
    best=np.full(len(count),np.inf)
    np.minimum.at(best,inv,dist)
    ids=np.arange(len(q),dtype=np.int64)
    winners=np.full(len(count),len(q),dtype=np.int64)
    np.minimum.at(winners,inv,np.where(dist==best[inv],ids,len(q)))
    return winners


@dataclass
class SceneSegment:
    segment_id: int
    frames: np.ndarray
    Ra: np.ndarray
    ta: np.ndarray
    Rb: np.ndarray
    tb: np.ndarray
    q: list
    normals: list
    source_frames: list
    compact_factors: list | None = None
    point_counts: np.ndarray | None = None

    def __post_init__(self):
        n=len(self.frames)
        if n<3 or any(len(a)!=n for a in (self.Ra,self.ta,self.Rb,self.tb)):
            raise ValueError("Insufficient or inconsistent center support")
        if self.compact_factors is None:
            if any(len(a)!=n for a in (self.q,self.normals,self.source_frames)):
                raise ValueError("Insufficient or inconsistent point support")
            factors=[scene_factors(q,v) for q,v in zip(self.q,self.normals)]
            self.point_counts=np.array([len(q) for q in self.q])
        else:
            factors=self.compact_factors
            self.point_counts=np.asarray(self.point_counts)
            if len(factors)!=n or self.point_counts.shape!=(n,) or np.any(self.point_counts<3):
                raise ValueError("Invalid compact support")
        if self.q or self.source_frames:
            if len(self.q)!=n or len(self.source_frames)!=n or any(
                len(q)!=len(f) or len(q)!=c for q,f,c in zip(self.q,self.source_frames,self.point_counts)):
                raise ValueError("Source scan identity mismatch")
        self.point_factors=np.stack([f[0] for f in factors])
        self.plane_factors=np.stack([f[1] for f in factors])
        if any(a.shape!=(n,12,12) or not np.isfinite(a).all() for a in (self.point_factors,self.plane_factors)):
            raise ValueError("Invalid scene factors")
        self.compact_factors=None
        self.pairs=MotionPairs(self.Ra,self.ta,self.Rb,self.tb,
                              np.column_stack((np.arange(n),np.arange(n)+1)),
                              np.arange(n,dtype=np.int64))

    def difference(self,X,s):
        R,t=predicted_relative(self.Rb,self.tb,X,s)
        return np.concatenate((R-self.Ra,(t-self.ta)[:,:,None]),axis=2).reshape(-1,12)

    def raw_blocks(self,X,s,kind):
        if kind=="motion":
            return residuals(self.pairs,X[:3,:3],X[:3,3],float(s))
        if kind not in ("point","plane"):
            raise ValueError("Unknown scene residual")
        F=self.point_factors if kind=="point" else self.plane_factors
        return np.einsum("nij,nj->ni",F,self.difference(X,s))

    def maps(self,X,s):
        if not self.q:
            raise ValueError("Point support not retained; enable map export")
        R,t=predicted_relative(self.Rb,self.tb,X,s)
        reference=np.concatenate([q@a.T+b for q,a,b in zip(self.q,self.Ra,self.ta)])
        estimate=np.concatenate([q@a.T+b for q,a,b in zip(self.q,R,t)])
        return reference,estimate


def fit_scene(segments,X_he,scales,he,*,kind="point",refit_scales=False,
              ground_matches=(),ground_weight=0.,sigma_scene=.15,sigma_ground=.15,
              max_rotation_deg=5.,max_translation_m=1.,max_log_scale=.15,
              initial_X=None,linear=False):
    """Shared X, fixed poses, optional per-segment positive scale adjustment.

    Equal weight per segment and center, robustify center RMS before weighting.
    Ground uses fixed B4 support with offset rescaled consistently.
    """
    X0=validate_extrinsic(X_he)
    s0=np.asarray(scales,dtype=float)
    K=len(s0)
    if (K<1 or not np.isfinite(s0).all() or np.any(s0<=0)
            or sorted(seg.segment_id for seg in segments)!=list(range(K))
            or kind not in ("motion","point","plane")):
        raise ValueError("Require all saved segments and positive scales")
    if (not np.isfinite([ground_weight,sigma_scene,sigma_ground,max_rotation_deg,max_translation_m,max_log_scale]).all()
            or ground_weight<0 or min(sigma_scene,sigma_ground,max_rotation_deg,max_translation_m,max_log_scale)<=0
            or max_rotation_deg>=90 or (ground_weight>0 and not ground_matches)):
        raise ValueError("Invalid scene settings")
    dim=6+(K if refit_scales else 0)
    def state(x):
        X=X0.copy()
        X[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@X0[:3,:3]
        X[:3,3]+=x[3:6]
        s=s0*np.exp(x[6:]) if refit_scales else s0
        return X,s
    def objective(x,robust=True):
        X,s=state(x)
        out=[]
        for seg in segments:
            r=seg.raw_blocks(X,s[seg.segment_id],kind)
            if kind=="motion":
                r=r/np.r_[np.full(3,np.deg2rad(he.sigma_relative_rotation_deg)),
                           np.full(3,he.sigma_relative_translation_m)]
                if robust and not linear:
                    r=_robust(r,he.robust_loss)
            else:
                r=r/sigma_scene
                if robust and not linear:
                    # Block soft-L1 of RMS displacement per camera center.
                    z=np.linalg.norm(r,axis=1)
                    r=r*np.sqrt(2/(np.hypot(1,z)+1))[:,None]
            out.append(r.ravel()/np.sqrt(K*len(r)))
        if ground_weight>0:
            g=ground_residuals(ground_matches,X,s,s0)[:,3]/sigma_ground
            if robust and not linear:
                g=_robust(g,"soft_l1")
            out.append(g*np.sqrt(ground_weight/len(g)))
        return np.concatenate(out)
    def diagnostics(x):
        X,s=state(x)
        rows=[]
        for seg in segments:
            k=seg.segment_id
            p=seg.raw_blocks(X,s[k],"point");n=seg.raw_blocks(X,s[k],"plane")
            m=seg.raw_blocks(X,s[k],"motion")
            rows.append(dict(segment_id=k,centers=len(seg.frames),
                             sampled_returns=int(np.sum(seg.point_counts)),
                             point_rms_m=float(np.sqrt(np.mean(np.sum(p*p,axis=1)))),
                             plane_rms_m=float(np.sqrt(np.mean(np.sum(n*n,axis=1)))),
                             motion_translation_rms_m=float(np.sqrt(np.mean(np.sum(m[:,3:]**2,axis=1)))),
                             motion_rotation_rms_deg=float(np.rad2deg(np.sqrt(np.mean(np.sum(m[:,:3]**2,axis=1)))))))
        g=ground_residuals(ground_matches,X,s,s0)[:,3]
        return dict(T_camera_lidar=X.tolist(),scales=s.tolist(),segments=rows,
                    cost=float(.5*np.sum(objective(x)**2)),
                    ground_offset_m=g.tolist())
    def observability(x):
        units=np.r_[np.full(3,np.deg2rad(he.reference_rotation_deg)),
                    np.full(3,he.reference_translation_m),
                    np.full(dim-6,he.reference_log_scale)]
        J=np.column_stack([(objective(x+d,False)-objective(x-d,False))/2e-5
                           for d in np.eye(dim)*units*1e-5])
        sv=np.linalg.svd(J,compute_uv=False)
        nuisance=J[:,np.r_[0:3,6:dim]]
        U,S,_=np.linalg.svd(nuisance,full_matrices=False)
        U=U[:,S>max(S[0],1e-15)*he.rank_sv_ratio]
        T=J[:,3:6]-U@(U.T@J[:,3:6])
        _,st,vh=np.linalg.svd(T,full_matrices=False)
        rank=int(np.sum(sv>max(sv[0],1e-15)*he.rank_sv_ratio))
        return dict(rank=rank,full_rank=rank==dim,singular_values=sv.tolist(),
                    translation_marginal_singular_values=st.tolist(),
                    weakest_translation_camera_xyz=vh[-1].tolist())
    bounds=np.r_[np.full(3,np.deg2rad(max_rotation_deg)),np.full(3,max_translation_m),
                 np.full(dim-6,max_log_scale)]
    zero=np.zeros(dim); x0=zero.copy()
    if initial_X is not None:
        Y=validate_extrinsic(initial_X)
        x0[:3]=Rotation.from_matrix(Y[:3,:3]@X0[:3,:3].T).as_rotvec()
        x0[3:6]=Y[:3,3]-X0[:3,3]
        if np.any(abs(x0)>=bounds):
            return dict(success=False,reason="initializer_outside_bounds")
    pre=observability(x0)
    if not pre["full_rank"]:
        return dict(success=False,reason="rank_deficient",initial=diagnostics(x0),
                    initial_observability=pre)
    sol=least_squares(objective,x0,bounds=(-bounds,bounds),loss="linear",x_scale="jac",
                      max_nfev=he.max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    post=observability(sol.x)
    return dict(success=bool(sol.success and post["full_rank"]),solver_converged=bool(sol.success),
                message=str(sol.message),nfev=int(sol.nfev),bound_hit=bool(np.any(sol.active_mask)),
                kind=kind,refit_scales=refit_scales,ground_weight=ground_weight,
                sigma_scene_m=sigma_scene,sigma_ground_m=sigma_ground,linear=linear,
                baseline=diagnostics(zero),initial=diagnostics(x0),final=diagnostics(sol.x),
                initial_observability=pre,final_observability=post,
                warning="Unrobustified scaled Jacobian, not covariance. Same raw LiDAR measurements reused.")
