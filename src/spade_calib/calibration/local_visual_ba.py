"""Local visual BA with shared extrinsic and independent segment scales.

Coordinates in each anchor camera frame, metric-normalized by archived scale.
Anchor pose fixed, one nonzero baseline component fixed (7-DoF gauge).
GLIM fixed; scene factors depend on GLIM geometry only.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from spade_calib.io.colmap_observations import project
from spade_calib.calibration.scene_consistency import predicted_relative
from spade_calib.calibration.segmented_ground import _robust
from spade_calib.calibration.ba_diagnostics import marginal_information
from spade_calib.calibration.crossmodal_lines import residual as line_residual, metrics as line_metrics


@dataclass
class LocalBA:
    frames: np.ndarray
    anchor: int
    R: np.ndarray
    t: np.ndarray
    points: np.ndarray
    point_ids: np.ndarray
    obs_cam: np.ndarray
    obs_pt: np.ndarray
    uv: np.ndarray
    intr: np.ndarray
    centers: np.ndarray
    Ra: np.ndarray
    ta: np.ndarray
    factors: np.ndarray
    ground_ids: np.ndarray
    ground_normal: np.ndarray
    ground_offset: float
    lidar_normal: np.ndarray
    lidar_offset: float

    def __post_init__(self):
        n=len(self.frames);m=len(self.points)
        if n<4 or m<10 or len(self.centers)!=n-1 or set(self.centers)!=set(range(n))-{self.anchor}:
            raise ValueError("Invalid BA camera support")
        if not np.allclose(self.R[self.anchor],np.eye(3),atol=1e-7) or np.linalg.norm(self.t[self.anchor])>1e-7:
            raise ValueError("BA requires anchor-local coordinates")
        if len(self.ground_ids)<6 or len(np.unique(self.ground_ids))!=len(self.ground_ids):
            raise ValueError("Insufficient ground landmark tracks")
        # Gauge: fix all six anchor increments and one baseline translation
        # component. Metric scale remains a separate explicit variable.
        j,d=np.unravel_index(np.argmax(abs(self.t)),self.t.shape)
        if abs(self.t[j,d])<.1 or j==self.anchor: raise ValueError("Weak scale gauge")
        self.gauge=(int(j),int(d))
        self.pose_cols=np.full((n,6),-1,dtype=int)
        mask=np.ones((n,6),bool);mask[self.anchor]=False;mask[j,3+d]=False
        self.pose_cols[mask]=np.arange(mask.sum())
        self.npose=int(mask.sum())
        self.pt_cols=np.arange(self.npose,self.npose+3*m).reshape(m,3)
        self.size=self.npose+3*m
        # Connected observation graph required; all landmarks at least 2 views.
        if np.any(np.bincount(self.obs_pt,minlength=m)<2): raise ValueError("Singleton BA landmark")
        if np.any(np.bincount(self.obs_cam,minlength=n)<12): raise ValueError("Insufficient observations per camera")
        connected={self.anchor}
        for _ in range(n):
            ps=np.unique(self.obs_pt[np.isin(self.obs_cam,list(connected))])
            new=set(self.obs_cam[np.isin(self.obs_pt,ps)])
            if new==connected: break
            connected=new
        if len(connected)!=n: raise ValueError("Disconnected BA camera graph")

    def state(self,x):
        d=np.zeros_like(self.pose_cols,dtype=float);mask=self.pose_cols>=0
        d[mask]=x[self.pose_cols[mask]]
        R=Rotation.from_rotvec(d[:,:3]).as_matrix()@self.R
        t=self.t+d[:,3:]
        P=self.points+x[self.pt_cols]
        return R,t,P

    def visual(self,R,t,P):
        q=np.einsum("ni,nij->nj",P[self.obs_pt]-t[self.obs_cam],R[self.obs_cam])
        r=project(q,self.intr[self.obs_cam])-self.uv
        return r,q[:,2]

    def ground(self,P,X,ratio):
        # Preserve B4 signed offset at initializer, update it by displacement
        # of fixed, observed ground landmarks along frozen camera normal.
        delta=(P[self.ground_ids]-self.points[self.ground_ids]).mean(axis=0)
        d=self.ground_offset-self.ground_normal@delta
        return float(ratio*d-self.lidar_offset+(X[:3,:3]@self.lidar_normal)@X[:3,3])

    def motion(self,R,t,X,ratio,kind,he,rotation_multiplier=1.):
        B=R[self.centers];b=t[self.centers]
        if kind=="he":
            left=B@X[:3,:3];right=X[:3,:3]@self.Ra
            rot=Rotation.from_matrix(left.transpose(0,2,1)@right).as_rotvec()
            tr=(B-np.eye(3))@X[:3,3]+ratio*b-self.ta@X[:3,:3].T
            return np.column_stack((rotation_multiplier*rot/np.deg2rad(he.sigma_relative_rotation_deg),
                                    tr/he.sigma_relative_translation_m))
        r,u=predicted_relative(B,b,X,ratio)
        D=np.concatenate((rotation_multiplier*(r-self.Ra),(u-self.ta)[:,:,None]),axis=2).reshape(-1,12)
        return np.einsum("nij,nj->ni",self.factors,D)


def solve_local_ba(segments,X0,scales,he,*,kind="visual",fixed_local=None,
                   visual_weight=1.,motion_weight=1.,ground_weight=.1,
                   sigma_px=1.,sigma_scene=.15,sigma_ground=.15,
                   max_nfev=80,pose_rotation_deg=5.,pose_translation_m=2.,
                   max_log_scale=.15,rotation_multiplier=1.,he_block_robust=False,
                   information=False,information_maxiter=10000,
                   line_records=None,line_K=None,line_weight=0.,line_sigma_px=2.):
    """Common initial state for H0/H1/H2; sparse FD + LSMR, no pose priors.

    Per-segment mean visual and motion losses, robustified before weights.
    fixed_local: optional list of frozen local parameter vectors for controls.
    """
    if kind not in ("visual","he","scene") or (kind=="visual" and fixed_local is not None):
        raise ValueError("Invalid BA mode")
    K=len(segments);joint=kind!="visual";free=fixed_local is None
    if K<1 or len(scales)!=K or not np.isfinite(scales).all() or np.any(np.asarray(scales)<=0):
        raise ValueError("Invalid segment scales")
    if not np.isfinite([visual_weight,motion_weight,ground_weight,sigma_px,sigma_scene,sigma_ground,pose_rotation_deg,pose_translation_m,max_log_scale]).all() or min(visual_weight,motion_weight,sigma_px,sigma_scene,sigma_ground,pose_rotation_deg,pose_translation_m,max_log_scale)<=0 or ground_weight<0:
        raise ValueError("Invalid BA settings")
    if not np.isfinite(rotation_multiplier) or rotation_multiplier<=0:
        raise ValueError("Invalid rotation multiplier")
    if not np.isfinite([line_weight,line_sigma_px]).all() or line_weight<0 or line_sigma_px<=0:
        raise ValueError("Invalid line settings")
    use_lines=line_weight>0
    if use_lines:
        if not joint or not line_records or line_K is None:
            raise ValueError("Positive line weight requires motion BA and actual line records/intrinsics")
        line_K=np.asarray(line_K,float)
        if (line_K.shape!=(3,3) or not np.isfinite(line_K).all()
                or min(line_K[0,0],line_K[1,1])<=0 or not np.allclose(line_K[2],[0,0,1])):
            raise ValueError("Invalid rectified line intrinsics")
        initial_line=line_metrics(X0,line_records,line_K)
        if initial_line["nonpositive_or_near_samples"]:
            raise ValueError("Line support behind/too near camera at initial state")
    base=6+K if joint else 0
    offsets=np.r_[base,base+np.cumsum([s.size if free else 0 for s in segments])]
    dim=int(offsets[-1]);lo=np.full(dim,-np.inf);hi=np.full(dim,np.inf)
    if joint:
        hi[:base]=np.r_[np.full(3,np.deg2rad(5.)),np.full(3,1.),np.full(K,max_log_scale)]
        lo[:base]=-hi[:base]
    for k,s in enumerate(segments):
        if free:
            cols=s.pose_cols[s.pose_cols>=0]+offsets[k]
            v=np.broadcast_to(np.r_[np.full(3,np.deg2rad(pose_rotation_deg)),np.full(3,pose_translation_m)],s.pose_cols.shape)
            hi[cols]=v[s.pose_cols>=0];lo[cols]=-hi[cols]
    def unpack(x):
        X=np.array(X0,copy=True);ratio=np.ones(K)
        if joint:
            X[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@X0[:3,:3]
            X[:3,3]+=x[3:6];ratio=np.exp(x[6:base])
        local=[x[offsets[k]:offsets[k+1]] if free else fixed_local[k] for k in range(K)]
        return X,ratio,local
    def evaluate(x,report=False,raw=False):
        X,ratio,local=unpack(x);pieces=[];rows=[]
        for k,s in enumerate(segments):
            R,t,P=s.state(local[k]);v,z=s.visual(R,t,P)
            if free:
                # Positive-depth barrier retained as separate residual row.
                pieces.append(np.column_stack(((v/sigma_px if raw else _robust(v/sigma_px,"soft_l1")),np.maximum(.05-z,0)*100)).ravel()*np.sqrt(visual_weight/(K*len(v))))
            if joint:
                r=s.motion(R,t,X,ratio[k],kind,he,rotation_multiplier)
                if kind=="scene": r=r/sigma_scene
                if not raw:
                    if kind=="he" and not he_block_robust:
                        r=_robust(r,he.robust_loss)
                    else:
                        r=r*np.sqrt(2/(np.hypot(1,np.linalg.norm(r,axis=1))+1))[:,None]
                pieces.append(r.ravel()*np.sqrt(motion_weight/(K*len(r))))
                if ground_weight>0:
                    g=np.array([s.ground(P,X,ratio[k])/sigma_ground])
                    pieces.append((g if raw else _robust(g,"soft_l1"))*np.sqrt(ground_weight/K))
            if report:
                rows.append(dict(centers=len(s.frames),landmarks=len(P),observations=len(v),
                    reprojection_rmse_px=float(np.sqrt(np.mean(np.sum(v*v,axis=1)))),
                    reprojection_median_px=float(np.median(np.linalg.norm(v,axis=1))),
                    nonpositive_depth=int(np.count_nonzero(z<=0)),
                    camera_center_change_rms_m=float(np.sqrt(np.mean(np.sum((t-s.t)**2,axis=1)))),
                    rotation_change_rms_deg=float(np.rad2deg(np.sqrt(np.mean(Rotation.from_matrix(R@s.R.transpose(0,2,1)).magnitude()**2)))),
                    landmark_change_rms_m=float(np.sqrt(np.mean(np.sum((P-s.points)**2,axis=1)))),
                    ground_offset_m=s.ground(P,X,ratio[k]),
                    he_normalized_rms=float(np.sqrt(np.mean(np.sum(s.motion(R,t,X,ratio[k],"he",he)**2,axis=1)))),
                    scene_rms_m=float(np.sqrt(np.mean(np.sum(s.motion(R,t,X,ratio[k],"scene",he)**2,axis=1))))))
        # Each line is in its synchronous LiDAR frame: X alone projects it
        # into the actual image. No local BA pose/scale dependence.
        if use_lines:
            lr=line_residual(X,line_records,line_K,line_sigma_px).reshape(-1,2)
            if not raw: lr[:,0]=_robust(lr[:,0],"soft_l1")
            pieces.append(lr.ravel()*np.sqrt(line_weight/len(lr)))
        r=np.concatenate(pieces)
        return (dict(T_camera_lidar=X.tolist(),scales=(np.asarray(scales)*ratio).tolist(),
                     segments=rows,cost=float(.5*(r@r)),
                     line_metrics=line_metrics(X,line_records,line_K) if use_lines else None),local) if report else r
    # Exact structural Jacobian sparsity including the ground centroid.
    x0=np.zeros(dim);J=lil_matrix((len(evaluate(x0)),dim),dtype=np.int8);row=0;ground_rows=[]
    for k,s in enumerate(segments):
        off=offsets[k]
        if free:
            for i,j in zip(s.obs_cam,s.obs_pt):
                cols=np.r_[s.pose_cols[i][s.pose_cols[i]>=0],s.pt_cols[j]]+off
                J[row:row+3,cols]=1;row+=3
        if joint:
            glob=np.r_[0:6,6+k];width=6 if kind=="he" else 12
            for i in s.centers:
                cols=np.r_[glob,s.pose_cols[i][s.pose_cols[i]>=0]+off] if free else glob
                J[row:row+width,cols]=1;row+=width
            if ground_weight>0:
                cols=np.r_[glob,s.pt_cols[s.ground_ids].ravel()+off] if free else glob
                J[row,cols]=1;ground_rows.append(row);row+=1
    if use_lines:
        # Five samples per 3D segment, signed pixel distance + depth barrier.
        J[row:,:6]=1;row+=len(line_residual(X0,line_records,line_K,line_sigma_px))
    if row!=J.shape[0]: raise RuntimeError("BA sparsity row mismatch")
    sol=least_squares(evaluate,x0,jac_sparsity=J.tocsr(),tr_solver="lsmr",x_scale="jac",
                      bounds=(lo,hi),max_nfev=max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    initial,_=evaluate(x0,True);final,local=evaluate(sol.x,True)
    # No false covariance or global rank claims from a sparse local solver.
    ok=bool(sol.success and all(r["nonpositive_depth"]==0 for r in final["segments"])
            and (not use_lines or final["line_metrics"]["nonpositive_or_near_samples"]==0))
    active=[]
    for k,s in enumerate(segments):
        if free:
            for i,d in np.argwhere(s.pose_cols>=0):
                j=offsets[k]+s.pose_cols[i,d]
                if sol.active_mask[j]:
                    active.append(dict(segment=k,frame=int(s.frames[i]),component=int(d),
                                       side=int(sol.active_mask[j]),value=float(sol.x[j])))
    global_active=[dict(component=int(j),side=int(sol.active_mask[j]),value=float(sol.x[j]))
                   for j in range(base) if sol.active_mask[j]]
    info=None
    if information:
        if not joint or not free: raise ValueError("Information requires joint free BA")
        import logging
        log=logging.getLogger(__name__)
        info={}
        mask=np.ones(J.shape[0],bool);mask[ground_rows]=False
        for label,at in (("common_initial",x0),("solution",sol.x)):
            info[label]={}
            for mode in ("raw","robust"):
                log.info("Information %s %s: sparse differentiation and nuisance projection",label,mode)
                # Public SciPy API: a one-evaluation unconstrained call
                # evaluates the 3-point Jacobian at the specified state.
                jac=least_squares(lambda x:evaluate(x,raw=(mode=="raw")),at,
                                  jac="3-point",jac_sparsity=J.tocsr(),max_nfev=1).jac
                result=marginal_information(jac,at[:3],maxiter=information_maxiter)
                entry={"all_nuisance_free":result}
                if label=="solution" and ground_rows:
                    entry["remove_ground_same_state"]=marginal_information(jac[mask],at[:3],maxiter=information_maxiter)
                if label=="solution" and np.any(sol.active_mask[6:]):
                    entry["active_nuisance_fixed"]=marginal_information(jac,at[:3],
                        fixed_nuisance=sol.active_mask[6:]!=0,maxiter=information_maxiter)
                info[label][mode]=entry
        info["warning"]="Raw uses linear whitened weighted residuals; robust uses derivative of solver's transformed residual, not exact robust Hessian or frozen IRLS. Unconstrained tangent; active nuisance conditional fixes only active nuisance. Active extrinsic bounds are reported separately."
    return dict(success=ok,solver_converged=bool(sol.success),status=int(sol.status),
                information=info,
                active_pose_bounds=active,active_global_bounds=global_active,
                rotation_multiplier=float(rotation_multiplier),he_block_robust=bool(he_block_robust),
                ground_weight=float(ground_weight),line_weight=float(line_weight),line_sigma_px=float(line_sigma_px),
                message=str(sol.message),nfev=int(sol.nfev),optimality=float(sol.optimality),
                bound_hit=bool(np.any(sol.active_mask)),variables=dim,jacobian_nnz=int(J.nnz),
                initial=initial,final=final),local
