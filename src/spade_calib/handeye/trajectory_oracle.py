"""Fixed-pair KITTI motion substitution and local residual bias diagnostics."""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from spade_calib.handeye.residuals import residuals
from spade_calib.selection.trajectory import MotionPairs


def subset(p,m):
    return MotionPairs(p.Ra[m],p.ta[m],p.Rb[m],p.tb[m],p.indices[m],p.starts_ns[m])


def kitti_oracle_pairs(poses,cal,camera_index,frames,pairs):
    """GT poses are camera0-to-world; derive both sensor motions with GT X."""
    raw=np.asarray(poses,dtype=float)
    if raw.ndim!=2 or raw.shape[1]!=12 or not np.isfinite(raw).all():
        raise ValueError("Expected finite KITTI GT Nx12 poses")
    def proper(R):
        if not np.allclose(np.swapaxes(R,-1,-2)@R,np.eye(3),atol=1e-4) or np.any(np.linalg.det(R)<=0):
            raise ValueError("GT rotation not rigid within rounding tolerance")
        return Rotation.from_matrix(R).as_matrix()
    P=raw.reshape(-1,3,4)
    C=np.tile(np.eye(4),(len(P),1,1))
    C[:,:3,:3]=proper(P[:,:,:3]); C[:,:3,3]=P[:,:,3]
    L=np.eye(4); L[:3,:3]=proper(cal["Tr"][:,:3]); L[:3,3]=cal["Tr"][:,3]
    P0,Pk=cal["P0"],cal[f"P{camera_index}"]
    T=np.eye(4)
    T[:3,3]=np.linalg.solve(Pk[:,:3],Pk[:,3])-np.linalg.solve(P0[:,:3],P0[:,3])
    frames=np.asarray(frames)
    if frames.ndim!=1 or frames.dtype.kind not in "iu" or np.any(frames<0) or np.any(frames>=len(P)) or np.any(np.diff(frames)<=0):
        raise ValueError("Invalid matched raw frame IDs")
    i,j=frames[pairs.indices].T
    M=np.linalg.inv(C[i])@C[j]
    A=np.linalg.inv(L)@M@L; B=T@M@np.linalg.inv(T)
    return MotionPairs(A[:,:3,:3],A[:,:3,3],B[:,:3,:3],B[:,:3,3],pairs.indices,pairs.starts_ns),T@L


def substitute(a,g,lidar_gt,camera_gt):
    if not np.array_equal(a.indices,g.indices) or not np.array_equal(a.starts_ns,g.starts_ns):
        raise ValueError("Motion identity mismatch")
    return MotionPairs(g.Ra if lidar_gt else a.Ra,g.ta if lidar_gt else a.ta,
                       g.Rb if camera_gt else a.Rb,g.tb if camera_gt else a.tb,a.indices,a.starts_ns)


def rows(p,groups,s,X):
    r=np.empty((p.count,6))
    for k,scale in enumerate(s):
        m=groups==k
        r[m]=residuals(subset(p,m),X[:3,:3],X[:3,3],float(scale))
    return r


def whiten(r,h):
    return r/np.r_[np.full(3,np.deg2rad(h.sigma_relative_rotation_deg)),
                    np.full(3,h.sigma_relative_translation_m)]


def profile_scales(p,groups,X,s,h):
    f=least_squares(lambda x:whiten(rows(p,groups,np.exp(x),X),h).ravel(),np.log(s),
                    bounds=(-30.,30.),loss=h.robust_loss,x_scale="jac",
                    max_nfev=h.max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    if not f.success:
        raise RuntimeError(f.message)
    return np.exp(f.x)


def sqrt_weights(r,loss):
    if loss=="linear": return np.ones_like(r)
    if loss=="soft_l1": return (1+r*r)**(-.25)
    if loss=="huber": return 1/np.sqrt(np.maximum(1,abs(r)))
    if loss=="cauchy": return 1/np.sqrt(1+r*r)
    if loss=="arctan": return 1/np.sqrt(1+r**4)
    raise ValueError("Unsupported robust loss")


def components(a,g,groups,s,X):
    """Exact translation split at common GT X and common scales."""
    c=(a.Rb-g.Rb)@X[:3,3]+a.tb*s[groups,None]-g.tb
    l=-(a.ta-g.ta)@X[:3,:3].T
    closure=rows(g,groups,np.ones(len(s)),X)[:,3:]
    if not np.allclose(rows(a,groups,s,X)[:,3:],c+l+closure,atol=1e-9):
        raise ValueError("Translation decomposition failed closure")
    return {"camera_motion_and_scale":c,"lidar_translation":l,"GT_closure":closure}


def bias(p,groups,X,s,h,metric_camera=False,terms=None):
    """Local marginalized GN / frozen IRLS bias; not a causal estimate."""
    units=np.r_[np.full(3,np.deg2rad(h.reference_rotation_deg)),
                np.full(3,h.reference_translation_m),
                np.full(0 if metric_camera else len(s),h.reference_log_scale)]
    def fun(x):
        Y=X.copy();Y[:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@X[:3,:3]
        Y[:3,3]+=x[3:6]
        return whiten(rows(p,groups,s if metric_camera else s*np.exp(x[6:]),Y),h).ravel()
    r=fun(np.zeros(len(units)))
    J=np.column_stack([(fun(d)-fun(-d))/2e-5 for d in np.eye(len(units))*units*1e-5])
    def analyze(w):
        M=J*w[:,None]
        U,S,_=np.linalg.svd(M[:,np.r_[0:3,6:len(units)]],full_matrices=False)
        U=U[:,S>max(S[0],1e-15)*h.rank_sv_ratio]
        T=M[:,3:6]-U@(U.T@M[:,3:6])
        u,sv,vh=np.linalg.svd(T,full_matrices=False)
        valid=sv>max(sv[0],1e-15)*h.rank_sv_ratio
        pinv=(vh[valid].T/sv[valid])@u[:,valid].T
        def delta(e):
            e=e*w
            return (-h.reference_translation_m*pinv@(e-U@(U.T@e))).tolist()
        result={"translation_singular_values":sv.tolist(),"translation_rank":int(valid.sum()),
                "weakest_translation_camera_xyz":vh[-1].tolist(),
                "predicted_translation_error_camera_m":delta(r),
                "segment_contributions_camera_m":[delta(np.where(np.repeat(groups==k,6),r,0))
                                                  for k in range(len(s))]}
        if terms is not None:
            part={}
            for name,t in terms.items():
                e=np.zeros((p.count,6));e[:,3:]=t/h.sigma_relative_translation_m
                part[name]=delta(e.ravel())
            e=r.reshape(-1,6).copy();e[:,3:]=0
            part["rotation_residual_combined"]=delta(e.ravel())
            result["component_contributions_camera_m"]=part
        return result
    def stats(q):
        return {"count":len(q),"translation_mean_camera_m":q[:,3:].mean(0).tolist(),
                "translation_rms_m":float(np.sqrt(np.mean(np.sum(q[:,3:]**2,axis=1)))),
                "rotation_median_deg":float(np.rad2deg(np.median(np.linalg.norm(q[:,:3],axis=1))))}
    raw=rows(p,groups,s,X)
    return {"reference_scales":s.tolist(),"at_GT":stats(raw),
            "segments":[stats(raw[groups==k]) for k in range(len(s))],
            "unweighted":analyze(np.ones_like(r)),"frozen_IRLS":analyze(sqrt_weights(r,h.robust_loss)),
            "warning":"First-order local approximation; overlapping pairs correlated; not covariance or causal proof."}
