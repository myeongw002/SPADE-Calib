"""S0-R: multi-segment scale oracle and paired noisy geometry controls.

Piecewise constant true scales, shared X, fixed measured camera/LiDAR
relative poses. Ground offset provides height information. No BA here.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from spade_calib.calibration.scale_geometry_synthetic import (
    K, synthetic_case, motion_residual, line_residual, central_jac)
from spade_calib.calibration.crossmodal_lines import image_line, perturb

CASES=("planar","general")
ARMS=("motion","motion_single","motion_multiview","motion_glim_multiview")
MODELS=("global","multi","oracle")
REGIMES={"clean":(0.,0.,0.,0.,0.,0.),"both_mild":(.5,.02,.05,.02,.05,.01),
         "lidar_high":(.5,.02,.05,.08,.2,.01)}
TRUE_SCALES=np.array([1.08,1.2,1.32])
X_UNITS=np.r_[np.full(3,np.deg2rad(1.)),np.full(3,.1)]


def dataset(case,regime,seed,drift=True):
    if case not in CASES or regime not in REGIMES: raise ValueError("Unknown S0-R condition")
    pix,ct,cr,lt,lr,ground=REGIMES[regime]
    scales=TRUE_SCALES.copy() if drift else np.full(3,1.2)
    records=[]
    for k,s in enumerate(scales):
        rng=np.random.default_rng([seed,k])
        case_name=("straight_bilateral","turn_bilateral","turn_bilateral")[k] if case=="planar" else "general_diverse"
        d=synthetic_case(case_name)
        X=d["X"];R=d["R"].copy();c=d["b"]*d["scale"]
        # Vary baseline and yaw direction across segments; anchor remains I.
        c*=([.8,1.,1.2][k])
        if k==2:
            c[:,0]*=-1
            v=Rotation.from_matrix(R).as_rotvec();v[:,1]*=-1
            R=Rotation.from_rotvec(v).as_matrix()
        B=np.repeat(np.eye(4)[None],len(c),axis=0)
        B[:,:3,:3]=R;B[:,:3,3]=c
        A=np.linalg.inv(X)@B@X
        p=[l@X[:3,:3].T+X[:3,3] for l in d["lines"]]
        uv=[]
        for j in range(len(c)):
            row=[]
            for l in p:
                q=(l-c[j])@R[j]
                if np.any(q[:,2]<=.5): raise ValueError("Invalid synthetic depth")
                u=(q@K.T)[:,:2]/q[:,2,None]
                u+=pix*rng.standard_normal(u.shape)
                row.append(image_line(u))
            uv.append(row)
        # Perturb anchor-relative poses, preserving exact anchor identity.
        # Draw every noise source even when sigma=0 to pair noise sweeps.
        b=c/s
        b[1:]+=ct/s*rng.standard_normal(b[1:].shape)
        R[1:]=Rotation.from_rotvec(np.deg2rad(cr)*rng.standard_normal((len(c)-1,3))).as_matrix()@R[1:]
        A[1:,:3,3]+=lt*rng.standard_normal(A[1:,:3,3].shape)
        A[1:,:3,:3]=Rotation.from_rotvec(np.deg2rad(lr)*rng.standard_normal((len(c)-1,3))).as_matrix()@A[1:,:3,:3]
        n=np.array([0.,-1.,0.]);height=1.5
        nL=X[:3,:3].T@n
        dc=(height+ground*rng.standard_normal())/s
        dl=height+n@X[:3,3]+ground*rng.standard_normal()
        records.append(dict(X=X,R=R,b=b,A=A,lines=d["lines"],uv=np.asarray(uv),
                            nL=nL,dc=dc,dl=dl))
    return records,scales,X


def residual(records,X,scales,arm):
    if arm not in ARMS or len(scales)!=len(records): raise ValueError("Invalid S0-R arm")
    out=[]
    for d,s in zip(records,scales):
        out.append(motion_residual(d,X,s))
        # Same fixed ground-offset factor in every arm, including motion-only.
        out.append(np.array([(s*d["dc"]-d["dl"]+(X[:3,:3]@d["nL"])@X[:3,3])/.05*np.sqrt(.1)]))
        if arm=="motion_single": out.append(line_residual(d,X,s,"single"))
        elif arm=="motion_multiview": out.append(line_residual(d,X,s,"camera"))
        elif arm=="motion_glim_multiview": out.append(line_residual(d,X,s,"glim"))
    return np.concatenate(out)/np.sqrt(len(records))


def information(J):
    _,sv,Vh=np.linalg.svd(J,full_matrices=False)
    floor=max(1e-8,1e-7*sv[0])
    rank=int(np.count_nonzero(sv>floor))
    B=J[:,:6];A=J[:,6:]
    if A.shape[1]:
        U,a,_=np.linalg.svd(A,full_matrices=False)
        Q=U[:,a>floor];E=B-Q@(Q.T@B)
    else: E=B
    H=E.T@E;F=B.T@B
    axis_loss=np.diag(F-H)
    fractions=[float(x/y) if y>floor**2 else None for x,y in zip(axis_loss,np.diag(F))]
    return dict(rank=rank,variables=J.shape[1],singular_values=sv.tolist(),threshold=float(floor),
                null_space_rows=Vh[rank:].tolist(),scale_marginal_eigenvalues=np.linalg.eigvalsh(H).tolist(),
                scale_fixed_eigenvalues=np.linalg.eigvalsh(F).tolist(),axis_loss_fraction=fractions)


def run_trial(case,regime,seed,drift=True):
    records,truth,X=dataset(case,regime,seed,drift)
    rng=np.random.default_rng([seed,999])
    # Paired initial X; different seeds change noise AND start, not an
    # independent factorial experiment on initialization.
    delta=np.r_[np.deg2rad(rng.uniform(-.8,.8,3)),rng.uniform(-.15,.15,3)]
    X0=perturb(X,delta);s0=1.1
    out={}
    for model in MODELS:
        ns={"global":1,"multi":len(truth),"oracle":0}[model]
        def state(v):
            Y=perturb(X0,v[:6]*X_UNITS)
            if model=="oracle": scales=truth.copy()
            elif model=="global": scales=np.full(len(truth),s0*np.exp(.1*v[6]))
            else: scales=s0*np.exp(.1*v[6:])
            return Y,scales
        z=np.zeros(6+ns);limits=np.r_[np.full(3,5.),np.full(3,10.),np.full(ns,5.)]
        out[model]={}
        for arm in ARMS:
            fun=lambda v:residual(records,*state(v),arm)
            sol=least_squares(fun,z,bounds=(-limits,limits),max_nfev=200,
                              ftol=1e-9,xtol=1e-9,gtol=1e-9,x_scale=1.)
            Y,est=state(sol.x)
            # Analyze at fitted state: global model under drift has no exact truth.
            J=central_jac(fun,sol.x)
            out[model][arm]=dict(converged=bool(sol.success),bound_hit=bool(np.any(sol.active_mask)),
                nfev=int(sol.nfev),cost=float(sol.cost),initial_cost=float(.5*np.linalg.norm(fun(z))**2),
                rotation_error_deg=float(np.rad2deg(Rotation.from_matrix(Y[:3,:3]@X[:3,:3].T).magnitude())),
                translation_error_m=float(np.linalg.norm(Y[:3,3]-X[:3,3])),
                translation_error_vector_m=(Y[:3,3]-X[:3,3]).tolist(),
                scale_relative_error=(est/truth-1).tolist(),
                scale_relative_rmse=float(np.sqrt(np.mean((est/truth-1)**2))),
                scales=est.tolist(),information_at_solution=information(J))
    return dict(case=case,regime=regime,seed=seed,drift=drift,truth_scales=truth.tolist(),arms=out)
