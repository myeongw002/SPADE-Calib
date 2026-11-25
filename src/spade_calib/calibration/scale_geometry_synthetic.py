"""S0 synthetic scale/extrinsic observability, exact correspondences.

Camera-frame axes x right, y down, z forward. A_j maps LiDAR j to
anchor LiDAR; B_j maps camera j to anchor camera. X maps L to C.
No real dataset, no matching or accumulation claims.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from spade_calib.calibration.crossmodal_lines import image_line, sample_line, perturb

UNITS=np.r_[np.full(3,np.deg2rad(1.)),np.full(3,.1),.1]
CASES=("stationary_one","straight_one","straight_bilateral","turn_bilateral","general_diverse")
ARMS=("motion","single","multiview","motion_single","motion_multiview","motion_glim_multiview")
K=np.array([[700.,0.,620.],[0.,700.,180.],[0.,0.,1.]])


def synthetic_case(case, seed=17, noisy=False):
    if case not in CASES: raise ValueError("Unknown S0 case")
    rng=np.random.default_rng(seed)
    X=np.eye(4)
    X[:3,:3]=Rotation.from_euler("xyz",[2.,-3.,4.],degrees=True).as_matrix()
    X[:3,3]=[.18,-.12,.25]
    scale=1.2
    c=np.array([[0.,0.,0.],[0.,0.,.7],[0.,0.,1.5],[0.,0.,2.4],[0.,0.,3.2]])
    angles=np.zeros_like(c)
    if case=="stationary_one": c[:]=0.
    if case=="turn_bilateral":
        c[:,0]=[0.,.06,.22,.48,.85]
        angles[:,1]=[0.,2.,5.,8.,12.]
    if case=="general_diverse":
        c[:,0]=[0.,.2,-.25,.55,.8];c[:,1]=[0.,.15,-.12,.25,-.18]
        angles=np.array([[0,0,0],[3,2,-2],[-2,5,3],[4,8,-3],[-4,12,5]])
    R=Rotation.from_euler("xyz",angles,degrees=True).as_matrix()
    world_lines=[np.array([[-2.,-1.,10.],[-2.,1.,10.]])]
    if case not in ("stationary_one","straight_one"):
        world_lines.append(np.array([[3.,-1.2,13.],[3.,1.1,13.]]))
    if case=="general_diverse":
        world_lines.extend([np.array([[-2.,.6,9.],[2.,.6,11.]]),
                            np.array([[-1.8,-1.2,12.],[1.5,.9,14.]]),
                            np.array([[1.,-1.,8.],[2.,.7,10.]])])
    # Construct physical lines in anchor LiDAR, not re-created each view.
    lines=[(p-X[:3,3])@X[:3,:3] for p in world_lines]
    B=np.repeat(np.eye(4)[None],len(c),axis=0)
    B[:,:3,:3]=R;B[:,:3,3]=c
    A=np.linalg.inv(X)@B@X
    uv=[]
    for j in range(len(c)):
        row=[]
        for p in world_lines:
            q=(p-c[j])@R[j]
            u=(q@K.T)[:,:2]/q[:,2,None]
            if noisy: u+=rng.normal(0,.5,u.shape)
            row.append(image_line(u))
        uv.append(row)
    b=c/scale
    if noisy:
        # Same measured camera trajectory and image labels in all arms.
        b[1:]+=rng.normal(0,.01,b[1:].shape)
        R[1:]=Rotation.from_rotvec(rng.normal(0,np.deg2rad(.05),(len(c)-1,3))).as_matrix()@R[1:]
    return dict(X=X,scale=scale,R=R,b=b,A=A,lines=lines,uv=np.asarray(uv))


def line_residual(data,X,scale,mode):
    """Fixed 3D support and image lines, signed perpendicular pixel distance."""
    out=[]
    views=range(1) if mode=="single" else range(len(data["R"]))
    for j in views:
        for i,p in enumerate(data["lines"]):
            p=sample_line(p)
            if mode=="glim":
                A=data["A"][j]
                q=((p-A[:3,3])@A[:3,:3])@X[:3,:3].T+X[:3,3]
            else:
                q=(p@X[:3,:3].T+X[:3,3]-scale*data["b"][j])@data["R"][j]
            z=q[:,2];uv=(q@K.T)[:,:2]/np.maximum(z[:,None],.1)
            l=data["uv"][j,i]
            out.extend(np.column_stack(((uv@l[:2]+l[2])/2.,100*np.maximum(.5-z,0))).ravel())
    return np.asarray(out)/np.sqrt(len(out)/2)


def motion_residual(data,X,scale):
    B=data["R"][1:];b=data["b"][1:]
    A=data["A"][1:];R=X[:3,:3];t=X[:3,3]
    rot=Rotation.from_matrix((B@R).transpose(0,2,1)@(R@A[:,:3,:3])).as_rotvec()
    tr=(B-np.eye(3))@t+scale*b-A[:,:3,3]@R.T
    return np.column_stack((rot/np.deg2rad(.1),tr/.05)).ravel()/np.sqrt(len(B))


def residual(data,X,scale,arm):
    if arm not in ARMS: raise ValueError("Unknown S0 arm")
    pieces=[]
    if arm.startswith("motion"): pieces.append(motion_residual(data,X,scale))
    if arm.endswith("single"): pieces.append(line_residual(data,X,scale,"single"))
    elif arm.endswith("multiview"):
        pieces.append(line_residual(data,X,scale,"glim" if "glim" in arm else "camera"))
    return np.concatenate(pieces)


def central_jac(fun, x, step=1e-5):
    return np.column_stack([(fun(x+d)-fun(x-d))/(2*step) for d in np.eye(len(x))*step])


def observability(J):
    """Normalized 7-column Jacobian, common floor, complete null space."""
    _,sv,Vh=np.linalg.svd(J,full_matrices=True)
    sv=np.pad(sv,(0,7-len(sv)))
    floor=max(1e-8,1e-7*sv[0])
    rank=int(np.count_nonzero(sv>floor))
    X=J[:,:6];s=J[:,6]
    U,sx,_=np.linalg.svd(X,full_matrices=False)
    Q=U[:,sx>floor]
    rem=s-Q@(Q.T@s)
    scale_info=float(rem@rem)
    coupling=float(np.linalg.norm(Q.T@s)/np.linalg.norm(s)) if np.linalg.norm(s)>floor else None
    # Same scale-fixed vs free information, no other nuisance variables.
    H=X.T@X
    D=np.outer(X.T@s,X.T@s)/(s@s) if np.linalg.norm(s)>floor else np.zeros((6,6))
    return dict(rank=rank,singular_values=sv.tolist(),null_space_rows=Vh[rank:].tolist(),
                threshold=float(floor),scale_column_norm=float(np.linalg.norm(s)),
                scale_conditional_information=scale_info,
                scale_extrinsic_cosine=coupling,
                scale_fixed_information=H.tolist(),scale_free_information=(H-D).tolist(),
                scale_information_loss_trace_fraction=float(np.trace(D)/np.trace(H)) if np.trace(H)>0 else None)


def analytic_scale_jacobian(data,X,scale):
    """Analytic log-scale derivative of the actual multi-view line residual."""
    out=[]
    for j,R in enumerate(data["R"]):
        dq=-scale*data["b"][j]@R
        for i,p in enumerate(data["lines"]):
            q=(sample_line(p)@X[:3,:3].T+X[:3,3]-scale*data["b"][j])@R
            z=q[:,2];h=q@K.T;dh=dq@K.T
            if np.any(z<=.5): raise ValueError("S0 analytic check requires positive depth")
            du=(dh[:2]*z[:,None]-h[:,:2]*dq[2])/z[:,None]**2
            out.extend(np.column_stack((du@data["uv"][j,i,:2]/2.,np.zeros(len(q)))).ravel())
    return np.asarray(out)*UNITS[6]/np.sqrt(len(out)/2)


def run_trial(case, seed=17, noisy=False):
    data=synthetic_case(case,seed,noisy)
    X=data["X"];s=data["scale"]
    # Truth is used for simulation/evaluation only. All arms share one perturbed start.
    start=perturb(X,np.r_[np.deg2rad([.5,-.6,.4]),[.12,-.08,.15]])
    s0=s*.90
    def state(x,origin=start,scale=s0):
        return perturb(origin,x[:6]*UNITS[:6]),scale*np.exp(x[6]*UNITS[6])
    fd=central_jac(lambda x:line_residual(data,*state(x,X,s),"camera"),np.zeros(7))[:,6]
    analytic=analytic_scale_jacobian(data,X,s)
    if not np.allclose(fd,analytic,atol=1e-7,rtol=1e-6):
        raise RuntimeError("S0 analytic scale derivative check failed")
    result={}
    for arm in ARMS:
        true_fun=lambda x:residual(data,*state(x,X,s),arm)
        zero=np.zeros(7);info=observability(central_jac(true_fun,zero))
        f=lambda x:residual(data,*state(x),arm)
        limit=np.r_[np.full(3,5.),np.full(3,10.),5.]
        sol=least_squares(f,zero,bounds=(-limit,limit),max_nfev=200,
                          ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale=1.)
        Y,scale=state(sol.x)
        result[arm]=dict(observability_at_truth=info,truth_residual_norm=float(np.linalg.norm(true_fun(zero))),
                         converged=bool(sol.success),nfev=int(sol.nfev),bound_hit=bool(np.any(sol.active_mask)),
                         cost=float(sol.cost),initial_cost=float(.5*np.linalg.norm(f(zero))**2),
                         rotation_error_deg=float(np.rad2deg(Rotation.from_matrix(Y[:3,:3]@X[:3,:3].T).magnitude())),
                         translation_error_m=float(np.linalg.norm(Y[:3,3]-X[:3,3])),
                         relative_scale_error=float(scale/s-1),T_camera_lidar=Y.tolist(),scale=float(scale))
    return dict(case=case,seed=seed,noisy=noisy,scale_derivative_max_abs_error=float(np.max(abs(fd-analytic))),
                truth_scale=s,truth_transform=X.tolist(),arms=result)
