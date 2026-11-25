"""Local GN information diagnostics and exact point-moment controls.

No dense full BA Hessian, no statistical covariance interpretation.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr
from spade_calib.calibration.scene_consistency import sqrt_factor


def left_jacobian(phi):
    """d Exp(phi) in left tangent coordinates."""
    x,y,z=np.asarray(phi,float)
    A=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    t=np.linalg.norm(phi)
    if t<1e-4:
        a=.5-t*t/24
        b=1/6-t*t/120
    else:
        a=(1-np.cos(t))/t**2
        b=(t-np.sin(t))/t**3
    return np.eye(3)+a*A+b*(A@A)


def marginal_information(J, phi=None, *, fixed_nuisance=None, rtol=1e-8,
                         atol=1e-10, maxiter=10000):
    """Project J_X onto orthogonal complement of nuisance column space.

    LSMR on column-normalized sparse J_eta avoids squaring its condition
    number. Approximate projection is checked for normal residual and
    solver termination, and marked unreliable if it fails. Rank uses
    a relative singular-value threshold, not a statistical test.
    X coordinates: left rotation tangent 1 degree, additive camera t .1m.
    Fixed nuisance is an optional active-set conditional diagnostic,
    NOT constrained tangent-cone information.
    """
    J=sparse.csr_matrix(J,dtype=float)
    if J.shape[1]<6 or not np.isfinite(J.data).all():
        raise ValueError("Invalid Jacobian")
    if not 0<rtol<1 or not 0<atol<1 or maxiter<1:
        raise ValueError("Invalid information tolerances")
    units=np.r_[np.full(3,np.deg2rad(1.)),np.full(3,.1)]
    B=J[:,:6].toarray()
    if phi is not None:
        B[:,:3]=np.linalg.solve(left_jacobian(phi).T,B[:,:3].T).T
    B*=units
    A=J[:,6:].tocsc()
    fixed=np.zeros(A.shape[1],bool) if fixed_nuisance is None else np.asarray(fixed_nuisance,bool)
    if fixed.shape!=(A.shape[1],): raise ValueError("Nuisance mask shape mismatch")
    A=A[:,~fixed]
    norms=np.sqrt(np.asarray(A.power(2).sum(axis=0)).ravel())
    keep=norms>0
    A=A[:,keep]@sparse.diags(1/norms[keep]) if np.any(keep) else A[:,:0]
    A=A.tocsr()
    E=B.copy();solves=[]
    for j,b in enumerate(B.T):
        if A.shape[1] and np.linalg.norm(b)>0:
            sol=lsmr(A,b,atol=atol,btol=atol,conlim=0.,maxiter=maxiter)
            E[:,j]=b-A@sol[0]
            nr=float(np.linalg.norm(A.T@E[:,j])/max(np.linalg.norm(b),1e-30))
            solves.append(dict(axis=j,stop=int(sol[1]),iterations=int(sol[2]),
                               normal_residual_relative=nr,
                               reliable=bool(sol[1] in (0,1,2,4,5) and nr<=1e-7)))
        else:
            solves.append(dict(axis=j,stop=0,iterations=0,normal_residual_relative=0.,reliable=True))
    U,sv,Vh=np.linalg.svd(E,full_matrices=False)
    if len(sv)<6:
        raise ValueError("Information requires at least six rows")
    H=E.T@E
    threshold=rtol*sv[0] if sv[0]>0 else 0.
    rank=int(np.count_nonzero(sv>threshold))
    reliable=all(s["reliable"] for s in solves)
    # Numerical leakage from an incomplete projection can inflate weak modes.
    # Report both computed spectrum and a conservative numerical floor.
    floor=max(threshold, max(s["normal_residual_relative"] for s in solves)*np.linalg.norm(B,2))
    return dict(reliable=reliable,rank_rtol=rtol,effective_rank=rank if reliable else None,
                singular_values=sv.tolist(),eigenvalues=(sv**2).tolist(),
                information_matrix=H.tolist(),fixed_pose_information_matrix=(B.T@B).tolist(),
                weakest_direction_normalized=Vh[-1].tolist(),
                weakest_direction_physical_rad_m=(Vh[-1]*units).tolist(),
                eigenvectors_columns=Vh.T.tolist(),
                condition_number_J=float(sv[0]/sv[-1]) if reliable and rank==6 else None,
                condition_number_H=float((sv[0]/sv[-1])**2) if reliable and rank==6 else None,
                axis_conditional_information=np.diag(H).tolist(),
                axis_order=["rx","ry","rz","tx","ty","tz"],
                units_rad_m=units.tolist(),projection_checks=solves,
                numerical_floor_heuristic=float(floor),
                fixed_nuisance_count=int(fixed.sum()),zero_nuisance_columns=int((~keep).sum()),
                nuisance_columns=int(A.shape[1]),
                warning="Local weighted GN, not covariance. Correlated motion/visual observations; rank threshold is numerical. Floor is heuristic, not a certified singular-value error bound.")


def moments_from_factors(F):
    """Recover mu and centered covariance from exact P2P F^T F = I3 kron M."""
    F=np.asarray(F,float)
    if F.ndim!=3 or F.shape[1:]!=(12,12) or not np.isfinite(F).all():
        raise ValueError("Expected finite P2P factors")
    G=F.transpose(0,2,1)@F
    M=G[:,:4,:4]
    expected=np.zeros_like(G)
    for k in range(3): expected[:,4*k:4*k+4,4*k:4*k+4]=M
    if not np.allclose(G,expected,rtol=1e-8,atol=1e-7) or not np.allclose(M[:,3,3],1.,atol=1e-8):
        raise ValueError("Not normalized point moment factors")
    mu=M[:,:3,3].copy()
    C=M[:,:3,:3]-mu[:,:,None]*mu[:,None,:]
    C=(C+C.transpose(0,2,1))*.5
    if np.any(np.linalg.eigvalsh(C)<-1e-7):
        raise ValueError("Non-PSD covariance")
    return mu,C


def moment_factors(mu,C):
    mu=np.asarray(mu,float);C=np.asarray(C,float)
    if mu.ndim!=2 or mu.shape[1]!=3 or C.shape!=(len(mu),3,3):
        raise ValueError("Invalid virtual moment shapes")
    M=np.zeros((len(mu),4,4))
    M[:,:3,:3]=C+mu[:,:,None]*mu[:,None,:]
    M[:,:3,3]=mu;M[:,3,:3]=mu;M[:,3,3]=1.
    return np.stack([sqrt_factor(np.kron(np.eye(3),m)) for m in M])


def virtual_moments(factors,seed=17,shuffle_repeats=3):
    """Equal-segment, equal-center pooled moments, with fixed translation block.

    Shuffling within each segment exactly preserves its multiset of moments.
    Pooled isotropic trace matches the uncentered second moment (not the
    centered covariance). J6/J7 isolate components; J7 may change trace.
    """
    if shuffle_repeats<1: raise ValueError("Need at least one shuffle")
    data=[moments_from_factors(f) for f in factors]
    mu0=np.mean([mu.mean(0) for mu,C in data],axis=0)
    C0=np.mean([C.mean(0) for mu,C in data],axis=0)
    S0=np.mean([(C+mu[:,:,None]*mu[:,None,:]).mean(0) for mu,C in data],axis=0)
    Cp=(S0-np.outer(mu0,mu0));Cp=(Cp+Cp.T)*.5
    iso=np.eye(3)*np.trace(S0)/3
    out={k:[] for k in ("J3_isotropic","J4_pooled","J6_real_mu","J7_real_cov")}
    for mu,C in data:
        n=len(mu);m=np.broadcast_to(mu0,mu.shape)
        for key,a,b in (
                ("J3_isotropic",np.zeros_like(mu),np.broadcast_to(iso,C.shape)),
                ("J4_pooled",m,np.broadcast_to(Cp,C.shape)),
                ("J6_real_mu",mu,np.broadcast_to(C0,C.shape)),
                ("J7_real_cov",m,C)):
            out[key].append(moment_factors(a,b))
    permutations={}
    for r in range(shuffle_repeats):
        key=f"J5_shuffle_{r}"
        rng=np.random.default_rng([seed,r])
        order=[rng.permutation(len(f)) for f in factors]
        out[key]=[f[p] for f,p in zip(factors,order)]
        permutations[key]=[p.tolist() for p in order]
    def stats(fs):
        data=[moments_from_factors(f) for f in fs]
        return dict(mean_mu=np.mean([m.mean(0) for m,c in data],axis=0).tolist(),
                    mean_covariance=np.mean([c.mean(0) for m,c in data],axis=0).tolist(),
                    mean_second_moment_trace=float(np.mean([np.mean(np.trace(c,axis1=1,axis2=2)+np.sum(m*m,axis=1)) for m,c in data])))
    return out,dict(reference=stats(factors),controls={k:stats(v) for k,v in out.items()},
                    permutations=permutations,seed=seed,weighting="equal segment / equal center")


def scale_coupling_information(J, n_scales, phi=None, *, maxiter=10000,
                               rtol=1e-8, atol=1e-10):
    """Same-state nested nuisance projection: eta vs [eta, log-scale].

    Columns [X(6), log-scales, local BA]. Units identical to I experiment.
    The PSD difference is information lost by freeing scale, not bias or
    a statistical covariance. Use a common threshold for both spectra.
    """
    J=sparse.csr_matrix(J,dtype=float)
    if not 1<=n_scales<=J.shape[1]-6:
        raise ValueError("Invalid scale column count")
    fixed=np.zeros(J.shape[1]-6,bool);fixed[:n_scales]=True
    kw=dict(maxiter=maxiter,rtol=rtol,atol=atol)
    f=marginal_information(J,phi,fixed_nuisance=fixed,**kw)
    u=marginal_information(J,phi,**kw)
    F=np.asarray(f["information_matrix"]);U=np.asarray(u["information_matrix"])
    D=F-U;D=(D+D.T)/2
    ev,V=np.linalg.eigh(F);de=np.linalg.eigvalsh(D)
    svmax=max(f["singular_values"][0],u["singular_values"][0])
    floor=max(f["numerical_floor_heuristic"],u["numerical_floor_heuristic"],rtol*svmax)
    # PSD tolerance is a numerical consistency check, not a certified bound.
    tol=max(1e-12,1e-7*max(float(ev[-1]),0.),10*floor*svmax)
    reliable=bool(f["reliable"] and u["reliable"] and de[0]>=-tol)
    def ratio(a,b):
        return float(a/b) if b>floor**2 and b>0 else None
    keep=ev>floor**2
    fractions=[]
    if reliable and np.any(keep):
        Q=V[:,keep]/np.sqrt(ev[keep])
        fractions=np.linalg.eigvalsh(Q.T@D@Q).tolist()
    weak=np.asarray(u["weakest_direction_normalized"])
    return dict(reliable=reliable,scale_columns=int(n_scales),
                fixed_scale=f,free_scale=u,information_loss_matrix=D.tolist(),
                loss_eigenvalues=de.tolist(),psd_tolerance=tol,
                common_sv_floor=float(floor),
                rank_fixed_common=int(np.count_nonzero(np.asarray(f["singular_values"])>floor)) if reliable else None,
                rank_free_common=int(np.count_nonzero(np.asarray(u["singular_values"])>floor)) if reliable else None,
                trace_loss_fraction=ratio(np.trace(D),np.trace(F)) if reliable else None,
                axis_loss_fraction=[ratio(a,b) for a,b in zip(np.diag(D),np.diag(F))] if reliable else None,
                free_weak_direction_loss_fraction=ratio(weak@D@weak,weak@F@weak) if reliable else None,
                generalized_loss_fractions=fractions,
                warning="Fractions are local directional information losses on the numerically supported subspace; not covariance, error attribution or evidence of scale bias. Bounds ignored; tiny negative fractions may be numerical.")
