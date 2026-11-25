import numpy as np
from scipy.spatial.transform import Rotation
from spade_calib.io.colmap_observations import camera_parameters,project,load_observations
from spade_calib.calibration.local_visual_ba import LocalBA,solve_local_ba
from spade_calib.calibration.scene_consistency import scene_factors
from spade_calib.handeye.solver import HandEyeSettings


def test_projection_models_and_text_observations(tmp_path):
    p=np.array([[1.,2.,10.]])
    k=camera_parameters(("PINHOLE",np.array([100.,200.,10.,20.])))
    np.testing.assert_allclose(project(p,k[None]),[[20.,60.]])
    k=camera_parameters(("SIMPLE_RADIAL",np.array([100.,10.,20.,.1])))
    np.testing.assert_allclose(project(p,k[None]),[[20.05,40.1]])
    f=tmp_path/"images.txt"
    f.write_text("1 1 0 0 0 0 0 0 1 000000.png\n20 40 42 5 6 -1\n")
    c,ids,uv=load_observations(f,{1})[1]
    assert c==1 and ids.tolist()==[42]
    np.testing.assert_array_equal(uv,[[20.,40.]])


def synthetic():
    rng=np.random.default_rng(2)
    n=5;m=24
    R=Rotation.from_rotvec(np.array([[0,0,0],[.08,.02,0],[-.02,.12,.04],[.05,-.1,.06],[.1,.05,-.08]])).as_matrix()
    t=np.array([[0.,0,0],[.5,.1,.05],[1.,-.1,.1],[1.5,.2,.2],[2.,.1,-.2]])
    P=rng.uniform([-2,-1,6],[2,1,12],size=(m,3))
    intr=np.tile([500,500,300,200,0,0,0,0],(n,1))
    oc=np.repeat(np.arange(n),m);op=np.tile(np.arange(m),n)
    q=np.einsum("ni,nij->nj",P[op]-t[oc],R[oc])
    uv=project(q,intr[oc])
    X=np.eye(4)
    f=np.stack([scene_factors(P,np.tile([0,0,1.],(m,1)))[0] for _ in range(n-1)])
    seg=LocalBA(np.arange(n),0,R,t,P,np.arange(m),oc,op,uv,intr,np.arange(1,n),
                R[1:],t[1:],f,np.arange(6),np.array([0,-1.,0]),1.5,np.array([0,-1.,0]),1.5)
    return seg,X


def test_gauge_and_zero_residual():
    seg,X=synthetic()
    x=np.full(seg.size,.001)
    R,t,P=seg.state(x)
    np.testing.assert_allclose(R[0],np.eye(3));np.testing.assert_allclose(t[0],0)
    j,d=seg.gauge;assert t[j,d]==seg.t[j,d]
    for kind in ("he","scene"):
        r=seg.motion(seg.R,seg.t,X,1.,kind,HandEyeSettings())
        np.testing.assert_allclose(r,0,atol=1e-10)
    assert abs(seg.ground(seg.points,X,1.))<1e-12
    for kind in ("visual","he","scene"):
        fit,_=solve_local_ba([seg],X,[1.],HandEyeSettings(),kind=kind,max_nfev=20)
        assert fit["success"]
        assert fit["final"]["cost"]<1e-15


def test_ba_decreases_pixel_error_with_perturbed_landmarks():
    seg,X=synthetic()
    seg.points=seg.points+np.random.default_rng(8).normal(0,.01,seg.points.shape)
    fit,_=solve_local_ba([seg],X,[1.],HandEyeSettings(),kind="visual",max_nfev=30)
    assert fit["final"]["cost"]<fit["initial"]["cost"]*.01
    assert fit["final"]["segments"][0]["nonpositive_depth"]==0


def test_duplicate_landmark_observations_keep_lowest_error_and_measured_pixel(tmp_path):
    from spade_calib.io.colmap_observations import unique_view_observations
    f=tmp_path/"images.txt"
    f.write_text("1 1 0 0 0 0 0 0 1 000000.png\n20 40 42 22 41 42 30 50 9 23 41 42\n")
    _,ids,uv=load_observations(f,{1})[1]
    p,pixels,removed=unique_view_observations(ids,uv,np.array([2.,.2,.5,.2]))
    assert removed==2
    np.testing.assert_array_equal(p,[42,9])
    np.testing.assert_array_equal(pixels,[[22.,41.],[30.,50.]])
    p,pixels,removed=unique_view_observations(np.array([],int),np.empty((0,2)),np.empty(0))
    assert removed==0 and pixels.shape==(0,2)


def test_rotation_attenuation_and_ground_off():
    seg,X=synthetic()
    R=seg.R.copy()
    R[1:]=Rotation.from_rotvec([.01,.02,-.01]).as_matrix()@R[1:]
    he=HandEyeSettings()
    a=seg.motion(R,seg.t,X,1.,"he",he)
    b=seg.motion(R,seg.t,X,1.,"he",he,.5)
    np.testing.assert_allclose(b[:,:3],.5*a[:,:3])
    np.testing.assert_allclose(b[:,3:],a[:,3:])
    a=seg.motion(R,seg.t,X,1.,"scene",he)
    b=seg.motion(R,seg.t,X,1.,"scene",he,.5)
    np.testing.assert_allclose(b,.5*a,atol=1e-12)
    for kind in ("he","scene"):
        fit,_=solve_local_ba([seg],X,[1.],he,kind=kind,ground_weight=0.,
                              rotation_multiplier=.5,he_block_robust=True,max_nfev=20)
        assert fit["success"] and fit["final"]["cost"]<1e-15


def test_marginal_information_matches_dense_projection():
    from spade_calib.calibration.ba_diagnostics import marginal_information
    from scipy.sparse import csr_matrix
    rng=np.random.default_rng(13)
    A=rng.normal(size=(50,8));A[:,7]=A[:,0]+A[:,1]
    B=rng.normal(size=(50,6));B[:,5]=A[:,0]
    J=np.column_stack((B,A))
    result=marginal_information(csr_matrix(J))
    units=np.r_[np.full(3,np.deg2rad(1.)),np.full(3,.1)]
    E=(B-A@np.linalg.pinv(A)@B)*units
    np.testing.assert_allclose(result["information_matrix"],E.T@E,atol=1e-10)
    assert result["reliable"] and result["effective_rank"]==5
    fixed=np.zeros(8,bool);fixed[0]=True;fixed[7]=True
    r=marginal_information(csr_matrix(J),fixed_nuisance=fixed)
    C=A[:,~fixed]
    E=(B-C@np.linalg.pinv(C)@B)*units
    np.testing.assert_allclose(r["information_matrix"],E.T@E,atol=1e-10)


def test_virtual_moments_preserve_pooled_and_shuffle():
    from spade_calib.calibration.ba_diagnostics import moment_factors,moments_from_factors,virtual_moments
    rng=np.random.default_rng(14)
    fs=[]
    for n in (5,8):
        mu=rng.normal(size=(n,3))
        A=rng.normal(size=(n,3,3));C=A@A.transpose(0,2,1)
        F=moment_factors(mu,C)
        a,b=moments_from_factors(F)
        np.testing.assert_allclose(a,mu,atol=1e-10)
        np.testing.assert_allclose(b,C,atol=1e-10)
        fs.append(F)
    controls,meta=virtual_moments(fs,shuffle_repeats=2)
    for name in ("J3_isotropic","J4_pooled","J6_real_mu","J5_shuffle_0","J5_shuffle_1"):
        np.testing.assert_allclose(meta["controls"][name]["mean_second_moment_trace"],
                                   meta["reference"]["mean_second_moment_trace"],atol=1e-10)
    for f,g in zip(fs,controls["J5_shuffle_0"]):
        np.testing.assert_array_equal(g,f[meta["permutations"]["J5_shuffle_0"][0 if len(f)==5 else 1]])


def test_information_with_ground_on_and_off():
    seg,X=synthetic()
    for weight in (0.,.1):
        r,_=solve_local_ba([seg],X,[1.],HandEyeSettings(),kind="scene",
                           ground_weight=weight,information=True,max_nfev=20)
        assert r["success"]
        for mode in ("raw","robust"):
            entry=r["information"]["solution"][mode]
            assert len(entry["all_nuisance_free"]["singular_values"])==6
            assert ("remove_ground_same_state" in entry)==(weight>0)
