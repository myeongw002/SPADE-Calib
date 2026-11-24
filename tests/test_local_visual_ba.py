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
