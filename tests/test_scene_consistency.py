import numpy as np
from scipy.spatial.transform import Rotation
from spade_calib.calibration.scene_consistency import SceneSegment,scene_factors,fit_scene,predicted_relative
from spade_calib.handeye.solver import HandEyeSettings


def test_compact_factors_equal_explicit_spatial_error():
    rng=np.random.default_rng(3)
    q=rng.normal(size=(70,3));n=rng.normal(size=q.shape)
    n/=np.linalg.norm(n,axis=1,keepdims=True)
    D=rng.normal(size=(3,4))
    p,h=scene_factors(q,n)
    d=np.column_stack((q,np.ones(len(q))))@D.T
    np.testing.assert_allclose(np.sum((p@D.ravel())**2),np.mean(np.sum(d*d,axis=1)),rtol=1e-12)
    np.testing.assert_allclose(np.sum((h@D.ravel())**2),np.mean(np.sum(n*d,axis=1)**2),rtol=1e-12)


def test_scene_fit_recovers_shared_extrinsic_and_scale():
    rng=np.random.default_rng(5)
    X=np.eye(4);X[:3,:3]=Rotation.from_rotvec([.1,.2,-.3]).as_matrix()
    X[:3,3]=[.3,-.1,.2]
    R=Rotation.from_rotvec(rng.normal(0,.3,(25,3))).as_matrix()
    t=rng.normal(0,2,(25,3))
    B=X[:3,:3]@R@X[:3,:3].T
    scale=4.
    b=(t@X[:3,:3].T+X[:3,3]-B@X[:3,3])/scale
    q=[rng.normal(size=(40,3)) for _ in R]
    n=[rng.normal(size=(40,3)) for _ in R]
    n=[v/np.linalg.norm(v,axis=1,keepdims=True) for v in n]
    seg=SceneSegment(0,np.arange(len(R)),R,t,B,b,q,n,[np.zeros((40,2),int) for _ in R])
    np.testing.assert_allclose(seg.difference(X,scale),0,atol=1e-12)
    p,z=seg.maps(X,scale);np.testing.assert_allclose(p,z,atol=1e-12)
    start=X.copy();start[:3,3]+=[.1,-.07,.09]
    start[:3,:3]=Rotation.from_rotvec([.01,-.02,.01]).as_matrix()@start[:3,:3]
    for kind in ("point","plane","motion"):
        f=fit_scene([seg],start,np.array([3.9]),HandEyeSettings(),kind=kind,refit_scales=True)
        assert f["success"]
        np.testing.assert_allclose(f["final"]["T_camera_lidar"],X,atol=2e-6)
        np.testing.assert_allclose(f["final"]["scales"],[scale],atol=2e-6)


def test_planar_motion_does_not_gain_vertical_observability_from_scene():
    rng=np.random.default_rng(6)
    R=Rotation.from_rotvec(np.column_stack((np.zeros(20),np.zeros(20),np.linspace(-.3,.3,20)))).as_matrix()
    t=rng.normal(size=(20,3));t[:,2]=0
    q=[rng.normal(size=(30,3)) for _ in R]
    n=[v/np.linalg.norm(v,axis=1,keepdims=True) for v in q]
    seg=SceneSegment(0,np.arange(len(R)),R,t,R,t,q,n,[np.zeros((30,2),int) for _ in R])
    f=fit_scene([seg],np.eye(4),np.ones(1),HandEyeSettings(),kind="point")
    assert not f["success"] and f["reason"]=="rank_deficient"


def test_voxel_retains_nearest_actual_return_and_no_cap():
    from spade_calib.calibration.scene_consistency import voxel_return_indices
    p=np.array([[.01,0,0],[.09,0,0],[.12,0,0],[.8,0,0],[-.1,0,0]])
    ids=voxel_return_indices(p,.5)
    assert set(ids)=={1,3,4}
    many=np.column_stack((np.arange(500)*.2,np.zeros(500),np.zeros(500)))
    assert len(voxel_return_indices(many,.1))==500


def test_compact_support_equivalent_without_retained_points():
    rng=np.random.default_rng(17)
    q=[rng.normal(size=(40,3)) for _ in range(4)]
    n=[v/np.linalg.norm(v,axis=1,keepdims=True) for v in q]
    R=Rotation.from_rotvec(rng.normal(0,.1,(4,3))).as_matrix()
    t=rng.normal(size=(4,3))
    a=SceneSegment(0,np.arange(4),R,t,R,t,q,n,[np.zeros((40,2),int) for _ in q])
    b=SceneSegment(0,np.arange(4),R,t,R,t,[],[],[],
                   compact_factors=[scene_factors(x,y) for x,y in zip(q,n)],
                   point_counts=np.full(4,40))
    X=np.eye(4);X[:3,3]=[.2,-.3,.4]
    for kind in ("point","plane","motion"):
        np.testing.assert_allclose(a.raw_blocks(X,1.,kind),b.raw_blocks(X,1.,kind))
