import numpy as np
from scipy.spatial.transform import Rotation

from spade_calib.calibration.ground import Plane
from spade_calib.calibration.segmented_ground import (
    SegmentGround, GroundSettings, extract_segment_ground, ground_residuals, fit_ground_handeye,
)
from spade_calib.selection.scale_segmentation import ScalePartition
from spade_calib.selection.trajectory import MotionPairs


def test_metric_offset_scale_ratio_and_sign():
    X = np.eye(4)
    X[:3,3] = [0.,0.,.3]
    m = SegmentGround(0,Plane([0,0,1],2.),Plane([0,0,1],1.7))
    np.testing.assert_allclose(ground_residuals([m],X,np.array([5.]),np.array([5.])),0,atol=1e-12)
    np.testing.assert_allclose(ground_residuals([m],X,np.array([10.]),np.array([5.]))[0,3],1.7)


def test_ground_roi_rejects_roof_and_fits_common_support():
    rng = np.random.default_rng(4)
    # KITTI axes: LiDAR forward/left/up -> camera right/down/forward.
    X = np.eye(4)
    X[:3,:3] = [[0,-1,0],[0,0,-1],[1,0,0]]
    X[:3,3] = [.1,.1,.2]
    l = np.column_stack((rng.uniform(4,20,1200),rng.uniform(-2.5,2.5,1200),np.full(1200,-1.5)))
    c = l[::3]@X[:3,:3].T+X[:3,3]
    roof = c.copy(); roof[:,1] = -2.
    match, report, cp, lp = extract_segment_ground(np.r_[c,roof],l,X,0)
    assert report["accepted"] and match is not None
    assert np.all(cp[:,1]>0) and np.all(lp[:,2]<0)
    np.testing.assert_allclose(ground_residuals([match],X,np.ones(1),np.ones(1)),0,atol=1e-8)
    match, report, *_ = extract_segment_ground(roof,l,X,0)
    assert match is None and not report["accepted"]


def test_ground_complements_yaw_motion_vertical_translation():
    rng = np.random.default_rng(3)
    N = 80
    X = np.eye(4)
    # Use abstract z-up sensor coordinates to isolate observability.
    X[:3,:3] = Rotation.from_rotvec([.03,-.02,.01]).as_matrix()
    X[:3,3] = [.2,-.1,.4]
    R = X[:3,:3]
    Ra = Rotation.from_rotvec(np.column_stack((np.zeros(N),np.zeros(N),
                                               rng.uniform(-.3,.3,N)))).as_matrix()
    Rb = R@Ra@R.T
    s = np.array([5.,8.])
    i = np.r_[np.arange(40),100+np.arange(40)]
    ta = np.column_stack((rng.normal(1.,.2,N),rng.normal(0.,.3,N),np.zeros(N)))
    tB = (ta@R.T-np.einsum("nij,j->ni",Rb-np.eye(3),X[:3,3]))/np.repeat(s,40)[:,None]
    p = MotionPairs(Ra,ta,Rb,tB,np.column_stack((i,i+1)),i.astype(np.int64))
    normal = R@np.array([0.,0.,1.])
    matches = [SegmentGround(k,Plane([0,0,1],1.5),Plane(normal,1.5-normal@X[:3,3]))
               for k in range(2)]
    X0 = X.copy(); X0[:3,3] += .2*normal
    for refit in [False,True]:
        fit = fit_ground_handeye(p,ScalePartition((0,100,200)),np.ones(N,dtype=bool),
                                 X0,s,matches,refit_scales=refit)
        assert fit["success"]
        assert not fit["HE_only_observability_at_baseline"]["full_rank"]
        np.testing.assert_allclose(fit["final"]["T_camera_lidar"],X,atol=1e-6)
        np.testing.assert_allclose(fit["final"]["scales"],s,atol=1e-6)


def test_missing_ground_refuses_fit():
    p = MotionPairs(np.empty((0,3,3)),np.empty((0,3)),np.empty((0,3,3)),
                    np.empty((0,3)),np.empty((0,2),dtype=int),np.empty(0,dtype=np.int64))
    result = fit_ground_handeye(p,ScalePartition((0,10)),np.empty(0,dtype=bool),
                                np.eye(4),np.ones(1),[])
    assert result["reason"]=="no_ground_pairs"
