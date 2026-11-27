import numpy as np
from scipy.spatial.transform import Rotation
from spade_calib.selection.trajectory import MatchedTrajectories,MotionPairs
from spade_calib.selection.scale_segmentation import ScalePartition,build_frame_lag_pairs
from spade_calib.handeye.variable_scale import ScaleMotion,blocked_masks,fit
from spade_calib.handeye.solver import HandEyeSettings
from spade_calib.handeye.residuals import residuals


def synthetic():
    n=100;time=np.arange(n)*.1
    R=Rotation.from_rotvec(np.column_stack((.2*np.sin(time),.3*np.cos(time*.7),.2*np.sin(time*.4)))).as_matrix()
    p=np.column_stack((np.sin(time),np.cos(time*.7),time*.5))
    X=np.eye(4);X[:3,:3]=Rotation.from_rotvec([.1,-.2,.3]).as_matrix();X[:3,3]=[.3,-.2,.1]
    tau=(time[:-1]+time[1:]-time[0]-time[-1])/(time[-1]-time[0])
    metric=np.vstack((np.zeros((1,3)),np.cumsum(np.diff(p,axis=0)*np.exp(np.log(2.)+.12*tau)[:,None],axis=0)))
    L=R@X[:3,:3];l=metric+R@X[:3,3]
    s=MatchedTrajectories(np.rint(time*1e9).astype(np.int64),tuple(map(str,range(n))),L,l,R,p,n,0)
    return s,X


def test_integrated_scale_and_known_motion():
    s,X=synthetic();p=build_frame_lag_pairs(s,5);m=ScaleMotion(s,p,ScalePartition((0,len(s.timestamps_ns))))
    np.testing.assert_allclose(m.metric_tb([np.log(2.)],[0]),2*p.tb,atol=1e-12)
    np.testing.assert_allclose(m.raw(X,[np.log(2.)],[0]),residuals(p,X[:3,:3],X[:3,3],2.),atol=1e-12)
    np.testing.assert_allclose(m.raw(X,[np.log(2.)],[.12]),0,atol=1e-12)
    r=fit(m,X,[2.],HandEyeSettings(robust_loss="linear"),np.ones(p.count,bool),linear=True)
    assert r["success"] and r["cost"]<1e-12
    np.testing.assert_allclose(r["log_slopes"],[.12],atol=1e-7)
    np.testing.assert_allclose(r["T_camera_lidar"],X,atol=1e-7)


def test_blocked_edges_disjoint():
    s,_=synthetic();p=build_frame_lag_pairs(s,5);m=ScaleMotion(s,p,ScalePartition((0,len(s.timestamps_ns))))
    train,test=blocked_masks(m)
    edges=lambda mask:{k for i,j in p.indices[mask] for k in range(i,j)}
    assert train.any() and test.any() and not edges(train)&edges(test)
