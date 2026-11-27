import numpy as np
from scipy.spatial.transform import Rotation
from spade_calib.selection.trajectory import MotionPairs
from spade_calib.selection.scale_segmentation import ScalePartition
from spade_calib.handeye.multiscale import fit_segmented_motion_pairs
from spade_calib.handeye.solver import HandEyeSettings
from spade_calib.handeye.trajectory_oracle import kitti_oracle_pairs,rows,components,bias


def test_oracle_camera_baseline_and_frozen_pairs():
    rng=np.random.default_rng(11)
    n=30
    P=np.tile(np.eye(4),(n,1,1))
    P[:,:3,:3]=Rotation.from_rotvec(rng.normal(0,.2,(n,3))).as_matrix()
    P[:,:3,3]=rng.normal(0,1,(n,3))
    Tr=np.column_stack((Rotation.from_rotvec([.1,-.2,.3]).as_matrix(),[.3,.1,.2]))
    K=np.array([[100.,0,40],[0,100,30],[0,0,1]])
    cal={"Tr":Tr,"P0":np.column_stack((K,np.zeros(3))),"P2":np.column_stack((K,K@np.array([.2,0.,0.])))}
    i=np.arange(n-1)
    p=MotionPairs(np.tile(np.eye(3),(n-1,1,1)),np.zeros((n-1,3)),
                  np.tile(np.eye(3),(n-1,1,1)),np.zeros((n-1,3)),
                  np.column_stack((i,i+1)),i.astype(np.int64))
    gt,X=kitti_oracle_pairs(P[:,:3,:].reshape(-1,12),cal,2,np.arange(n),p)
    np.testing.assert_allclose(X[:3,3],Tr[:,3]+[.2,0.,0.])
    np.testing.assert_allclose(rows(gt,np.zeros(n-1,dtype=int),np.ones(1),X),0,atol=1e-10)
    start=X.copy();start[:3,3]+=.2
    mask=np.ones(n-1,dtype=bool);mask[0]=False
    f=fit_segmented_motion_pairs(gt,ScalePartition((0,n)),start,pair_mask=mask,
                                  initial_scales=np.ones(1),freeze_pair_mask=True,fix_scales=True)
    np.testing.assert_array_equal(f.pair_mask,mask)
    np.testing.assert_allclose(f.T_camera_lidar,X,atol=1e-7)
    assert f.diagnostics["parameter_count"]==6


def test_translation_decomposition_and_bias_closure():
    rng=np.random.default_rng(2)
    n=24
    R=Rotation.from_rotvec(rng.normal(0,.2,(n,3))).as_matrix()
    t=rng.normal(0,1,(n,3))
    i=np.arange(n)
    g=MotionPairs(R,t,R,t,np.column_stack((i,i+1)),i.astype(np.int64))
    a=MotionPairs(R,t+.01,R,t/5.+.02,g.indices,g.starts_ns)
    groups=np.zeros(n,dtype=int)
    comp=components(a,g,groups,np.array([5.]),np.eye(4))
    np.testing.assert_allclose(comp["camera_motion_and_scale"],.1)
    np.testing.assert_allclose(comp["lidar_translation"],-.01)
    b=bias(a,groups,np.eye(4),np.array([5.]),HandEyeSettings(),terms=comp)["frozen_IRLS"]
    np.testing.assert_allclose(sum(np.array(x) for x in b["component_contributions_camera_m"].values()),
                               b["predicted_translation_error_camera_m"],atol=1e-9)
