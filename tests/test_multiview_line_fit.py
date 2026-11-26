import numpy as np
from scipy.spatial.transform import Rotation
from spade_calib.calibration.multiview_line_fit import camera_points,glim_points,solve_fixed,prepare_tracks
from test_local_visual_ba import synthetic
from spade_calib.handeye.solver import HandEyeSettings


def test_camera_glim_equivalent_at_consistent_motion():
    X=np.eye(4);X[:3,:3]=Rotation.from_rotvec([.1,.2,-.05]).as_matrix();X[:3,3]=[.3,-.1,.2]
    A=Rotation.from_rotvec([.02,.1,.03]).as_matrix();a=np.array([1.,.1,.2])
    B=X[:3,:3]@A@X[:3,:3].T
    b=X[:3,:3]@a+X[:3,3]-B@X[:3,3]
    p=np.array([[1.,0,8],[1.,2,8]])
    np.testing.assert_allclose(camera_points(p,X,B,b/1.2,1.2),glim_points(p,X,A,a),atol=1e-12)
    assert np.linalg.norm(camera_points(p,X,B,b/1.2,1.1)-glim_points(p,X,A,a))>.01


def test_fixed_solver_controls():
    seg,X=synthetic();K=np.array([[500.,0,300],[0,500,200],[0,0,1.]])
    p=np.array([[1.,-1.,8],[1.,1.,8]])
    views=[]
    for j in (0,2,3):
        q=camera_points(p,X,seg.R[j],seg.t[j],1.)
        uv=(q@K.T)[:,:2]/q[:,2,None]
        views.append(dict(frame=j,view=j,uv=uv))
    tracks=[dict(id="a",segment=0,xyz=np.linspace(p[0],p[1],5),views=views)]
    for mode in ("none","anchor","glim","camera"):
        result=solve_fixed([seg],[(seg.R,seg.t,seg.points)],tracks,X,[1.],HandEyeSettings(),np.zeros(7),K,
                           mode=mode,weight=0 if mode=="none" else .1,max_nfev=20)
        assert result["success"] and result["final"]["cost"]<1e-14
    start=np.zeros(7);start[6]=.01
    result=solve_fixed([seg],[(seg.R,seg.t,seg.points)],tracks,X,[1.],HandEyeSettings(),start,K,
                       mode="camera",weight=.1,fixed_scale=True)
    np.testing.assert_allclose(result["final"]["scales"],np.exp(.01))


def test_track_support_and_anchor():
    seg,_=synthetic()
    src=dict(id="a",frame=0,xyz_lidar=[[1,-1,8],[1,1,8]],uv_image=[[50,20],[50,80]],split="train",manual_identity_confirmed=True)
    r=dict(id="r",source_id="a",anchor_frame=0,frame=2,xyz_lidar=src["xyz_lidar"],uv_image=[[40,20],[40,80]],split="train",manual_identity_confirmed=True)
    tracks,skipped=prepare_tracks([seg],[src],[r,dict(r,id="bad",frame=9)])
    assert len(tracks)==1 and [v["frame"] for v in tracks[0]["views"]]==[0,2]
    assert skipped[0]["reason"]=="not_in_same_archived_BA_support"
