import numpy as np
from scipy.spatial.transform import Rotation
from spade_calib.calibration.line_matching import transform_line,clip_projected_line,rank_candidates,reject_conflicts


def test_pose_and_projection():
    R=Rotation.from_euler("z",35,degrees=True).as_matrix()
    X=np.eye(4);X[:3,3]=[.2,0,0]
    p=np.array([[0,0,5],[0,1,5.]])
    t=np.array([2,3,1.])
    np.testing.assert_allclose(transform_line(p,R,t,R,t,X),p+X[:3,3],atol=1e-12)
    q=transform_line(p,np.eye(3),np.zeros(3),np.eye(3),np.array([1,0,0]),X)
    np.testing.assert_allclose(q,p+[-.8,0,0])
    K=np.array([[100.,0,50],[0,100,50],[0,0,1]])
    uv=clip_projected_line(np.array([[-5,0,5],[5,0,5]]),K,(100,100))
    np.testing.assert_allclose(uv,[[0,50],[99,50]])
    assert clip_projected_line(np.array([[0,0,-1],[1,0,-2]]),K,(100,100)) is None
    assert clip_projected_line(np.array([[0,0,0],[1,0,2]]),K,(100,100),min_length=1) is None  # ray projects to one pixel


def test_rank_gates_and_ambiguity():
    p=[[50,20],[50,100]]
    seg=[[51,20,51,100],[0,60,100,60],[50,120,50,180],[90,20,90,100]]
    a=rank_candidates(p,seg)
    assert a["status"]=="proposed" and a["selected_index"]==0 and a["gated_count"]==1
    a=rank_candidates(p,[[51,100,51,20],[49,20,49,100]])
    assert a["status"]=="ambiguous" and a["selected_index"] is None
    assert rank_candidates(p,[])["status"]=="no_candidate"


def test_conflicts():
    rows=[dict(frame=5,status="proposed",selected_index=1),dict(frame=5,status="proposed",selected_index=1),
          dict(frame=6,status="proposed",selected_index=1)]
    reject_conflicts(rows)
    assert [r["status"] for r in rows]==["conflict","conflict","proposed"]
