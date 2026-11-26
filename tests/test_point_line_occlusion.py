import numpy as np
from spade_calib.calibration.point_line_occlusion import (
    project_grid, occlusion_edges, match_points, evaluate_fixed,
)

def test_grid_observed_depth_jump_and_unknown():
    # 3 occupied cells along x, middle foreground near-side edge.
    K=np.array([[100.,0,0],[0,100.,0],[0,0,1.]])
    p=np.array([[.1,.1,5],[.3,.1,5],[1.1,.2,10],[.6,.2,10]])
    q=project_grid(p,np.eye(4),K,(20,20),np.ones(4,bool),cell=4)
    # Near point at u=6 wins cell 1 against farther point at u=6.
    assert set(q["ids"])=={0,1,2}
    edge=occlusion_edges(q)
    assert q["ids"][edge.any(1)].tolist()==[1]
    q2=project_grid(p[:2],np.eye(4),K,(20,20),np.ones(2,bool),cell=4)
    assert not occlusion_edges(q2).any()

def test_matching_finite_unique_and_gt_fixed():
    uv=np.array([[10.,3],[10,12],[10,21],[10,30],[10,90]])
    p=np.c_[uv,np.ones(len(uv))]
    q=dict(ids=np.arange(len(uv)),uv=uv,non_ground=np.ones(len(uv),bool))
    d=np.tile([True,False,False,False],(len(uv),1))
    s=np.array([[[10.,0],[10.,32]]])
    groups,stats=match_points(q,d,s,gate=3.)
    assert stats["retained_points"]==4
    assert stats["matched_lines"]==1
    assert 4 not in groups[0]["point_indices"]
    ev=evaluate_fixed(p,groups,s,np.eye(4),np.eye(3),(100,100))
    assert ev[0]["inside"].all() and np.allclose(ev[0]["signed_px"],0)
    _,stats=match_points(q,d,np.repeat(s,2,axis=0),gate=3.)
    assert stats["retained_points"]==0

def test_empty():
    q=project_grid(np.empty((0,3)),np.eye(4),np.eye(3),(20,20),np.empty(0,bool))
    groups,stats=match_points(q,occlusion_edges(q),np.empty((0,2,2)))
    assert not groups and stats["retained_points"]==0
