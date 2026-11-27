import numpy as np
from spade_calib.calibration.native_edge_validation import native_visibility_masks,anchor_edge_support
from spade_calib.calibration.uniform_range_grid import spherical_grid


def points(az,el,r):
    a=np.deg2rad(az);e=np.deg2rad(el)
    return np.asarray(r)[:,None]*np.c_[np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)]


def test_native_masks_distinguish_near_and_support_and_unknown():
    e=np.array([[0,1,2],[3,1,2],[4,5,2],[0,5,2]])
    eligible=np.array([1,1,1,1,0,0],bool)
    retained=np.array([1,0,1,0,0,0],bool)
    ghost=np.array([1,1,1,1,1,0],bool)
    a,b,bg=native_visibility_masks(e,eligible,retained,ghost)
    assert a.tolist()==[True,False,True,True]
    assert b.tolist()==[False,False,True,True]
    assert bg.tolist()==[False,False,False,False]


def test_anchor_horizontal_candidate_not_grid_winner_and_seam():
    p=points([-.3,-.1,.1,-.09,-.08],[.1]*5,[5,5,10,5.1,6])
    g,_=spherical_grid(p,np.ones(5,bool),el_min=-.2,el_max=.6)
    q=anchor_edge_support(p,np.array([1,3,4]),g,el_min=-.2,el_max=.6)
    assert q["center_supported"].tolist()==[True,True,False]
    assert q["h"].tolist()==[True,True,False]
    assert q["h_far"].tolist()==[2,2,-1]
    assert q["h_back"].tolist()==[0,0,-1]
    assert not q["v"].any()


def test_anchor_vertical_needs_observed_both_sides_and_gap():
    p=points([.1]*3,[-.3,-.1,.1],[10,5,5])
    ok=np.ones(3,bool)
    g,_=spherical_grid(p,ok,el_min=-.6,el_max=.6)
    q=anchor_edge_support(p,np.array([1]),g,el_min=-.6,el_max=.6)
    assert q["v"][0] and not q["h"][0]
    assert q["v_far"][0]==0 and q["v_back"][0]==2
    assert not anchor_edge_support(p,np.array([1]),g,el_min=-.6,el_max=.6,el_gap_deg=.15)["v"][0]
    g,_=spherical_grid(p,np.array([1,1,0],bool),el_min=-.6,el_max=.6)
    assert not anchor_edge_support(p,np.array([1]),g,el_min=-.6,el_max=.6)["v"][0]


def test_empty_and_unknown_center():
    p=np.empty((0,3));g=np.full((2,10),-1,int)
    assert not anchor_edge_support(p,np.empty(0,int),g)["h"].size
    p=points([.1],[.1],[5])
    q=anchor_edge_support(p,np.array([0]),g)
    assert not q["center_supported"][0] and not q["h"][0] and not q["v"][0]
