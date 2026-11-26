import numpy as np
from spade_calib.calibration.uniform_range_grid import spherical_grid,grid_edges


def points(az,el,r):
    a=np.deg2rad(az);e=np.deg2rad(el);r=np.asarray(r)
    return r[:,None]*np.c_[np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.sin(e)]


def test_nearest_and_unknown_and_seam():
    p=points([.1,.1,359.9,.1],[.1,.1,.1,5],[10,5,8,5])
    g,m=spherical_grid(p,np.ones(4,bool),el_min=0,el_max=1)
    assert g[0,0]==1 and g[0,-1]==2
    assert (g>=0).sum()==2 and m["eligible_outside_fov"]==1


def test_horizontal_and_vertical_independent():
    p=points([.1,.3,.5,.1,.1,.1],[.1,.1,.1,.3,.5,.7],[5,5,10,5,5,10])
    g,_=spherical_grid(p,np.ones(6,bool),el_min=0,el_max=1)
    h,v=grid_edges(p,g)
    assert h.tolist()==[[1,2,0]]
    assert v.tolist()==[[4,5,3]]


def test_vertical_unknown_gap_and_no_wrap():
    p=points([.1,.1,.1],[.1,.3,1.3],[5,5,10])
    g,_=spherical_grid(p,np.ones(3,bool),el_min=0,el_max=2)
    h,v=grid_edges(p,g,el_gap_deg=.8)
    assert len(h)==len(v)==0
    p=points([.1,.1,.1],[.1,.3,.5],[5,10,10])
    g,_=spherical_grid(p,np.ones(3,bool),el_min=0,el_max=1)
    assert len(grid_edges(p,g)[1])==0


def test_empty_grid():
    p=np.empty((0,3));g,_=spherical_grid(p,np.empty(0,bool))
    assert np.all(g==-1)
    assert all(x.shape==(0,3) for x in grid_edges(p,g))
