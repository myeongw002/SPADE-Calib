import numpy as np
from spade_calib.calibration.uniform_range_grid import spherical_grid,grid_edges
from spade_calib.calibration.surface_visibility import ray_triangle_depth,visibility_depths,filter_occluded


def test_ray_triangle_intersection_and_gates():
    d=np.array([[1.,0,0],[1.,0,0]])
    a=np.array([[5.,-.1,-.1],[5.,.1,.1]])
    b=np.array([[5.,.1,-.1],[5.,.3,.1]])
    c=np.array([[5.,0,.1],[5.,.2,.3]])
    t=ray_triangle_depth(d,a,b,c)
    assert np.isclose(t[0],5) and np.isinf(t[1])
    assert np.isinf(ray_triangle_depth(d,a,b,c,max_edge_m=.1)).all()


def make_hole():
    az=np.deg2rad([-.1,.3,-.1,.3,.1])
    el=np.deg2rad([-.1,-.1,.3,.3,.1])
    unit=np.c_[np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)]
    return unit*np.array([5,5,5,5,10])[:,None]


def test_surface_rejects_hole_at_azimuth_seam_and_protects_target():
    p=make_hole();ok=np.ones(5,bool)
    g,_=spherical_grid(p,ok,el_min=-.2,el_max=.6)
    ds,dt=visibility_depths(p,g,ok,el_min=-.2,el_max=.6)
    assert ds[4]<6 and dt[4]<6
    for d in (ds,dt):
        keep,info=filter_occluded(p,ok,np.zeros(5,bool),d)
        assert keep.tolist()==[True,True,True,True,False]
        assert info["removed"]==1
        keep,info=filter_occluded(p,ok,np.array([0,0,0,0,1],bool),d)
        assert keep.all() and info["protected_conflicts"]==1
        g2,_=spherical_grid(p,np.array([1,1,1,1,0],bool),el_min=-.2,el_max=.6)
        assert 4 not in g2
        assert set(g2[g2>=0])=={0,1,2,3}


def test_surface_does_not_extrapolate_from_one_side_or_join_long_edges():
    p=make_hole();ok=np.array([1,0,1,0,1],bool)
    g,_=spherical_grid(p,ok,el_min=-.2,el_max=.6)
    _,dt=visibility_depths(p,g,ok,el_min=-.2,el_max=.6)
    assert np.isinf(dt[4])
    ok=np.ones(5,bool);g,_=spherical_grid(p,ok,el_min=-.2,el_max=.6)
    _,dt=visibility_depths(p,g,ok,el_min=-.2,el_max=.6,max_edge_m=.01)
    assert np.isinf(dt[4])


def test_empty_input():
    p=np.empty((0,3));ok=np.empty(0,bool)
    g,_=spherical_grid(p,ok)
    ds,dt=visibility_depths(p,g,ok,el_min=-26,el_max=4)
    assert ds.size==dt.size==0
    assert filter_occluded(p,ok,ok,dt)[1]["removed"]==0
