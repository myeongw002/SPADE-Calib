import numpy as np
from scipy.spatial.transform import Rotation
from spade_calib.calibration.multiview_virtual_ring import reframe_points,view_indices,union_view_edges


def test_reframe_world_convention_and_roundtrip():
    p=np.array([[1.,2.,3.],[4.,-2.,1.]])
    ra=Rotation.from_euler("xyz",[10,20,30],degrees=True).as_matrix()
    rv=Rotation.from_euler("xyz",[-10,5,90],degrees=True).as_matrix()
    ta=np.array([2.,3.,4.]);tv=np.array([-1.,5.,2.])
    q=reframe_points(p,ra,ta,rv,tv)
    assert np.allclose(q@rv.T+tv,p@ra.T+ta)
    assert np.allclose(reframe_points(q,rv,tv,ra,ta),p)


def test_view_selection():
    assert view_indices(21,10,5).tolist()==[0,5,10,15,20]
    assert view_indices(21,10,1).tolist()==list(range(21))
    assert view_indices(8,1,5).tolist()==[1,6]


def test_union_counts_once_per_view():
    a=np.array([[1,2,3],[1,3,2],[4,2,3]])
    b=np.array([[1,0,2],[5,4,3]])
    ids,c=union_view_edges([a,b],6)
    assert ids.tolist()==[1,4,5] and c.tolist()==[0,2,0,0,1,1]
    assert union_view_edges([np.empty((0,3),int)],6)[0].size==0
