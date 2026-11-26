import numpy as np
from spade_calib.calibration.virtual_ring import virtual_rings,angular_grid,grid_stats
from spade_calib.calibration.multiview_edge_evidence import ring_edges


def xyz(az,r):
    a=np.deg2rad(az);r=np.asarray(r)
    return np.c_[r*np.cos(a),r*np.sin(a),np.zeros(len(a))]


def test_nearest_and_target_priority_and_gap_fill():
    p=xyz([.1,.1,.5,.9],[10,5,5,10])
    s=np.array([0,1,1,0]);r=np.zeros(4,int)
    t=angular_grid(p,r,s,0,bin_deg=.2,policy="target")
    f=angular_grid(p,r,s,0,bin_deg=.2,policy="fill")
    n=angular_grid(p,r,s,0,bin_deg=.2,policy="nearest")
    assert t[0,0]==f[0,0]==0 and n[0,0]==1
    assert t[0,2]==-1 and f[0,2]==2
    a=grid_stats(f,t,[0])
    assert a["added_bins"]==1 and a["replaced_target_bins"]==0
    assert grid_stats(n,t,[0])["replaced_target_bins"]==1


def test_gap_fill_enables_jump():
    p=xyz([0.1,.3,.5,1.1],[5.,5.,5.,10.])
    s=np.array([0,0,1,0]);r=np.zeros(4,int)
    t=angular_grid(p,r,s,0,bin_deg=.2,policy="target")
    f=angular_grid(p,r,s,0,bin_deg=.2,policy="fill")
    # with 0.7deg gate, target gap is .8deg, filled gap .6deg
    ti=t[t>=0];fi=f[f>=0]
    assert len(ring_edges(p[ti],r[ti],az_gap_deg=.7))==0
    e=ring_edges(p[fi],r[fi],az_gap_deg=.7)
    assert fi[e].tolist()==[[2,3,1]]


def test_virtual_subset_nearest_and_reject():
    a=np.linspace(-25,2,64);b=np.c_[np.arange(64),a,np.zeros((64,2))]
    theta=np.deg2rad([a[3],a[7],a[8]])
    p=10*np.c_[np.cos(theta),np.zeros(3),np.sin(theta)]
    rec=virtual_rings(p,b,[3,7,11],max_error_deg=.2)
    assert rec["ring"].tolist()==[3,7,-1]


def test_angular_seam_and_empty():
    p=xyz([359.9,.1],[5,5])
    g=angular_grid(p,np.zeros(2,int),np.zeros(2,int),0)
    assert g[0,0]==1 and g[0,-1]==0
    assert grid_stats(g,g,[0,1])["empty_rings"]==1
    empty=angular_grid(p[:0],np.empty(0,int),np.empty(0,int),0)
    assert np.all(empty==-1)
