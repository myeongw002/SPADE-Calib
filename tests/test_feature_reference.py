import numpy as np
from spade_calib.calibration.feature_reference import select_tiles,angular_coordinates,local_evidence,blinded_samples

def test_tiles_disjoint_and_reproducible():
    uv=np.array([[5.,5.],[15.,15.],[230.,50.]])
    a=select_tiles((248,448),uv,seed=4)
    assert a==select_tiles((248,448),uv,seed=4)
    assert len(a)==4
    assert len({tuple(t["box"]) for t in a})==4

def test_angular_seam_and_unknown():
    a=np.array([[179.9,0,5],[-179.9,.1,10],[0,0,4]])
    ids,d=local_evidence(a,np.array([179.95,0,5]),az_radius=.3,el_radius=.2)
    assert ids.tolist()==[0,1]
    assert np.allclose(d[:,0],[-.05,.15])
    assert np.allclose(d[:,2],[5,10])
    assert np.allclose(angular_coordinates(np.array([[1.,0,0]])),[[0,0,1]])

def test_sample_overlap_not_double_reviewed():
    a={"a":np.array([1,2]),"b":np.array([2,3])}
    rows,meta=blinded_samples(a,per_arm=3)
    assert len(rows)==3
    assert dict(rows)[2]==["a","b"]
    assert meta["a"]=={"population":2,"sample":2}
