import numpy as np
from spade_calib.calibration.ring_smoothness import ring_smoothness,select_top


def test_constant_range_circle_and_seam():
    az=np.deg2rad(np.arange(0,360,.5))
    p=10*np.c_[np.cos(az),np.sin(az),np.zeros(len(az))]
    score,clean=ring_smoothness(p,np.zeros(len(p),int))
    expected=1-np.cos(np.deg2rad(.5)*np.arange(1,6)).mean()
    assert np.all(clean)
    assert np.allclose(score,expected)


def test_gap_and_unknown_ring():
    az=np.deg2rad(np.arange(0,360,.5))
    p=10*np.c_[np.cos(az),np.sin(az),np.zeros(len(az))]
    ring=np.zeros(len(p),int);ring[0]=-1
    score,clean=ring_smoothness(p,ring)
    assert np.isnan(score[0]) and not clean[0]
    assert np.isnan(score[1]) and np.isnan(score[-1])
    assert np.isfinite(score[20])


def test_jump_rejected_but_score_retained():
    az=np.deg2rad(np.arange(40)*.3)
    r=np.r_[np.full(20,5.),np.full(20,10.)]
    p=r[:,None]*np.c_[np.cos(az),np.sin(az),np.zeros(len(az))]
    score,clean=ring_smoothness(p,np.zeros(len(p),int))
    assert np.isfinite(score[19]) and not clean[19] and not clean[20]
    assert clean[10] and score[19]>score[10]
    assert np.all(clean[select_top(score,clean,3)])


def test_top_budget_and_ties():
    s=np.array([2.,np.nan,3.,3.,1.])
    assert select_top(s,[1,1,1,1,0],2).tolist()==[2,3]
    assert select_top(s,[0,0,0,0,0],4).size==0
    assert select_top(s,[1,1,1,1,1],0).size==0
