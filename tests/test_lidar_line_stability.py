import numpy as np
import pytest
from dataclasses import replace
from spade_calib.calibration.lidar_line_stability import StabilitySettings,evaluate,fit_subset


def corner():
    x,z=np.meshgrid(np.linspace(.01,.6,12),np.linspace(0,2,21))
    a=np.column_stack((x.ravel(),np.zeros(x.size),z.ravel()))
    b=np.column_stack((np.zeros(x.size),x.ravel(),z.ravel()))
    # Each scan covers the full surface in this ideal synthetic test.
    p=[np.tile(a,(6,1)),np.tile(b,(6,1))]
    s=[np.repeat(np.arange(6),len(a)) for _ in range(2)]
    return np.array([[0,0,.2],[0,0,1.8]]),p,s


def test_exact_and_temporal_drift():
    r,p,s=corner();cfg=replace(StabilitySettings(),bootstrap=12)
    result=evaluate(r,p,s,np.arange(6),cfg)
    assert result["odd_even"]["comparison"]["rmse_m"]<1e-10
    assert result["temporal"]["comparison"]["rmse_m"]<1e-10
    assert result["bootstrap"]["rmse_m_p90"]<1e-10
    p[0][s[0]>=3,1]+=.08
    p[1][s[1]>=3,0]+=.08
    result=evaluate(r,p,s,np.arange(6),cfg)
    assert result["temporal"]["comparison"]["rmse_m"]==pytest.approx(np.sqrt(2)*.08)
    assert result["bootstrap"]["valid"]>0


def test_insufficient_scan_and_weight_gate():
    r,p,s=corner();cfg=replace(StabilitySettings(),bootstrap=5)
    s=[np.zeros(len(q),int) for q in p]
    result=evaluate(r,p,s,np.arange(6),cfg)
    assert result["odd_even"]["status"]=="subset_failed"
    assert result["bootstrap"]["valid"]==0
    f,reason,_=fit_subset(p[0][:3],np.arange(3),np.full(3,100),cfg)
    assert f is None and reason=="insufficient_points_or_scans"


def test_disjoint_temporal_support():
    r,p,s=corner();cfg=replace(StabilitySettings(),bootstrap=5)
    # Time subsets each contain a short but different part of the edge.
    for i in range(2):
        p[i][s[i]<3,2]*=.4
        p[i][s[i]>=3,2]=1.2+.4*p[i][s[i]>=3,2]
    result=evaluate(r,p,s,np.arange(6),cfg)
    assert result["temporal"]["comparison"] is None
