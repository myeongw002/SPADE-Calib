import numpy as np
import pytest
from spade_calib.calibration.lidar_line_support import sample_support,support_metrics,common_interval,interval_error,manual_matches,paired_merge_error


def test_dual_support_and_threshold():
    x=np.linspace(0,1,101)
    ref=np.array([[0,0,0],[1,0,0]])
    p=np.column_stack((x,np.full(len(x),.02),np.zeros(len(x))))
    q=np.column_stack((x,np.zeros(len(x)),np.full(len(x),.04)))
    t,xyz,d=sample_support(ref,p,q)
    m=support_metrics(t,xyz,d,.03,.06)
    assert m["accepted"] and m["length_coverage"]==pytest.approx(1)
    assert m["longest_supported_span_m"]==pytest.approx(1)
    assert not support_metrics(t,xyz,d,.01,.06)["accepted"]
    m2=support_metrics(t,xyz,d[:,::-1],.03,.06)
    assert m2==m


def test_gap_and_empty_support():
    x=np.linspace(0,1,101)
    ref=np.array([[0,0,0],[1,0,0]])
    p=np.column_stack((x,np.full(len(x),.02),np.zeros(len(x))))
    q=p[(x<.25)|(x>.75)].copy()
    t,xyz,d=sample_support(ref,p,q)
    m=support_metrics(t,xyz,d,.03,.06,min_span=.1)
    assert len(m["accepted_spans"])==2
    assert m["length_coverage"]<.7
    assert m["max_unsupported_gap_upper_m"]>.35
    assert support_metrics(t,xyz,d,.001,.002)["accepted_spans"]==[]


def test_same_interval_and_union():
    ref=np.array([[0,0,0],[2,0,0]])
    a=np.array([[.2,.1,0],[1.2,.1,0]])
    b=np.array([[.8,.2,0],[1.8,.2,0]])
    interval=common_interval(ref,[a,b])
    assert interval==pytest.approx((.8,1.2))
    assert interval_error(ref,a,interval)["rmse_m"]==pytest.approx(.1)
    assert interval_error(ref,b,interval)["rmse_m"]==pytest.approx(.2)
    m=manual_matches(ref,[a,b])
    assert m["union_overlap_fraction"]==pytest.approx(.8)
    assert paired_merge_error(ref,[a],[b])["length_m"]==pytest.approx(.4)
