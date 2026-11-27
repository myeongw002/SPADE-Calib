import numpy as np
import pytest
from spade_calib.calibration.edge_detector_pilot import raster_segments,line_metrics


def test_exact_and_shifted():
    uv=[[10,10],[10,50]]
    exact=raster_segments([[10,10,10,50]],(64,64))
    a=line_metrics(exact,uv)
    assert a["thresholds"]["1.0"]["coverage"]==1.
    assert a["nearest_rmse_px"]==0.
    shifted=raster_segments([[13,10,13,50]],(64,64))
    b=line_metrics(shifted,uv)
    assert b["thresholds"]["2.0"]["coverage"]==0.
    assert b["thresholds"]["3.0"]["coverage"]==1.
    assert b["nearest_rmse_px"]==3.


def test_empty_and_fragmented():
    uv=[[10,10],[10,50]]
    a=line_metrics(np.zeros((64,64),np.uint8),uv)
    assert a["nearest_rmse_px"] is None
    assert a["thresholds"]["2.0"]["coverage"]==0.
    assert a["thresholds"]["2.0"]["longest_uncovered_px"]==40.
    b=line_metrics(raster_segments([[10,10,10,20],[10,40,10,50]],(64,64)),uv)
    assert 0<b["thresholds"]["2.0"]["coverage"]<1
    assert b["thresholds"]["2.0"]["longest_uncovered_px"]>10
    with pytest.raises(ValueError):
        line_metrics(np.zeros((64,64)),[[10,10],[100,100]])
