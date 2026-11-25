
import numpy as np
import pytest
from test_local_visual_ba import synthetic
from spade_calib.calibration.local_visual_ba import solve_local_ba
from spade_calib.handeye.solver import HandEyeSettings


def test_fixed_scale_removed_from_state():
    seg,X=synthetic()
    x=np.zeros(7+seg.size)
    r,_=solve_local_ba([seg],X,[1.],HandEyeSettings(),kind="he",
                       initial_parameters=x,fixed_scales=[1.],max_nfev=20)
    assert r["success"] and r["final"]["cost"]<1e-15
    assert r["variables"]==6+seg.size
    assert r["fixed_scale_outside_original_bounds"]==[]
    r,_=solve_local_ba([seg],X,[1.],HandEyeSettings(),kind="he",
                       initial_parameters=x,fixed_scales=[1.25],max_nfev=5)
    np.testing.assert_allclose(r["final"]["scales"],[1.25],rtol=1e-13)
    assert r["fixed_scale_outside_original_bounds"]==[0]
    assert not any(v["component"]==6 for v in r["active_global_bounds"])
    with pytest.raises(ValueError):
        solve_local_ba([seg],X,[1.],HandEyeSettings(),kind="he",fixed_scales=[0.])


def test_archive_cost_and_warm_free():
    seg,X=synthetic()
    x=np.zeros(7+seg.size);x[3]=.03;x[6]=.01
    r,_=solve_local_ba([seg],X,[1.],HandEyeSettings(),kind="he",
                       scale_diagnostic_state=x,archive_cost_only=True)
    f,_=solve_local_ba([seg],X,[1.],HandEyeSettings(),kind="he",
                       initial_parameters=x,max_nfev=30)
    assert r["diagnostic_only"]
    np.testing.assert_allclose(r["final"]["cost"],f["initial"]["cost"])
    assert f["final"]["cost"]<f["initial"]["cost"]
