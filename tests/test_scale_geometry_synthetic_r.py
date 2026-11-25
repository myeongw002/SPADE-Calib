
import numpy as np
from spade_calib.calibration.scale_geometry_synthetic_r import dataset,residual,run_trial,ARMS


def test_exact_ground_and_motion_with_segment_scale():
    for case in ("planar","general"):
        records,s,X=dataset(case,"clean",17)
        for arm in ARMS:
            assert np.linalg.norm(residual(records,X,s,arm))<1e-9
        # Incorrect global scale must not explain exact nonzero translations.
        assert np.linalg.norm(residual(records,X,np.full(3,1.2),"motion"))>1.


def test_paired_noise():
    a,s,X=dataset("planar","both_mild",17)
    b,_,_=dataset("planar","lidar_high",17)
    for p,q in zip(a,b):
        for key in ("R","b","uv","dc","dl"):
            np.testing.assert_array_equal(p[key],q[key])
        assert not np.allclose(p["A"][1:],q["A"][1:])
        np.testing.assert_array_equal(p["A"][0],q["A"][0])


def test_general_noiseless_multi_and_oracle_recovery():
    r=run_trial("general","clean",17,True)
    for model in ("multi","oracle"):
        a=r["arms"][model]["motion_multiview"]
        assert a["converged"]
        assert a["information_at_solution"]["rank"]==a["information_at_solution"]["variables"]
        assert a["translation_error_m"]<1e-5
        assert a["scale_relative_rmse"]<1e-6
    assert r["arms"]["global"]["motion_multiview"]["scale_relative_rmse"]>.05
