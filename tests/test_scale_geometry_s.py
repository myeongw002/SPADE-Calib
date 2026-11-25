import numpy as np
from scipy.sparse import csr_matrix
from spade_calib.calibration.ba_diagnostics import scale_coupling_information
from spade_calib.calibration.scale_geometry_synthetic import (
    CASES,UNITS,synthetic_case,residual,central_jac,observability,
    analytic_scale_jacobian,line_residual,run_trial)
from spade_calib.calibration.crossmodal_lines import perturb


def test_nested_projection_matches_dense():
    rng=np.random.default_rng(2)
    J=rng.normal(size=(60,12))
    J[:,6]=J[:,3]+.2*J[:,6]
    J[:,8]=J[:,4]+.1*J[:,8]
    result=scale_coupling_information(csr_matrix(J),2)
    B=J[:,:6]*UNITS[:6]
    def H(A):
        E=B-A@np.linalg.pinv(A)@B
        return E.T@E
    F=H(J[:,8:]);U=H(J[:,6:])
    assert result["reliable"]
    np.testing.assert_allclose(result["fixed_scale"]["information_matrix"],F,atol=1e-9)
    np.testing.assert_allclose(result["free_scale"]["information_matrix"],U,atol=1e-9)
    np.testing.assert_allclose(result["information_loss_matrix"],F-U,atol=1e-9)


def test_scale_null_and_analytic_derivative():
    for case in CASES:
        d=synthetic_case(case)
        X=d["X"];s=d["scale"]
        def f(x,arm):
            return residual(d,perturb(X,x[:6]*UNITS[:6]),s*np.exp(x[6]*UNITS[6]),arm)
        for arm in ("single","multiview","motion_glim_multiview","motion_multiview"):
            assert np.linalg.norm(f(np.zeros(7),arm))<1e-9
        J=central_jac(lambda x:f(x,"multiview"),np.zeros(7))
        np.testing.assert_allclose(J[:,6],analytic_scale_jacobian(d,X,s),atol=1e-7,rtol=1e-6)
        np.testing.assert_allclose(central_jac(lambda x:f(x,"single"),np.zeros(7))[:,6],0,atol=1e-12)
        if case=="stationary_one": assert np.linalg.norm(J[:,6])<1e-12
        else: assert np.linalg.norm(J[:,6])>1e-3
        if case=="general_diverse": assert observability(J)["rank"]==7
    d=synthetic_case("general_diverse")
    np.testing.assert_allclose(line_residual(d,d["X"],1.2,"glim"),line_residual(d,d["X"],2.4,"glim"))


def test_noiseless_general_recovery():
    r=run_trial("general_diverse")
    a=r["arms"]["motion_multiview"]
    assert a["converged"] and a["observability_at_truth"]["rank"]==7
    assert a["translation_error_m"]<1e-5
    assert a["rotation_error_deg"]<1e-5
    assert abs(a["relative_scale_error"])<1e-6


def test_archived_diagnostic_no_refit():
    from test_local_visual_ba import synthetic
    from spade_calib.calibration.local_visual_ba import solve_local_ba
    from spade_calib.handeye.solver import HandEyeSettings
    seg,X=synthetic()
    x=np.zeros(7+seg.size)
    r,local=solve_local_ba([seg],X,[1.],HandEyeSettings(),kind="he",
                          scale_diagnostic_state=x)
    assert r["diagnostic_only"] and r["final"]["cost"]<1e-15
    np.testing.assert_array_equal(local[0],x[7:])
    for state in ("common_initial","archived_solution"):
        for mode in ("raw","robust"):
            assert r["scale_information"][state][mode]["reliable"]
