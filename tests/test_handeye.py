"""Synthetic numerical calibration checks with controlled ground-truth transforms."""
from __future__ import annotations

import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spade_calib.handeye import (
    HandEyeSettings, fit_full_trajectory, fit_motion_pairs,
    analyze_local_uncertainty, block_bootstrap_uncertainty,
)
from spade_calib.handeye.residuals import residuals
from spade_calib.handeye.report import report_handeye, write_handeye_report
from spade_calib.selection.trajectory import (
    MatchedTrajectories, build_motion_pairs, SelectionSettings,
)


def make_samples(kind='rich', *, noise=0., seed=22):
    n = 501
    times = np.arange(n, dtype=np.float64) * .1
    ns = (1_800_000_000_000_000_000 +
          (times * 1e9).astype(np.int64))
    if kind == 'rich':
        ang = np.stack((.24 * np.sin(.27 * times), .36 * np.sin(.21 * times),
                        .23 * np.cos(.37 * times)), axis=1)
        Lr = Rotation.from_euler('xyz', ang).as_matrix()
        Lp = np.stack((.2 * times, np.sin(.4 * times), .5 * np.cos(.31 * times)), axis=1)
    elif kind == 'yaw':
        Lr = Rotation.from_euler('z',(.45 * np.sin(.23 * times))[:, None]).as_matrix()
        Lp = np.stack((.2 * times, np.sin(.4 * times), np.zeros(n)), axis=1)
    else:
        Lr = np.tile(np.eye(3), (n,1,1))
        Lp = np.column_stack((times * .3,np.zeros(n),np.zeros(n)))
    RX = Rotation.from_euler('xyz', [.11, -.06, .18]).as_matrix()
    tx = np.array([.13, -.07, .18])
    sfm_scale = 7.8
    cameraR = Lr @ RX
    cameraP = (Lp + np.einsum('nij,j->ni', Lr, tx)) / sfm_scale + np.array([4., -3., 2.])
    if noise:
        rng = np.random.default_rng(seed)
        cameraR = Rotation.from_rotvec(rng.normal(0,noise,size=(n,3))).as_matrix() @ cameraR
        cameraP = cameraP + rng.normal(0,noise / 2, size=(n,3))
    return MatchedTrajectories(
        timestamps_ns=ns, image_names=tuple(f'{int(t)}.png' for t in ns),
        lidar_R_world=Lr, lidar_t_world_m=Lp,
        camera_R_world_sfm=cameraR, camera_t_world_sfm=cameraP,
        total_registered=n, discarded_outside_glim=0,
    ),RX,tx,sfm_scale


def test_exact_recovery_and_small_residual():
    samples,RX,t,s = make_samples()
    result = fit_full_trajectory(samples)
    assert result.diagnostics['converged']
    assert result.pairs.count > 20
    assert Rotation.from_matrix(result.rotation_camera_lidar @ RX).magnitude() < 1e-8
    np.testing.assert_allclose(result.translation_camera_lidar_m,-RX.T @ t,atol=1e-8)
    assert result.sfm_scale_to_metric == pytest.approx(s, abs=1e-8)
    assert np.max(np.abs(residuals(result.pairs,result.rotation_camera_lidar,
                                   result.translation_camera_lidar_m,result.sfm_scale_to_metric))) < 1e-8


def test_joint_jacobian_and_marginal_covariance():
    samples,_,_,_ = make_samples()
    result = fit_full_trajectory(samples)
    info=analyze_local_uncertainty(result)
    assert info['full_rank'] and info['rank']==7
    sigma=np.array(info['model_based_extrinsic_marginal_covariance_6x6'])
    joint=np.array(info['model_based_joint_covariance_7x7'])
    np.testing.assert_allclose(sigma, joint[:6,:6],atol=1e-7)
    direct=np.linalg.inv(np.array(info['information_matrix_7x7'])[:6,:6])
    assert np.linalg.norm(sigma-direct) > 1e-8, 'scale must be marginalized, not fixed'
    assert np.min(np.linalg.eigvalsh(sigma)) > -1e-12
    assert len(info['model_based_extrinsic_standard_deviations']['rotation_camera_axis_deg']) == 3
    assert len(info['model_based_extrinsic_correlation_matrix_6x6']) == 6


def test_degenerate_yaw_motion_refuses_covariance():
    samples,_,_,_=make_samples('yaw')
    pair=build_motion_pairs(samples,int(samples.timestamps_ns[0]),int(samples.timestamps_ns[-1]),SelectionSettings())
    # known-correct init: due to yaw-only, a subset of extrinsic parameters is not identifiable
    fit=fit_motion_pairs(pair, HandEyeSettings(),initial_rotation=Rotation.from_euler('xyz',[.11,-.06,.18]).as_matrix().T,
                        initial_translation_m=-Rotation.from_euler('xyz',[.11,-.06,.18]).as_matrix().T @ np.array([.13,-.07,.18]),initial_scale=7.8)
    result=analyze_local_uncertainty(fit)
    assert not result['full_rank']
    assert not result['covariance_available']
    assert len(result['weak_modes']) > 0


def test_covar_has_expected_scaling_with_assumed_noise():
    samples,_,_,_=make_samples()
    a=fit_full_trajectory(samples, HandEyeSettings(robust_loss='linear',sigma_relative_rotation_deg=.25,sigma_relative_translation_m=.02))
    b=fit_full_trajectory(samples, HandEyeSettings(robust_loss='linear',sigma_relative_rotation_deg=.5,sigma_relative_translation_m=.04))
    ca=np.array(analyze_local_uncertainty(a)['model_based_joint_covariance_7x7'])
    cb=np.array(analyze_local_uncertainty(b)['model_based_joint_covariance_7x7'])
    np.testing.assert_allclose(cb, ca*4,rtol=1e-4,atol=1e-9)


def test_bootstrap_deterministic_and_serializable(tmp_path):
    samples,_,_,_=make_samples(noise=0.0003)
    fit=fit_full_trajectory(samples)
    b1=block_bootstrap_uncertainty(fit,n_resamples=8,block_duration_s=7,seed=42)
    b2=block_bootstrap_uncertainty(fit,n_resamples=8,block_duration_s=7,seed=42)
    assert b1['available'] and b1['temporal_blocks']>=3
    np.testing.assert_array_equal(b1['empirical_joint_covariance_7x7'],b2['empirical_joint_covariance_7x7'])
    report=report_handeye(fit,bootstrap_resamples=8,bootstrap_block_s=7,bootstrap_seed=42)
    path=write_handeye_report(report,tmp_path/'out.json')
    loaded=json.loads(path.read_text())
    assert loaded['observability_and_model_covariance']['covariance_available']
    assert loaded['bootstrap']['available']
    assert loaded['motion_consistency']['pairs'] == fit.pairs.count
    assert loaded['motion_consistency']['rotation_angle_difference_median_deg'] >= 0
    assert loaded['motion_consistency']['rotation_angle_difference_p95_deg'] >= 0
    assert loaded['handeye']['T_camera_lidar'][3]==[0,0,0,1]


def test_bootstrap_not_enough_blocks():
    samples,_,_,_=make_samples()
    fit=fit_full_trajectory(samples)
    b=block_bootstrap_uncertainty(fit,n_resamples=8,block_duration_s=100)
    assert not b['available']


def test_bootstrap_settings_validation():
    samples,_,_,_=make_samples()
    fit=fit_full_trajectory(samples)
    with pytest.raises(ValueError,match='n_resamples'):
        block_bootstrap_uncertainty(fit,n_resamples=2)
    with pytest.raises(ValueError,match='block_duration_s'):
        block_bootstrap_uncertainty(fit,block_duration_s=-1)


def test_pure_straight_motion_is_rank_deficient():
    samples,R,t,s=make_samples('straight')
    pairs=build_motion_pairs(samples,int(samples.timestamps_ns[0]),int(samples.timestamps_ns[-1]),SelectionSettings())
    fit=fit_motion_pairs(pairs,HandEyeSettings(),initial_rotation=R.T,
                        initial_translation_m=-R.T @ t,initial_scale=s)
    analysis=analyze_local_uncertainty(fit)
    assert analysis['rank'] < 7
    assert not analysis['covariance_available']


def test_marginalization_matches_schur_complement():
    sample,_,_,_=make_samples()
    fit=fit_full_trajectory(sample)
    analysis=analyze_local_uncertainty(fit)
    h=np.array(analysis['information_matrix_7x7'])
    marginalized_precision=h[:6,:6] - h[:6,6,None] @ h[None,6,:6] / h[6,6]
    expected=np.linalg.inv(marginalized_precision)
    measured=np.array(analysis['model_based_extrinsic_marginal_covariance_6x6'])
    np.testing.assert_allclose(expected,measured,rtol=1e-6,atol=1e-6)