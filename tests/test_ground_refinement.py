"""Synthetic tests for 3-DoF fixed-R versus 6-DoF Hand-Eye + Ground.

Metric plane measurements are generated independently of KITTI calibration GT;
these tests validate geometry/code, not real-world performance.
"""
from __future__ import annotations

import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spade_calib.calibration import (
    GroundMatch, Plane, GroundRefinementSettings, compare_modes, refine_with_ground,
    load_ground_matches, transform_plane_to_camera,
)
from spade_calib.selection.trajectory import MotionPairs
from spade_calib.evaluation.kitti import evaluate_camera_lidar


def synthetic(*, rich: bool) -> tuple[np.ndarray, MotionPairs, GroundMatch, float]:
    rng = np.random.default_rng(20261117)
    nominal = np.array([[0., -1., 0.],
                        [0., 0., -1.],
                        [1., 0., 0.]])
    R = Rotation.from_rotvec([.015, -.01, .005]).as_matrix() @ nominal
    t = np.array([.06, -.07, -.32])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    n = 34
    if rich:
        rotvec = rng.normal(0., .12, (n, 3))
    else:
        rotvec = np.zeros((n, 3))
        rotvec[:, 2] = np.linspace(-.3, .3, n)
    Ra = Rotation.from_rotvec(rotvec).as_matrix()
    ta = rng.normal(0., .4, (n, 3))
    if not rich:
        ta[:, 2] = 0.
    Rb = np.einsum('ij,njk,kl->nil', R, Ra, R.T)
    scale = 8.3
    tb = ((R @ ta.T).T -
          np.einsum('nij,j->ni', Rb - np.eye(3), t)) / scale
    pairs = MotionPairs(Ra, ta, Rb, tb,
                        np.column_stack([np.arange(n), np.arange(n) + 1]),
                        np.arange(n, dtype=np.int64) * 1000000000)
    lidar = Plane(np.array([0., 0., 1.]), -.45)
    camera = transform_plane_to_camera(lidar, R, t)
    # Opposite signs describe exactly the same geometric plane.
    sign_flipped = Plane(-camera.normal, -camera.offset_m)
    match = GroundMatch('ground_001', lidar, sign_flipped)
    return T, pairs, match, scale


def test_plane_transform_preserves_incidence() -> None:
    T, _, match, _ = synthetic(rich=True)
    p_l = np.array([1.3, -.6, .45])
    assert abs(match.lidar.normal @ p_l + match.lidar.offset_m) < 1e-12
    p_c = T[:3, :3] @ p_l + T[:3, 3]
    predicted = transform_plane_to_camera(match.lidar, T[:3, :3], T[:3, 3])
    assert abs(predicted.normal @ p_c + predicted.offset_m) < 1e-12
    np.testing.assert_allclose(predicted.normal, -match.camera.normal)


def test_translation_only_recovers_normal_axis_and_keeps_rotation() -> None:
    gt, pairs, match, scale = synthetic(rich=False)
    init = gt.copy()
    nc = gt[:3, :3] @ match.lidar.normal
    init[:3, 3] += .4 * nc
    result = refine_with_ground(init, scale, pairs, [match], mode='fixed_rotation',
                                settings=GroundRefinementSettings(robust_loss='linear'))
    T = np.array(result['T_camera_lidar'])
    np.testing.assert_array_equal(T[:3, :3], gt[:3, :3])
    assert np.linalg.norm(T[:3, 3] - gt[:3, 3]) < 2e-5
    assert result['jacobian_ground_only']['rank'] == 1
    assert result['optimizer']['success']
    assert not result['optimizer']['any_bound_hit']


def test_joint_six_dof_recovers_rotation_and_translation() -> None:
    gt, pairs, match, scale = synthetic(rich=True)
    init = gt.copy()
    init[:3, :3] = Rotation.from_rotvec([.035, -.025, .02]).as_matrix() @ gt[:3, :3]
    init[:3, 3] += [.12, .18, -.11]
    compared = compare_modes(
        init, scale, pairs, [match], settings=GroundRefinementSettings(robust_loss='linear'),
    )
    full = compared['optimize_rotation']
    T = np.array(full['T_camera_lidar'])
    evaluated = evaluate_camera_lidar(T, gt)
    assert evaluated['rotation_error_total_deg'] < 0.02
    assert evaluated['translation_error_total_m'] < 2e-3
    assert full['optimizer']['success']
    assert full['jacobian_joint']['rank'] == 6
    assert full['jacobian_ground_only']['rank'] <= 3
    np.testing.assert_array_equal(
        np.array(compared['fixed_rotation']['T_camera_lidar'])[:3, :3], init[:3, :3],
    )
    assert compared['fixed_rotation']['sfm_scale_to_metric_fixed'] == scale
    assert full['sfm_scale_to_metric_fixed'] == scale


def test_invalid_ground_geometry_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match='nonzero normal'):
        Plane([0., 0., 0.], 0.)
    bad = {'schema_version': 1, 'convention': 'T_camera_lidar',
           'matches': [{'id': 'g', 'lidar': {'frame': 'world', 'normal': [0,0,1],
                                          'offset_m': 0},
                        'camera': {'frame': 'camera', 'normal': [0,1,0],
                                   'offset_m': 0}}]}
    src = tmp_path / 'ground.json'
    src.write_text(json.dumps(bad), encoding='utf-8')
    with pytest.raises(ValueError, match='lidar_target'):
        load_ground_matches(src)
    bad['matches'][0]['lidar']['frame'] = 'lidar_target'
    src.write_text(json.dumps(bad), encoding='utf-8')
    assert len(load_ground_matches(src)) == 1


def test_unmatched_normals_and_duplicate_ids_rejected() -> None:
    gt, pairs, match, scale = synthetic(rich=True)
    wrong = GroundMatch('different', match.lidar,
                        Plane([0,0,1], -.5))
    with pytest.raises(ValueError, match='initial ground normals differ'):
        refine_with_ground(gt, scale, pairs, [wrong], mode='fixed_rotation')
    with pytest.raises(ValueError, match='Duplicate'):
        refine_with_ground(gt, scale, pairs, [match, match], mode='fixed_rotation')
    with pytest.raises(ValueError, match='handeye_weight'):
        GroundRefinementSettings(handeye_weight=0.)


def test_modes_differ_only_in_rotation_freedom() -> None:
    gt, pairs, match, scale = synthetic(rich=False)
    init = gt.copy()
    init[:3, 3] += [0., .3, 0.]
    compared = compare_modes(init, scale, pairs, [match])
    assert compared['fixed_rotation']['optimizer']['plane_count'] == 1
    assert compared['optimize_rotation']['optimizer']['plane_count'] == 1
    assert compared['fixed_rotation']['optimizer']['motion_pair_count'] == pairs.count
    assert compared['optimize_rotation']['optimizer']['motion_pair_count'] == pairs.count
    assert compared['fixed_rotation']['jacobian_ground_only']['rank'] == 1
