"""Conventions: Camera<-LiDAR Hand-Eye, residual invariance, KITTI evaluation."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spade_calib.cli import main
from spade_calib.evaluation.kitti import (
    evaluate_camera_lidar, evaluate_handeye_json, kitti_gt_camera_lidar,
    parse_kitti_odom_calib,
)
from spade_calib.handeye.residuals import residuals
from spade_calib.selection.trajectory import MotionPairs


def test_camera_lidar_residual_matches_inverse_equation_norm():
    rng = np.random.default_rng(432)
    n = 20
    ra = Rotation.from_rotvec(rng.normal(0, .12, (n, 3))).as_matrix()
    rb = Rotation.from_rotvec(rng.normal(0, .12, (n, 3))).as_matrix()
    ta = rng.normal(size=(n, 3))
    tb = rng.normal(size=(n, 3))
    pairs = MotionPairs(ra, ta, rb, tb, np.zeros((n, 2), dtype=np.int64),
                        np.arange(n, dtype=np.int64))
    R_CL = Rotation.from_euler('xyz', [.12, -.09, .18]).as_matrix()
    t_CL = np.array([.26, -.07, -.13])
    s = 8.3

    new_res = residuals(pairs, R_CL, t_CL, s)

    # Previous inverse convention A X_old = X_old B(s), X_old = X_CL^-1.
    R_LC = R_CL.T
    t_LC = -R_LC @ t_CL
    old_left = ra @ R_LC
    old_right = R_LC @ rb
    old_rot = Rotation.from_matrix(np.swapaxes(old_left, 1, 2) @ old_right).as_rotvec()
    old_t = (np.einsum('nij,j->ni', ra - np.eye(3), t_LC) + ta
             - s * np.einsum('ij,nj->ni', R_LC, tb))
    # Both residual norms must agree pairwise under an orthogonal frame change.
    np.testing.assert_allclose(np.linalg.norm(new_res[:, :3], axis=1),
                               np.linalg.norm(old_rot, axis=1), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(new_res[:, 3:], -(R_CL @ old_t.T).T,
                               rtol=1e-10, atol=1e-10)


def test_kitti_camera2_calibration_and_axis_evaluation(tmp_path):
    calib = tmp_path / 'calib.txt'
    # A camera 2 optical centre at [0.1, 0, 0] in camera 0 coordinates.
    P0 = np.array([[500, 0, 320, 0],
                   [0, 500, 200, 0],
                   [0, 0, 1, 0]], dtype=float)
    P2 = P0.copy()
    P2[:, 3] = -(P2[:, :3] @ np.array([.1, 0., 0.]))
    Tr = np.eye(4)[:3]
    Tr[:, 3] = [0.2, -.05, -.3]
    def line(name, x):
        return name + ': ' + ' '.join(str(float(v)) for v in x.ravel()) + '\n'
    calib.write_text(line('P0', P0) + line('P2', P2) + line('Tr', Tr))
    gt = kitti_gt_camera_lidar(calib, camera_index=2)
    np.testing.assert_allclose(gt[:3, 3], [.1, -.05, -.3])
    np.testing.assert_allclose(kitti_gt_camera_lidar(calib, 0)[:3, 3], [.2, -.05, -.3])

    est = gt.copy()
    est[:3, :3] = Rotation.from_euler('x', 1.0, degrees=True).as_matrix() @ gt[:3, :3]
    est[:3, 3] += [.04, -.03, .00]
    evaluation = evaluate_camera_lidar(est, gt)
    assert evaluation['rotation_error_total_deg'] == pytest.approx(1)
    np.testing.assert_allclose(evaluation['rotation_error_vector_camera_deg'], [1, 0, 0], atol=1e-10)
    np.testing.assert_allclose(evaluation['translation_error_vector_camera_m'], [.04, -.03, 0], atol=1e-12)
    assert evaluation['translation_error_total_m'] == pytest.approx(.05)

    result = tmp_path / 'handeye.json'
    result.write_text(json.dumps({'handeye': {'T_camera_lidar': est.tolist()}}))
    gt_eval = evaluate_handeye_json(result, calib, camera_index=2)
    assert gt_eval['gt_used_for_fitting'] is False
    assert gt_eval['rotation_error_total_deg'] == pytest.approx(1)

    output = tmp_path / 'eval.json'
    assert main(['evaluate-kitti-handeye', '--result', str(result),
                 '--calib', str(calib), '--camera-index', '2',
                 '--output', str(output)]) == 0
    assert json.loads(output.read_text())['translation_error_total_m'] == pytest.approx(.05)
    assert main(['evaluate-kitti-handeye', '--result', str(result),
                 '--calib', str(calib), '--camera-index', '1']) == 2


def test_kitti_calib_rejects_invalid_or_missing_data(tmp_path):
    bad = tmp_path / 'bad.txt'
    bad.write_text('P0: 1 2 3\n')
    with pytest.raises(ValueError, match='12 numbers'):
        parse_kitti_odom_calib(bad)
    bad.write_text('P0: ' + ' '.join(['1'] * 12) + '\n')
    with pytest.raises(ValueError, match='P0 and Tr'):
        parse_kitti_odom_calib(bad)
