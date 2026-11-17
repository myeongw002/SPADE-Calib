"""End-to-end CLI wiring test with controlled synthetic inputs (not KITTI07 evidence)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from spade_calib.cli import main
from spade_calib.selection.trajectory import MotionPairs


def _synthetic_files(tmp_path, *, saved_pair_count: int | None = None):
    rng = np.random.default_rng(91)
    n = 20
    Ra = Rotation.from_rotvec(rng.normal(0., .13, (n, 3))).as_matrix()
    ta = rng.normal(0., .3, (n, 3))
    scale = 7.2
    pairs = MotionPairs(Ra, ta, Ra.copy(), ta / scale,
                        np.column_stack([np.arange(n), np.arange(n) + 1]),
                        np.arange(n, dtype=np.int64))
    initial = np.eye(4)
    initial[:3, :3] = Rotation.from_rotvec([.01, -.02, .012]).as_matrix()
    initial[:3, 3] = [.08, -.16, .27]
    handeye = tmp_path / 'handeye.json'
    handeye.write_text(json.dumps({
        'handeye': {
            'convention': 'T_camera_lidar (multiply lidar point to camera frame)',
            'T_camera_lidar': initial.tolist(),
            'sfm_scale_to_metric': scale,
            'number_of_motion_pairs': n if saved_pair_count is None else saved_pair_count,
            'settings': {'pair_baseline_s': 2.0, 'pair_stride_s': 1.0,
                         'pair_tolerance_s': .12}
        }
    }), encoding='utf-8')
    match = tmp_path / 'matches.json'
    match.write_text(json.dumps({
        'schema_version': 1, 'convention': 'T_camera_lidar',
        'matches': [{
            'id': 'synthetic_g',
            'lidar': {'frame': 'lidar_target', 'normal': [0, 0, 1], 'offset_m': -.6},
            'camera': {'frame': 'camera', 'normal': [0, 0, 1], 'offset_m': -.6},
        }]
    }), encoding='utf-8')
    K = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
    P = np.c_[K, np.zeros(3)]
    Tr = np.eye(4)[:3]
    calib = tmp_path / 'calib.txt'
    def line(name, M):
        return name + ': ' + ' '.join(map(str, M.reshape(-1))) + '\n'
    calib.write_text(line('P0', P) + line('P2', P) + line('Tr', Tr))
    return handeye, match, calib, pairs


def test_refine_ground_cli_produces_both_ablation_and_gt_evaluation(tmp_path, monkeypatch):
    handeye, match, calib, pairs = _synthetic_files(tmp_path)
    import spade_calib.selection.trajectory as trajectory
    import spade_calib.calibration.cli as ground_cli
    monkeypatch.setattr(trajectory, 'load_trajectory_data', lambda *a, **kw:
                        SimpleNamespace(timestamps_ns=np.arange(200, dtype=np.int64)))
    monkeypatch.setattr(ground_cli, 'build_motion_pairs', lambda *a, **kw: pairs)
    out = tmp_path / 'refined.json'
    code = main(['refine-ground', '--handeye', str(handeye),
                 '--ground-matches', str(match), '--glim', 'synthetic_glim.txt',
                 '--sfm', 'synthetic_sfm.txt', '--calib', str(calib),
                 '--robust-loss', 'linear', '--output', str(out)])
    assert code == 0
    doc = json.loads(out.read_text(encoding='utf-8'))
    assert doc['gt_used_for_fitting'] is False
    assert doc['kitti_evaluation_only']['gt_used_for_fitting'] is False
    assert doc['input_ground_correspondence_count'] == 1
    assert doc['fixed_rotation']['optimizer']['success']
    assert doc['optimize_rotation']['optimizer']['success']
    assert doc['fixed_rotation']['sfm_scale_to_metric_fixed'] == 7.2
    assert doc['optimize_rotation']['sfm_scale_to_metric_fixed'] == 7.2
    np.testing.assert_allclose(
        np.array(doc['baseline']['T_camera_lidar'])[:3, :3],
        np.array(doc['fixed_rotation']['T_camera_lidar'])[:3, :3],
        atol=1e-14,
    )
    evaluated = doc['kitti_evaluation_only']
    assert evaluated['optimize_rotation']['translation_error_total_m'] < (
        evaluated['baseline']['translation_error_total_m']
    )


def test_refine_ground_cli_rejects_unmatched_motion_pair_count(tmp_path, monkeypatch):
    handeye, match, _, pairs = _synthetic_files(tmp_path, saved_pair_count=99)
    import spade_calib.selection.trajectory as trajectory
    import spade_calib.calibration.cli as ground_cli
    monkeypatch.setattr(trajectory, 'load_trajectory_data', lambda *a, **kw:
                        SimpleNamespace(timestamps_ns=np.arange(200, dtype=np.int64)))
    monkeypatch.setattr(ground_cli, 'build_motion_pairs', lambda *a, **kw: pairs)
    out = tmp_path / 'not_created.json'
    assert main(['refine-ground', '--handeye', str(handeye),
                 '--ground-matches', str(match), '--glim', 'synthetic_glim.txt',
                 '--sfm', 'synthetic_sfm.txt', '--output', str(out)]) == 2
    assert not out.exists()
