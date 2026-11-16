"""Synthetic tests for KITTI GT-pose reference; no real KITTI accuracy claims."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / 'scripts' / 'experiments'
sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location(
    'spade_oracle_accumulation', SCRIPT_DIR / 'run_oracle_accumulation.py'
)
assert spec is not None and spec.loader is not None
oracle = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oracle
spec.loader.exec_module(oracle)


def _write_dataset(root: Path, n: int = 8, glim_offset: float = 0.0,
                   dense: bool = False):
    times = np.arange(n, dtype=float) * .1
    times_file = root / 'times.txt'
    np.savetxt(times_file, times, fmt='%.9f')
    tr = np.eye(4)
    tr[:3, 3] = [.25, .1, -.05]
    calib = root / 'calib.txt'
    p0 = np.column_stack((np.eye(3), np.zeros(3)))
    calib.write_text('P0: ' + ' '.join(map(str, p0.flat)) + '\n' +
                     'Tr: ' + ' '.join(map(str, tr[:3].flat)) + '\n')
    tc = np.repeat(np.eye(4)[None], n, axis=0)
    tc[:, 0, 3] = np.arange(n) * .6
    poses = root / 'poses.txt'
    np.savetxt(poses, tc[:, :3, :4].reshape(n, 12), fmt='%.9f')
    gt = tc @ tr
    odom = np.zeros((n, 8))
    odom[:, 0] = times + 1
    odom[:, 1:4] = gt[:, :3, 3]
    odom[:, 1] += glim_offset * np.arange(n)
    odom[:, 7] = 1.0
    odom_file = root / 'odom.txt'
    np.savetxt(odom_file, odom, fmt='%.9f')
    scan_dir = root / 'velodyne'
    scan_dir.mkdir()
    world = np.array([[5., 0., 0.], [6., 2., 1.], [9., -2., -.5],
                      [12., 3., 2.], [15., -4., 0.], [18., 0., 1.]])
    if dense:
        # Three orthogonal planar patches; enough local geometry for the REAL
        # normal-estimation, pairwise PGO and Joint solvers (not mocked).
        u, v = np.meshgrid(np.linspace(5, 12, 17), np.linspace(-3, 3, 17))
        a, b = np.meshgrid(np.linspace(-3, 3, 17), np.linspace(-1, 3, 17))
        world = np.vstack((
            np.column_stack((u.ravel(), v.ravel(), np.full(u.size, -1.0))),
            np.column_stack((np.full(a.size, 6.0), a.ravel(), b.ravel())),
            np.column_stack((u.ravel(), np.full(u.size, -3.0), b.ravel())),
        ))
    for i in range(n):
        local = world - gt[i, :3, 3]
        np.column_stack((local, np.ones(len(local)))).astype('<f4').tofile(
            scan_dir / f'{i:06d}.bin'
        )
    return times_file, odom_file, poses, calib, scan_dir


def test_poses_and_composition(tmp_path: Path):
    _, _, poses, calib, _ = _write_dataset(tmp_path)
    twl = oracle.load_gt_poses(poses, calib, frame_count=8)
    assert twl[0, 0, 3] == pytest.approx(.25)
    assert twl[0, 1, 3] == pytest.approx(.1)
    rel = oracle.relative_poses(twl, np.array([1, 3, 5]), target=3)
    np.testing.assert_allclose(rel[1], np.eye(4), atol=1e-10)
    assert rel[0, 0, 3] == pytest.approx(-1.2)
    assert rel[2, 0, 3] == pytest.approx(1.2)
    assert oracle.pose_errors(rel, rel, 1)['pose_trans_max_m'] == pytest.approx(0.)
    assert oracle.pose_errors(rel, rel, 1)['pose_rot_max_deg'] == pytest.approx(0.)


def test_voxel_common_origin_and_consistency():
    roi = np.array([0., 5., -2., 2., -1., 2.])
    ref = oracle.voxel_keys(
        np.array([[.2, .1, 0.], [1.1, .1, 0.], [3.1, .1, 0.]]), roi, 1.
    )
    subset = oracle.voxel_keys(np.array([[1.2, .3, 0.], [3.2, .3, 0.]]), roi, 1.)
    metrics = oracle.occupancy_metrics(subset, ref)
    assert metrics['common_voxels'] == 2
    assert metrics['reference_recall'] == pytest.approx(2 / 3)
    assert metrics['reference_precision'] == pytest.approx(1.0)
    assert len(oracle.voxel_keys(np.array([[6., 0., 0.]]), roi, 1.)) == 0


def test_nonzero_rotation_and_pose_displacement():
    gt = np.repeat(np.eye(4)[None], 3, axis=0)
    est = gt.copy()
    est[0, :3, :3] = Rotation.from_euler('z', 5, degrees=True).as_matrix()
    est[2, 0, 3] = .2
    e = oracle.pose_errors(est, gt, anchor_local=1)
    assert e['pose_rot_max_deg'] == pytest.approx(5.)
    assert e['pose_trans_max_m'] == pytest.approx(.2)
    roi = np.array([-3., 3., -3., 3., -3., 3.])
    p = np.array([[0., 0., 0.], [1., 0., 0.]])
    d = oracle.point_displacement(p + np.array([.1, 0, 0]), p, roi)
    assert d['paired_rmse_m'] == pytest.approx(.1)


def test_cli_synthetic_oracle_glim_and_budget(tmp_path: Path):
    times, odom, poses, calib, velodyne = _write_dataset(tmp_path)
    out = tmp_path / 'results'
    argv = ['--times', str(times), '--odom', str(odom), '--poses', str(poses),
            '--calib', str(calib), '--velodyne', str(velodyne), '--out', str(out),
            '--start-s', '0', '--stop-s', '0.7', '--target-frame', '3',
            '--half-frames', '2', '--radii-m', '1.3', '--match-scan-budget',
            '--roi', '0', '22', '-6', '6', '-3', '4', '--voxel-m', '0.5']
    assert oracle.main(argv) == 0
    with (out / 'summary.csv').open() as fh:
        summary = list(csv.DictReader(fh))
    manifest = json.loads((out / 'manifest.json').read_text())
    assert len(summary) == 6
    assert manifest['reference']['not_surface_ground_truth'] is True
    for row in summary:
        assert 0 < float(row['reference_recall']) <= 1
        if row['stage'] == 'oracle':
            assert float(row['reference_precision']) == pytest.approx(1.)
        else:
            assert float(row['paired_rmse_m']) < 1.e-5
            assert float(row['pose_trans_mean_m']) < 1.e-10
    plans = {p['method']: p for p in manifest['comparisons']}
    assert len(plans['fixed_frame']['scan_frames']) == len(
        plans['distance_budget']['scan_frames']
    )


def test_glim_pose_perturbation_exposed(tmp_path: Path):
    times, odom, poses, calib, velodyne = _write_dataset(tmp_path, glim_offset=.02)
    out = tmp_path / 'error_results'
    oracle.main(['--times', str(times), '--odom', str(odom), '--poses', str(poses),
                 '--calib', str(calib), '--velodyne', str(velodyne), '--out', str(out),
                 '--start-s', '0', '--stop-s', '0.7', '--target-frame', '3',
                 '--half-frames', '2', '--radii-m', '1.3',
                 '--roi', '0', '22', '-6', '6', '-3', '4'])
    with (out / 'summary.csv').open() as fh:
        rows = list(csv.DictReader(fh))
    glim_rows = [r for r in rows if r['stage'] == 'glim']
    assert all(float(r['pose_trans_mean_m']) > 0 for r in glim_rows)
    assert all(float(r['paired_rmse_m']) > 0 for r in glim_rows)



def test_cli_pgo_and_both_use_independent_glim_initialization(tmp_path: Path):
    """PGO is a real solver call, and both stages share the same original GLIM cloud."""
    times, odom, poses, calib, velodyne = _write_dataset(
        tmp_path, glim_offset=.015, dense=True
    )
    out = tmp_path / 'pgo_results'
    args = [
        '--times', str(times), '--odom', str(odom), '--poses', str(poses),
        '--calib', str(calib), '--velodyne', str(velodyne), '--out', str(out),
        '--start-s', '0', '--stop-s', '0.7', '--target-frame', '3',
        '--half-frames', '1', '--radii-m', '.7',
        '--roi', '0', '22', '-6', '6', '-3', '4', '--voxel-m', '.5',
        '--refiner', 'both', '--refine-iterations', '2',
        '--refine-voxel-m', '.2', '--refine-max-samples', '1200',
        '--pgo-iterations', '2', '--pgo-pair-max-nfev', '15',
    ]
    assert oracle.main(args) == 0
    rows = list(csv.DictReader((out / 'summary.csv').open()))
    plans = json.loads((out / 'manifest.json').read_text())['comparisons']
    assert len(rows) == 4 * len(plans)  # oracle / glim / joint / pgo
    for plan in plans:
        name = plan['name']
        group = {r['stage']: r for r in rows if r['experiment'] == name}
        assert set(group) == {'oracle', 'glim', 'joint', 'pgo'}
        anchor = plan['scan_frames'].index(3)
        for method in ('joint', 'pgo'):
            transforms = np.load(out / f'{name}_{method}_corrections.npy')
            assert transforms.shape == (len(plan['scan_frames']), 4, 4)
            np.testing.assert_allclose(transforms[anchor], np.eye(4), atol=1e-8)
            diagnostics = json.loads(
                (out / f'{name}_{method}_diagnostics.json').read_text()
            )
            assert diagnostics['initialization'] == 'independent_original_GLIM_accumulation'
            assert diagnostics['iterations']
            assert np.isfinite(float(group[method]['pose_trans_mean_m']))
            assert np.isfinite(float(group[method]['paired_rmse_m']))
        diag_pgo = json.loads((out / f'{name}_pgo_diagnostics.json').read_text())
        assert diag_pgo['iterations'][0]['pair_edges'] >= 1
