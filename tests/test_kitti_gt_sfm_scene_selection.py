"""KITTI GT-fixed SfM scene-selection synthetic tests.

They verify KITTI Camera0->Camera2 geometry, window isolation, COLMAP-ID
association, unchanged poses, track geometry, and scene ranking.
Real COLMAP/KITTI07 image matching is not tested in CI.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (Path(__file__).resolve().parents[1] / 'scripts' / 'experiments' /
          'run_kitti_gt_sfm_scene_selection.py')
spec = importlib.util.spec_from_file_location('gt_sfm_select', SCRIPT)
assert spec is not None and spec.loader is not None
sfm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sfm
spec.loader.exec_module(sfm)


def make_kitti(root: Path, n: int = 31):
    sequence = root / 'sequences' / '07'
    images = sequence / 'image_2'
    images.mkdir(parents=True)
    # Synthetic KITTI rectified intrinsics; camera 2 is 0.5m right of Camera0
    # in the camera0 x axis, so P2[0,3]=-fx*0.5.
    k = np.array([[100., 0., 64.], [0., 100., 48.], [0., 0., 1.]])
    p0 = np.column_stack((k, np.zeros(3)))
    p2 = p0.copy()
    p2[:, 3] = -k @ np.array([.5, 0., 0.])
    calib_lines = []
    for i in range(4):
        p = p2 if i == 2 else p0
        calib_lines.append(f"P{i}: " + " ".join(f"{x:.12g}" for x in p.ravel()))
    calib_lines.append('Tr: ' + ' '.join(map(str, np.column_stack((np.eye(3), np.zeros(3))).ravel())))
    (sequence / 'calib.txt').write_text('\n'.join(calib_lines) + '\n')
    np.savetxt(sequence / 'times.txt', np.arange(n) * .1, fmt='%.9f')
    poses = np.repeat(np.eye(4)[None], n, axis=0)
    poses[:, 0, 3] = np.arange(n) * .1
    poses_file = root / 'poses' / '07.txt'
    poses_file.parent.mkdir()
    np.savetxt(poses_file, poses[:, :3, :4].reshape(n, 12), fmt='%.12g')
    # Only PNG header is necessary at PREPARE stage; no real pixel decoding.
    png = b'\x89PNG\r\n\x1a\n' + b'\x00\x00\x00\x0dIHDR' + struct.pack('>II', 128, 96)
    for i in range(n):
        (images / f'{i:06d}.png').write_bytes(png)
    return sequence, poses_file


def prepare(tmp_path: Path):
    seq, pose = make_kitti(tmp_path)
    out = tmp_path / 'experiment'
    sfm.main(['prepare', '--sequence', str(seq), '--poses', str(pose),
              '--out', str(out), '--sliding', '--window-s', '1',
              '--stride-s', '1', '--min-images', '6', '--max-images', '6'])
    return out


def create_model(base: Path, scene: dict, n_points: int):
    image_records = [(i * 11 + 2, name, 7)
                     for i, name in enumerate(scene['image_names'])]
    sfm._write_seed_model(base, scene, image_records)
    model = base / 'sparse_txt'
    model.mkdir()
    (model / 'cameras.txt').write_text((base / 'model_gt' / 'cameras.txt').read_text())
    expected = sfm._read_images_txt(base / 'model_gt' / 'images.txt')
    points = [np.array([2.5 + .25 * j, 0.15 * (j - 2), 5. + .3 * j])
              for j in range(n_points)]
    observations = {img_id: [] for img_id in expected}
    for point_id, xyz in enumerate(points, start=1):
        for image_id, record in expected.items():
            pose = sfm._pose_matrix(record)
            cam = pose[:3, :3] @ xyz + pose[:3, 3]
            fx, fy, cx, cy = scene['camera_params']
            xy = [fx * cam[0] / cam[2] + cx, fy * cam[1] / cam[2] + cy]
            observations[image_id].append((xy, point_id))
    lines = []
    for img_id, rec in expected.items():
        q = rec['qvec']
        t = rec['tvec']
        line = f"{img_id} " + ' '.join(map(str, np.r_[q, t])) + f" 7 {rec['name']}"
        points2d = ' '.join(
            f'{xy[0]:.17g} {xy[1]:.17g} {pid}' for xy, pid in observations[img_id]
        )
        lines.append(line + '\n' + points2d + '\n')
    (model / 'images.txt').write_text(''.join(lines))
    point_lines = []
    for pid, xyz in enumerate(points, start=1):
        track = ' '.join(
            f'{img_id} {pid-1}' for img_id in expected
        )
        point_lines.append(
            f'{pid} ' + ' '.join(map(str, xyz)) + f' 128 128 128 0.0 {track}'
        )
    (model / 'points3D.txt').write_text('\n'.join(point_lines) + '\n')
    return image_records


def test_camera2_metric_pose_and_preset_windows(tmp_path: Path):
    seq, poses = make_kitti(tmp_path)
    times, twc2, k = sfm.load_kitti_camera_gt(
        seq / 'times.txt', poses, seq / 'calib.txt', 2
    )
    assert len(times) == len(twc2) == 31
    np.testing.assert_allclose(twc2[0, :3, 3], [.5, 0, 0], atol=1e-10)
    np.testing.assert_allclose(twc2[10, :3, 3], [1.5, 0, 0], atol=1e-10)
    np.testing.assert_allclose(k, [[100, 0, 64], [0, 100, 48], [0, 0, 1]])
    with pytest.raises(ValueError, match='poses'):
        sfm.load_kitti_camera_gt(seq / 'times.txt', poses, seq / 'calib.txt', 4)


def test_prepare_window_frame_count_and_equal_budget(tmp_path: Path):
    out = prepare(tmp_path)
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['kind'] == 'kitti_fixed_gt_pose_sfm_scene_candidates'
    assert len(manifest['scenes']) == 3
    for scene in manifest['scenes']:
        assert scene['image_count'] == 6
        base = out / scene['name']
        assert np.load(base / 'gt_T_world_camera.npy').shape == (6, 4, 4)
        assert len((base / 'image_list.txt').read_text().splitlines()) == 6
    with pytest.raises(FileExistsError):
        sfm.main(['prepare', '--sequence', str(tmp_path / 'sequences' / '07'),
                  '--poses', str(tmp_path / 'poses' / '07.txt'),
                  '--out', str(out), '--sliding'])


def test_colmap_image_ids_are_not_assumed_to_be_sequential(tmp_path: Path):
    out = prepare(tmp_path)
    name = 'slide_000'
    base = out / name
    scene = json.loads((base / 'scene.json').read_text())
    db = base / 'database.db'
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE images (image_id INTEGER, name TEXT, camera_id INTEGER)')
        con.execute('CREATE TABLE cameras (camera_id INTEGER, model INTEGER, width INTEGER, height INTEGER, params BLOB)')
        for i, n in enumerate(scene['image_names']):
            con.execute('INSERT INTO images VALUES (?, ?, ?)', (i * 13 + 7, n, 5))
        con.execute('INSERT INTO cameras VALUES (?, ?, ?, ?, ?)',
                    (5, 1, 128, 96, np.asarray(scene['camera_params'], dtype='<f8').tobytes()))
    records = sfm._db_images(db, scene['image_names'], 128, 96, scene['camera_params'])
    assert [row[0] for row in records] == [7, 20, 33, 46, 59, 72]
    sfm._write_seed_model(base, scene, records)
    registered = sfm._read_images_txt(base / 'model_gt' / 'images.txt')
    assert set(registered) == {7, 20, 33, 46, 59, 72}
    assert all(len(row['xy']) == 0 for row in registered.values())


def test_reconstructed_scene_ranking_and_gt_pose_guard(tmp_path: Path):
    out = prepare(tmp_path)
    _, scenes = sfm._scene_entries(out)
    for k, scene in enumerate(scenes):
        base = out / scene['name']
        create_model(base, scene, [9, 4, 2][k])
    assert sfm.main(['evaluate', '--out', str(out), '--min-qualified-points', '1',
                     '--min-track', '3', '--min-parallax-deg', '.1']) == 0
    report = json.loads((out / 'scene_selection.json').read_text())
    assert len(report['all_candidates']) == 3
    rows = {x['name']: x for x in report['all_candidates']}
    assert [rows[f'slide_{i:03d}']['qualified_points'] for i in range(3)] == [9, 4, 2]
    assert rows['slide_000']['median_reproj_error_px'] < 1e-8
    assert rows['slide_000']['median_parallax_deg'] > .1
    assert report['scenes_selected'][0]['name'] == 'slide_000'
    with (out / 'scene_scores.csv').open() as fh:
        assert len(list(csv.DictReader(fh))) == 3
    # A tiny deliberately corrupted extrinsic must invalidate a "GT" reconstruction.
    model = out / 'slide_000' / 'sparse_txt' / 'images.txt'
    text = model.read_text()
    parts = text.splitlines()
    fields = parts[0].split()
    fields[5] = str(float(fields[5]) + .01)
    parts[0] = ' '.join(fields)
    model.write_text('\n'.join(parts) + '\n')
    with pytest.raises(ValueError, match='changed fixed pose'):
        sfm.main(['evaluate', '--out', str(out), '--min-qualified-points', '1'])


def test_invalid_track_detection(tmp_path: Path):
    out = prepare(tmp_path)
    scene = sfm._scene_entries(out)[1][0]
    base = out / scene['name']
    create_model(base, scene, 3)
    model = base / 'sparse_txt' / 'points3D.txt'
    lines = model.read_text().splitlines()
    parts = lines[0].split()
    parts[-1] = '9999999'  # Unknown POINT2D_IDX
    lines[0] = ' '.join(parts)
    model.write_text('\n'.join(lines) + '\n')
    eva = sfm.build_parser().parse_args(['evaluate', '--out', str(out)])
    with pytest.raises(ValueError, match='POINT2D_IDX'):
        sfm._score_scene(base, scene, eva)
