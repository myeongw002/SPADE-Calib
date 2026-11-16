#!/usr/bin/env python3
"""KITTI07 fixed-GT-pose COLMAP SfM and geometry-aware scene selection.

Stages:
  prepare: define identical candidate windows and known GT camera poses.
  reconstruct / evaluate: independent local-window COLMAP and ranking.
  reconstruct-global / evaluate-global: full-sequence COLMAP once, score
    tracked points in those SAME windows (global 3D coordinates may use
    observations OUTSIDE each window).
  compare: join rankings/scores using identical candidate IDs.

Sparse points are triangulated estimates, NOT ground-truth 3D geometry.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import sqlite3
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from spade_calib.evaluation.kitti import parse_kitti_odom_calib

LOG = logging.getLogger('kitti_gt_sfm_scenes')
PRESET_WINDOWS = {'005': (37.5, 52.5), '012': (90.0, 105.0)}
PNG_HEADER = b'\x89PNG\r\n\x1a\n'


def read_png_size(path: Path) -> tuple[int, int]:
    with path.open('rb') as fh:
        header = fh.read(24)
    if len(header) != 24 or header[:8] != PNG_HEADER or header[12:16] != b'IHDR':
        raise ValueError(f'Not a supported PNG: {path}')
    width, height = struct.unpack('>II', header[16:24])
    if not (0 < width < 100000 and 0 < height < 100000):
        raise ValueError(f'Invalid PNG size in {path}')
    return width, height


def load_kitti_camera_gt(times_path: Path, poses_path: Path, calib_path: Path,
                         camera_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frame times, T_world_camera_k, rectified intrinsic K_k.

    KITTI Odometry poses are rectified camera-0-to-world. The rectified Pk
    projection center is Ck=-inv(Kk)Pk[:,3], expressed in camera-0 axes.
    Hence T_world_Ck = T_world_C0 @ Trans(Ck-C0).
    """
    times = np.loadtxt(times_path, dtype=np.float64, ndmin=1)
    data = np.loadtxt(poses_path, dtype=np.float64, ndmin=2)
    if (len(times) < 5 or not np.isfinite(times).all() or
            not np.all(np.diff(times) > 0)):
        raise ValueError('times.txt must be finite and strictly increasing (>=5 rows)')
    if data.shape != (len(times), 12) or not np.isfinite(data).all():
        raise ValueError('KITTI GT poses must contain one finite 3x4 matrix per timestamp')
    gt = np.repeat(np.eye(4)[None], len(times), axis=0)
    gt[:, :3, :4] = data.reshape(-1, 3, 4)
    r = gt[:, :3, :3]
    if (not np.allclose(r @ np.swapaxes(r, 1, 2), np.eye(3), atol=2e-3) or
            not np.allclose(np.linalg.det(r), 1., atol=2e-3)):
        raise ValueError('KITTI GT contains nonrigid camera poses')
    # Camera-only KITTI color releases may omit Tr (LiDAR extrinsic).
    # GT camera-pose SfM needs only the rectified projection matrices.
    calib = parse_kitti_odom_calib(calib_path, require_tr=False)
    p0 = calib['P0']
    key = f'P{camera_index}'
    if key not in calib:
        raise ValueError(f'No {key} in KITTI calibration')
    pk = calib[key]
    k0, kk = p0[:, :3], pk[:, :3]
    if (abs(np.linalg.det(k0)) < 1e-10 or abs(np.linalg.det(kk)) < 1e-10 or
            not np.allclose(kk[2], [0, 0, 1], atol=1e-5)):
        raise ValueError('Invalid KITTI rectified camera intrinsics')
    c0 = -np.linalg.solve(k0, p0[:, 3])
    ck = -np.linalg.solve(kk, pk[:, 3])
    tc0ck = np.eye(4)
    tc0ck[:3, 3] = ck - c0
    return times, gt @ tc0ck, kk


def window_candidates(times: np.ndarray, args: argparse.Namespace) -> list[tuple[str, float, float, list[int]]]:
    if args.sliding:
        lo = float(times[0]) if args.start_s is None else args.start_s
        hi = float(times[-1]) if args.stop_s is None else args.stop_s
        if (not np.isfinite([lo, hi, args.window_s, args.stride_s]).all() or
                args.window_s <= 0 or args.stride_s <= 0 or hi - lo < args.window_s):
            raise ValueError('Invalid sliding-window bounds/step')
        specs = [(f'slide_{i:03d}', float(start), float(start + args.window_s))
                 for i, start in enumerate(np.arange(lo, hi - args.window_s + 1e-8,
                                                       args.stride_s))]
    else:
        specs = [(key, *PRESET_WINDOWS[key]) for key in args.windows]
    if not specs or len({s[0] for s in specs}) != len(specs):
        raise ValueError('Empty or duplicate scene windows')
    results = []
    for key, lo, hi in specs:
        frames = np.flatnonzero((times >= lo) & (times <= hi)).tolist()
        frames = frames[::args.frame_step]
        if args.max_images is not None and len(frames) > args.max_images:
            idx = np.rint(np.linspace(0, len(frames) - 1, args.max_images)).astype(int)
            frames = [frames[i] for i in idx]
        if len(frames) < args.min_images:
            LOG.warning('Skipping scene %s: only %d images', key, len(frames))
            continue
        results.append((key, float(lo), float(hi), frames))
    if not results:
        raise ValueError('No eligible candidate window with sufficient images')
    return results


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def prepare(args: argparse.Namespace) -> int:
    if args.frame_step < 1 or args.min_images < 3 or (
            args.max_images is not None and args.max_images < args.min_images):
        raise ValueError('Invalid frame-step or image-count limits')
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f'Output directory not empty: {args.out}')
    seq = args.sequence.resolve(strict=True)
    images = (args.images if args.images else seq / f'image_{args.camera}').resolve(strict=True)
    times, twc, intrinsic = load_kitti_camera_gt(
        seq / 'times.txt', args.poses, seq / 'calib.txt', args.camera)
    params = [float(intrinsic[0, 0]), float(intrinsic[1, 1]),
              float(intrinsic[0, 2]), float(intrinsic[1, 2])]
    work = []
    for name, lo, hi, frames in window_candidates(times, args):
        pngs = [images / f'{f:06d}.png' for f in frames]
        sizes = {read_png_size(path) for path in pngs}
        if len(sizes) != 1:
            raise ValueError(f'Mixed image sizes in scene {name}: {sizes}')
        width, height = sizes.pop()
        path = args.out / name
        path.mkdir(parents=True, exist_ok=False)
        np.save(path / 'gt_T_world_camera.npy', twc[frames])
        (path / 'image_list.txt').write_text(
            '\n'.join(p.name for p in pngs) + '\n', encoding='utf-8')
        window = {
            'name': name, 'start_s': lo, 'stop_s': hi,
            'frames': frames, 'image_names': [p.name for p in pngs],
            'image_count': len(frames), 'image_width': width, 'image_height': height,
            'camera_model': 'PINHOLE', 'camera_params': params,
            'pose_convention': f'T_world_rectified_camera_{args.camera}',
        }
        _write_json(path / 'scene.json', window)
        work.append(window)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        'schema_version': 1, 'kind': 'kitti_fixed_gt_pose_sfm_scene_candidates',
        'sequence': str(seq), 'poses_path': str(args.poses.resolve(strict=True)),
        'image_path': str(images), 'camera_index': args.camera,
        'intrinsics_source': f'{seq}/calib.txt P{args.camera}',
        'pose_source': str(args.poses), 'num_kitti_frames': len(times),
        'scenes': [{'name': v['name'], 'start_s': v['start_s'],
                    'stop_s': v['stop_s'], 'frames': v['frames'],
                    'image_count': v['image_count']} for v in work],
        'notes': [
            'Known poses remove camera-pose estimation, but SfM points are not GT.',
            'Each scene is triangulated separately; outside-window tracks are not available.',
            'Images must be KITTI rectified PNGs (PINHOLE intrinsics from Pk).',
            'Do NOT bundle-adjust poses in this reference experiment.',
        ],
    }
    _write_json(args.out / 'manifest.json', report)
    LOG.info('Prepared %d GT-pose scenes in %s', len(work), args.out)
    return 0


def _run_colmap(cmd: list[str], log_path: Path, timeout_sec: int) -> None:
    LOG.info('Running %s', ' '.join(cmd[:2]))
    try:
        with log_path.open('w', encoding='utf-8') as log:
            log.write(' '.join(cmd) + '\n\n')
            log.flush()
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout_sec)
    except subprocess.CalledProcessError as exc:
        tail = log_path.read_text(encoding='utf-8', errors='replace')[-2000:]
        raise RuntimeError(f'COLMAP failed ({exc.returncode}): {cmd[:2]}\n{tail}') from exc


def _db_images(db_path: Path, expected_names: list[str],
               width: int, height: int, params: list[float]) -> list[tuple[int, str, int]]:
    """Match IDs against the ACTUAL feature-extractor database, never guess IDs."""
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute('SELECT image_id, name, camera_id FROM images').fetchall()
        cams = conn.execute('SELECT camera_id, model, width, height, params FROM cameras').fetchall()
    got = {name: (int(image_id), int(camera_id)) for image_id, name, camera_id in rows}
    if len(got) != len(rows) or set(got) != set(expected_names):
        raise ValueError('COLMAP database image IDs/names disagree with expected KITTI frames')
    used_cameras = {v[1] for v in got.values()}
    if len(used_cameras) != 1:
        raise ValueError('Known-Pose experiment requires exactly one shared camera')
    camera = {int(c[0]): c for c in cams}[next(iter(used_cameras))]
    # Current COLMAP model ID 1 is PINHOLE (fx fy cx cy).
    param_db = np.frombuffer(camera[4], dtype='<f8')
    if (int(camera[1]) != 1 or int(camera[2]) != width or int(camera[3]) != height or
            param_db.shape != (4,) or
            not np.allclose(param_db, params, rtol=1e-7, atol=1e-5)):
        raise ValueError('COLMAP feature database intrinsics differ from KITTI Pk / image dimensions')
    return [(got[name][0], name, got[name][1]) for name in expected_names]


def _write_seed_model(scene_path: Path, scene: dict[str, Any],
                      records: list[tuple[int, str, int]]) -> None:
    poses = np.load(scene_path / 'gt_T_world_camera.npy', allow_pickle=False)
    if poses.shape != (len(records), 4, 4):
        raise ValueError('GT camera pose count mismatch')
    target = scene_path / 'model_gt'
    target.mkdir()
    camera_id = records[0][2]
    params = ' '.join(f'{x:.17g}' for x in scene['camera_params'])
    (target / 'cameras.txt').write_text(
        f"{camera_id} PINHOLE {scene['image_width']} {scene['image_height']} {params}\n")
    lines = []
    for k, (image_id, name, cid) in enumerate(records):
        tcw = np.linalg.inv(poses[k])
        q = Rotation.from_matrix(tcw[:3, :3]).as_quat()
        qwxyz = [q[3], q[0], q[1], q[2]]
        vals = [*qwxyz, *tcw[:3, 3]]
        lines.append(f"{image_id} " + ' '.join(f'{v:.17g}' for v in vals) +
                     f' {cid} {name}\n\n')
    (target / 'images.txt').write_text(''.join(lines), encoding='utf-8')
    (target / 'points3D.txt').write_text('', encoding='utf-8')


def _scene_entries(out: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind') != 'kitti_fixed_gt_pose_sfm_scene_candidates':
        raise ValueError('Wrong manifest type; run prepare first')
    scenes = []
    for item in manifest['scenes']:
        scene = json.loads((out / item['name'] / 'scene.json').read_text(encoding='utf-8'))
        if item['frames'] != scene['frames']:
            raise ValueError(f"Scene file / manifest disagree: {item['name']}")
        scenes.append(scene)
    return manifest, scenes


def _validate_reconstruction_options(args: argparse.Namespace) -> None:
    if args.timeout_sec < 1 or args.sequential_overlap < 1:
        raise ValueError('Invalid COLMAP timeout/overlap')
    if shutil.which(args.colmap) is None and not Path(args.colmap).is_file():
        raise FileNotFoundError(f'COLMAP binary not found: {args.colmap}')


def _reconstruct_fixed_model(base: Path, scene: dict[str, Any],
                             image_dir: Path, args: argparse.Namespace) -> None:
    """Common COLMAP runner. Each supplied model is constructed from fixed GT poses."""
    db = base / 'database.db'
    seed = base / 'model_gt'
    sparse = base / 'sparse'
    sparse_txt = base / 'sparse_txt'
    if any(path.exists() for path in (db, seed, sparse, sparse_txt)):
        raise FileExistsError(
            f'{base}: COLMAP files exist. Use a fresh directory to avoid stale-model reuse.')
    feature = [args.colmap, 'feature_extractor',
               '--database_path', str(db), '--image_path', str(image_dir),
               '--image_list_path', str(base / 'image_list.txt'),
               '--ImageReader.single_camera', '1',
               '--ImageReader.camera_model', 'PINHOLE',
               '--ImageReader.camera_params',
               ','.join(f'{x:.17g}' for x in scene['camera_params'])]
    _run_colmap(feature, base / '01_feature.log', args.timeout_sec)
    records = _db_images(db, scene['image_names'], scene['image_width'],
                         scene['image_height'], scene['camera_params'])
    _write_seed_model(base, scene, records)
    if args.matcher == 'sequential':
        matcher = [args.colmap, 'sequential_matcher', '--database_path', str(db),
                   '--SequentialMatching.overlap', str(args.sequential_overlap)]
    else:
        matcher = [args.colmap, 'exhaustive_matcher', '--database_path', str(db)]
    _run_colmap(matcher, base / '02_match.log', args.timeout_sec)
    sparse.mkdir()
    _run_colmap([args.colmap, 'point_triangulator',
                 '--database_path', str(db), '--image_path', str(image_dir),
                 '--input_path', str(seed), '--output_path', str(sparse)],
                base / '03_triangulate.log', args.timeout_sec)
    sparse_txt.mkdir()
    _run_colmap([args.colmap, 'model_converter', '--input_path', str(sparse),
                 '--output_path', str(sparse_txt), '--output_type', 'TXT'],
                base / '04_convert.log', args.timeout_sec)
    verify_fixed_poses(base, scene)


def reconstruct(args: argparse.Namespace) -> int:
    _validate_reconstruction_options(args)
    manifest, scenes = _scene_entries(args.out)
    if args.only:
        available = {s['name'] for s in scenes}
        unknown = set(args.only) - available
        if unknown:
            raise ValueError(f'Unknown local scene IDs: {sorted(unknown)}')
    for scene in scenes:
        if args.only and scene['name'] not in args.only:
            continue
        _reconstruct_fixed_model(args.out / scene['name'], scene,
                                 Path(manifest['image_path']), args)
        LOG.info('Completed fixed-pose local SfM: %s', scene['name'])
    return 0


def _make_global_scene(args: argparse.Namespace,
                       manifest: dict[str, Any]) -> dict[str, Any]:
    """Prepare *all* sequence camera frames; never use selected-window subsets."""
    seq = Path(manifest['sequence'])
    times, twc, intrinsics = load_kitti_camera_gt(
        seq / 'times.txt', Path(manifest['poses_path']), seq / 'calib.txt',
        int(manifest['camera_index']))
    if len(times) != int(manifest['num_kitti_frames']):
        raise ValueError('KITTI sequence changed since prepare: frame count mismatch')
    image_dir = Path(manifest['image_path'])
    names = [f'{frame:06d}.png' for frame in range(len(times))]
    sizes = {read_png_size(image_dir / name) for name in names}
    if len(sizes) != 1:
        raise ValueError(f'Global SfM images have inconsistent dimensions: {sizes}')
    width, height = sizes.pop()
    params = [float(intrinsics[0, 0]), float(intrinsics[1, 1]),
              float(intrinsics[0, 2]), float(intrinsics[1, 2])]
    # Check every candidate uses exactly the same calibration and dimensions.
    for window in manifest['scenes']:
        local = json.loads((args.out / window['name'] / 'scene.json').read_text())
        if (local['camera_params'] != params or
                (local['image_width'], local['image_height']) != (width, height)):
            raise ValueError(f'{window["name"]}: inconsistent local/global calibration')
    base = args.out / 'global'
    base.mkdir(exist_ok=False)
    scene = {
        'name': 'global', 'start_s': float(times[0]), 'stop_s': float(times[-1]),
        'frames': list(range(len(times))), 'image_names': names,
        'image_count': len(names), 'image_width': width, 'image_height': height,
        'camera_model': 'PINHOLE', 'camera_params': params,
        'pose_convention': f'T_world_rectified_camera_{manifest["camera_index"]}',
        'scope': 'full_sequence_all_images_not_selected_windows',
    }
    np.save(base / 'gt_T_world_camera.npy', twc)
    (base / 'image_list.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
    _write_json(base / 'scene.json', scene)
    return scene


def reconstruct_global(args: argparse.Namespace) -> int:
    """Triangulate exactly once using ALL available KITTI camera frames."""
    _validate_reconstruction_options(args)
    manifest, _ = _scene_entries(args.out)
    if (args.out / 'global').exists():
        raise FileExistsError(f'Global model already exists: {args.out / "global"}')
    scene = _make_global_scene(args, manifest)
    _reconstruct_fixed_model(args.out / 'global', scene,
                             Path(manifest['image_path']), args)
    LOG.info('Completed GT-fixed global SfM: %d images', scene['image_count'])
    return 0


def _read_images_txt(path: Path) -> dict[int, dict[str, Any]]:
    out = {}
    with path.open('r', encoding='utf-8') as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            tokens = line.split()
            if len(tokens) != 10:
                raise ValueError(f'Invalid COLMAP images.txt pose row: {line[:90]}')
            image_id = int(tokens[0])
            xy_line = fh.readline()
            if not xy_line:  # Empty POINTS2D requires a blank second line.
                raise ValueError('Missing COLMAP images.txt second line')
            values = xy_line.split()
            if len(values) % 3:
                raise ValueError('Invalid POINTS2D triples')
            obs = np.asarray(
                [[float(values[i]), float(values[i + 1])]
                 for i in range(0, len(values), 3)], dtype=np.float64).reshape(-1, 2)
            pids = [int(values[i + 2]) for i in range(0, len(values), 3)]
            if image_id in out:
                raise ValueError('Duplicate COLMAP image ID')
            out[image_id] = {
                'name': tokens[9], 'camera_id': int(tokens[8]),
                'qvec': np.asarray([float(x) for x in tokens[1:5]]),
                'tvec': np.asarray([float(x) for x in tokens[5:8]]),
                'xy': obs, 'point_ids': pids,
            }
    return out


def _pose_matrix(record: dict[str, Any]) -> np.ndarray:
    q = np.asarray(record['qvec'], dtype=np.float64)
    if not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-4:
        raise ValueError('Bad COLMAP rotation quaternion')
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    t[:3, 3] = record['tvec']
    return t


def verify_fixed_poses(base: Path, scene: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Refuse to call a model GT-pose SfM if COLMAP changed camera poses."""
    expected = _read_images_txt(base / 'model_gt' / 'images.txt')
    reconstructed = _read_images_txt(base / 'sparse_txt' / 'images.txt')
    if set(expected) != set(reconstructed):
        raise ValueError('COLMAP output has different camera/image IDs from GT model')
    if {x['name'] for x in expected.values()} != set(scene['image_names']):
        raise ValueError('GT model does not match prepared scene images')
    for image_id, e in expected.items():
        r = reconstructed[image_id]
        if e['name'] != r['name'] or e['camera_id'] != r['camera_id']:
            raise ValueError('COLMAP output has changed camera or image identity')
        est, gt = _pose_matrix(r), _pose_matrix(e)
        deg = math.degrees(Rotation.from_matrix(est[:3, :3] @ gt[:3, :3].T).magnitude())
        dist = float(np.linalg.norm(est[:3, 3] - gt[:3, 3]))
        if deg > 1e-5 or dist > 1e-6:
            raise ValueError(f'COLMAP changed fixed pose for {e["name"]}: {deg:g}deg {dist:g}m')
    # Check intrinsics are still the calibrated rectified Pk, not self-calibrated.
    camera_lines = [
        x.split() for x in (base / 'sparse_txt' / 'cameras.txt').read_text().splitlines()
        if x.strip() and not x.lstrip().startswith('#')
    ]
    if len(camera_lines) != 1:
        raise ValueError('Expected one shared rectified KITTI camera')
    c = camera_lines[0]
    if (len(c) != 8 or c[1] != 'PINHOLE' or
            (int(c[2]), int(c[3])) != (scene['image_width'], scene['image_height']) or
            not np.allclose([float(x) for x in c[4:]], scene['camera_params'],
                            atol=1e-5, rtol=1e-7)):
        raise ValueError('COLMAP camera intrinsics differ from fixed KITTI calibration')
    return reconstructed


def _iter_points(path: Path):
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            fields = line.split()
            if len(fields) < 12 or (len(fields) - 8) % 2:
                raise ValueError('Invalid COLMAP points3D.txt track row')
            point_id = int(fields[0])
            xyz = np.asarray([float(v) for v in fields[1:4]], dtype=np.float64)
            if not np.isfinite(xyz).all():
                continue
            pairs = [(int(fields[j]), int(fields[j + 1]))
                     for j in range(8, len(fields), 2)]
            yield point_id, xyz, pairs


def _score_scene(base: Path, scene: dict[str, Any], args: argparse.Namespace, *,
                 global_images: dict[int, dict[str, Any]] | None = None,
                 source_model: Path | None = None) -> dict[str, Any]:
    """Score only in-window feature observations.

    With global_images, XYZ is triangulated from the FULL sequence even though
    track support, parallax, reprojection and grid coverage use only observations
    from this window. This is a deliberately global-information-assisted proxy.
    """
    if global_images is None:
        images = verify_fixed_poses(base, scene)
        is_global = False
    else:
        selected_names = set(scene['image_names'])
        images = {i: record for i, record in global_images.items()
                  if record['name'] in selected_names}
        if len(images) != len(selected_names):
            missing = selected_names - {r['name'] for r in images.values()}
            raise ValueError(f"Global SfM missing window image poses: {sorted(missing)[:5]}")
        is_global = True
    width, height = scene['image_width'], scene['image_height']
    fx, fy, cx, cy = scene['camera_params']
    mats = {i: _pose_matrix(r) for i, r in images.items()}
    centers = {i: -m[:3, :3].T @ m[:3, 3] for i, m in mats.items()}
    bins = {i: np.zeros((args.grid_y, args.grid_x), dtype=bool) for i in images}
    all_points = qualified = 0
    external_tracks = external_qualified = 0
    lengths, parallax, reproj = [], [], []
    points_path = (source_model if source_model is not None
                   else base / 'sparse_txt' / 'points3D.txt')
    for point_id, xyz, full_track in _iter_points(points_path):
        track = [(image_id, point_index) for image_id, point_index in full_track
                 if image_id in images]
        if not is_global and len(track) != len(full_track):
            raise ValueError(f'Local COLMAP point {point_id} references an unknown image')
        if not track:
            continue  # Global point never observed in this window.
        all_points += 1
        observed_outside = len(track) < len(full_track)
        if observed_outside:
            external_tracks += 1
        if len(track) < args.min_track:
            continue
        ray_vectors = []
        err = []
        good_observations = []
        for img_id, feature_idx in track:
            if img_id not in images:
                raise ValueError(f'Point track refers to unknown image: {img_id}')
            record = images[img_id]
            if feature_idx < 0 or feature_idx >= len(record['xy']):
                raise ValueError(f'Invalid POINT2D_IDX for point {point_id}')
            if record['point_ids'][feature_idx] != point_id:
                raise ValueError(f'Inconsistent image/point track for point {point_id}')
            cam = mats[img_id][:3, :3] @ xyz + mats[img_id][:3, 3]
            if not np.isfinite(cam).all() or cam[2] <= 1e-8:
                continue
            xy_pred = np.asarray([fx * cam[0] / cam[2] + cx,
                                  fy * cam[1] / cam[2] + cy])
            xy_obs = record['xy'][feature_idx]
            if not np.isfinite(xy_obs).all():
                continue
            err.append(float(np.linalg.norm(xy_pred - xy_obs)))
            ray = xyz - centers[img_id]
            norm = np.linalg.norm(ray)
            if norm < 1e-8:
                continue
            ray_vectors.append(ray / norm)
            good_observations.append((img_id, xy_obs))
        if len(good_observations) < args.min_track:
            continue
        rays = np.asarray(ray_vectors)
        cosine = np.clip(rays @ rays.T, -1., 1.)
        angle = math.degrees(math.acos(float(np.min(cosine))))
        pixel_error = float(np.median(err))
        if angle < args.min_parallax_deg or pixel_error > args.max_reproj_px:
            continue
        qualified += 1
        if observed_outside:
            external_qualified += 1
        lengths.append(len(good_observations))
        parallax.append(angle)
        reproj.append(pixel_error)
        for img_id, xy in good_observations:
            if 0 <= xy[0] < width and 0 <= xy[1] < height:
                bx = min(args.grid_x - 1, int(xy[0] * args.grid_x / width))
                by = min(args.grid_y - 1, int(xy[1] * args.grid_y / height))
                bins[img_id][by, bx] = True
    n_images = len(images)
    mean_grid = float(np.mean([b.mean() for b in bins.values()]))
    median_parallax = float(np.median(parallax)) if qualified else 0.
    median_error = float(np.median(reproj)) if qualified else 0.
    points_per_image = qualified / n_images
    # Heuristic ranking, NOT Fisher information or calibrated extrinsic utility.
    score = (float(np.log1p(points_per_image) * np.log1p(median_parallax)
                   * mean_grid / (1. + median_error)) if qualified else 0.)
    return {
        'name': scene['name'], 'start_s': scene['start_s'], 'stop_s': scene['stop_s'],
        'image_count': n_images, 'triangulated_points': all_points,
        'qualified_points': qualified, 'qualified_fraction': qualified / max(all_points, 1),
        'source_scope': 'global_xyz_window_observations' if is_global else 'local_xyz',
        'points_with_external_observations': external_tracks,
        'qualified_points_with_external_observations': external_qualified,
        'qualified_points_per_image': points_per_image,
        'median_track_length': float(np.median(lengths)) if lengths else 0.,
        'median_parallax_deg': median_parallax,
        'parallax_p10_deg': float(np.percentile(parallax, 10)) if parallax else 0.,
        'median_reproj_error_px': median_error,
        'reproj_p95_px': float(np.percentile(reproj, 95)) if reproj else 0.,
        'mean_image_grid_coverage': mean_grid,
        'geometry_score_heuristic': score,
        'eligible': qualified >= args.min_qualified_points,
    }


def evaluate(args: argparse.Namespace) -> int:
    if (args.min_track < 2 or args.min_parallax_deg < 0 or
            args.max_reproj_px <= 0 or args.grid_x < 1 or args.grid_y < 1 or
            args.min_qualified_points < 1 or args.max_scenes < 1):
        raise ValueError('Invalid geometry quality constraints')
    _, scenes = _scene_entries(args.out)
    rows = []
    for scene in scenes:
        base = args.out / scene['name']
        if not (base / 'sparse_txt' / 'points3D.txt').is_file():
            raise FileNotFoundError(f"Run reconstruct for scene {scene['name']} first")
        row = _score_scene(base, scene, args)
        rows.append(row)
        LOG.info('%s: qualified=%d (%.1f/frame), median parallax=%.2f°, grid=%.3f',
                 row['name'], row['qualified_points'],
                 row['qualified_points_per_image'], row['median_parallax_deg'],
                 row['mean_image_grid_coverage'])
    ranked = sorted([x for x in rows if x['eligible']],
                    key=lambda x: (-x['geometry_score_heuristic'], x['start_s'], x['name']))
    selected = []
    for row in ranked:
        if (len(selected) < args.max_scenes and all(
                row['stop_s'] <= s['start_s'] or row['start_s'] >= s['stop_s']
                for s in selected)):
            selected.append(row)
    fields = list(rows[0])
    with (args.out / 'scene_scores.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        'schema_version': 1, 'kind': 'gt_fixed_camera_pose_sparse_sfm_scene_selection',
        'config': {k: v for k, v in vars(args).items()
                   if k not in ('func',) and not isinstance(v, Path)},
        'scenes_ranked': ranked, 'scenes_selected': selected, 'all_candidates': rows,
        'scenes_selected_count': len(selected),
        'selection_basis': 'heuristic_sparse_feature_track_geometry_not_calibration_accuracy',
        'caveats': [
            'KITTI camera poses are GT reference; triangulated points are not GT.',
            'Only independently reconstructed scene tracks contribute: no cross-window tracks.',
            'Point abundance is influenced by scene texture/lighting, not just geometry.',
            'Track parallax and reprojection error do not measure LiDAR-camera calibration utility.',
            'Windows are ranked by an explicit non-probabilistic heuristic score.',
            'Unverified KITTI camera/LiDAR synchronization and dynamic objects remain limitations.',
        ],
    }
    _write_json(args.out / 'scene_selection.json', report)
    return 0


def evaluate_global(args: argparse.Namespace) -> int:
    """Score only in-window tracks, using globally triangulated 3D positions."""
    if (args.min_track < 2 or args.min_parallax_deg < 0 or
            args.max_reproj_px <= 0 or args.grid_x < 1 or args.grid_y < 1 or
            args.min_qualified_points < 1 or args.max_scenes < 1 or
            not np.isfinite([args.min_parallax_deg, args.max_reproj_px]).all()):
        raise ValueError('Invalid geometry quality constraints')
    _, scenes = _scene_entries(args.out)
    base = args.out / 'global'
    meta_path = base / 'scene.json'
    if not meta_path.is_file():
        raise FileNotFoundError('Run reconstruct-global before evaluate-global')
    global_scene = json.loads(meta_path.read_text(encoding='utf-8'))
    if global_scene.get('scope') != 'full_sequence_all_images_not_selected_windows':
        raise ValueError('Not a full-sequence known-pose global model')
    images = verify_fixed_poses(base, global_scene)
    points = base / 'sparse_txt' / 'points3D.txt'
    if not points.is_file():
        raise FileNotFoundError('Missing global points3D.txt')
    rows = [_score_scene(base, scene, args, global_images=images, source_model=points)
            for scene in scenes]
    ranked = sorted((r for r in rows if r['eligible']),
                    key=lambda r: (-r['geometry_score_heuristic'],
                                   r['start_s'], r['name']))
    selected = []
    for row in ranked:
        if (len(selected) < args.max_scenes and
                all(row['stop_s'] <= x['start_s'] or row['start_s'] >= x['stop_s']
                    for x in selected)):
            selected.append(row)
    with (args.out / 'global_scene_scores.csv').open(
            'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(args.out / 'global_scene_selection.json', {
        'schema_version': 1,
        'kind': 'gt_fixed_camera_pose_global_sfm_window_geometry_selection',
        'model_scope': 'all_KITTI_camera_frames',
        'score_scope': 'global_xyz_with_in_window_feature_observations',
        'config': {k: v for k, v in vars(args).items()
                   if k != 'func' and not isinstance(v, Path)},
        'scenes_ranked': ranked, 'scenes_selected': selected,
        'scenes_selected_count': len(selected), 'all_candidates': rows,
        'caveats': [
            'Global XYZ was triangulated using outside-window observations.',
            'Window-only parallax/reprojection checks do not remove XYZ data leakage.',
            'Feature count and coverage depend on image texture, not just 3D geometry.',
            'No calibration was run; scene scores are not extrinsic accuracy.',
        ],
    })
    LOG.info('Global SfM scoring finished: %d candidates, %d selected',
             len(rows), len(selected))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare', help='Create KITTI GT camera trajectories and scene windows')
    prep.add_argument('--sequence', type=Path, required=True,
                      help='KITTI Odometry sequences/07')
    prep.add_argument('--poses', type=Path, required=True, help='KITTI Odometry poses/07.txt')
    prep.add_argument('--images', type=Path, help='Override sequence/image_2 directory')
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--camera', type=int, choices=[0, 1, 2, 3], default=2)
    prep.add_argument('--windows', nargs='+', choices=['005', '012'], default=['005', '012'])
    prep.add_argument('--sliding', action='store_true',
                      help='Generate all sliding windows instead of the 005/012 presets')
    prep.add_argument('--start-s', type=float, help='Sliding lower time boundary')
    prep.add_argument('--stop-s', type=float, help='Sliding upper time boundary')
    prep.add_argument('--window-s', type=float, default=15.)
    prep.add_argument('--stride-s', type=float, default=7.5)
    prep.add_argument('--min-images', type=int, default=10)
    prep.add_argument('--frame-step', type=int, default=1)
    prep.add_argument('--max-images', type=int, help='Equalize max image budget per scene')
    prep.set_defaults(func=prepare)

    run = sub.add_parser('reconstruct', help='COLMAP known-pose triangulation per scene')
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--colmap', type=str, default='colmap')
    run.add_argument('--matcher', choices=['sequential', 'exhaustive'], default='sequential')
    run.add_argument('--sequential-overlap', type=int, default=15)
    run.add_argument('--timeout-sec', type=int, default=7200)
    run.add_argument('--only', nargs='*', help='Subset of scene IDs')
    run.set_defaults(func=reconstruct)

    eva = sub.add_parser('evaluate', help='Verify fixed GT pose and rank SfM geometry')
    eva.add_argument('--out', type=Path, required=True)
    eva.add_argument('--min-track', type=int, default=3)
    eva.add_argument('--min-parallax-deg', type=float, default=1.0)
    eva.add_argument('--max-reproj-px', type=float, default=2.0)
    eva.add_argument('--grid-x', type=int, default=8)
    eva.add_argument('--grid-y', type=int, default=6)
    eva.add_argument('--min-qualified-points', type=int, default=100)
    eva.add_argument('--max-scenes', type=int, default=2)
    eva.set_defaults(func=evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    start = time.perf_counter()
    result = args.func(args)
    LOG.info('%s complete in %.1fs', args.action, time.perf_counter() - start)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
