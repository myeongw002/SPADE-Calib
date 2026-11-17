"""Synthetic DA2 metric alignment and COLMAP sparse projection regression tests.

No physical KITTI GT is used. These tests verify numerical geometry,
spatial holdout behavior, input safety and output units only.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('scipy')
pytest.importorskip('cv2')

from spade_calib.camera.metric_alignment import (
    MetricAlignmentSettings, align_relative_da2_to_sfm,
)
from spade_calib.camera.sfm_anchors import build_sfm_metric_anchors
from spade_calib.camera.planes import CameraGeometry
from spade_calib.io.legacy import LegacySfMAnchors
from spade_calib.cli import main


def synthetic_data(*, corrupt: bool = False):
    height, width = 120, 160
    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    z = 4. + .052 * uu + .034 * vv
    a, b = .006, -.011
    raw = ((1/z - b)/a).astype(np.float32)
    xs, ys = np.meshgrid(np.arange(8, width-8, 7), np.arange(7, height-7, 7))
    xy = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    rng = np.random.default_rng(100)
    depths = z[ys, xs].ravel() + rng.normal(0., .015, xy.shape[0])
    if corrupt:
        # Incorrect orientation makes inverse-depth regression impossible.
        raw = -raw
    anchors = LegacySfMAnchors(xy=xy, depth_m=depths,
        reprojection_error_px=np.full(len(xy), .25),
        image_shape=(height, width), sfm_scale=22.0)
    return raw, anchors, z


def test_spatial_holdout_recovers_metric_depth_without_gt():
    raw, anchors, z = synthetic_data()
    metric, info = align_relative_da2_to_sfm(raw, anchors)
    assert metric.shape == raw.shape
    assert info['heldout_anchor_count'] >= 10
    assert info['train_anchor_count'] >= 40
    assert info['heldout']['median_relative_error'] < .01
    assert np.nanmedian(np.abs(metric - z)) < .06
    assert info['a_positive'] == pytest.approx(.006, rel=.06)
    assert info['b_signed_inverse_m'] == pytest.approx(-.011, abs=.005)
    assert info['gt_used_for_fitting'] is False
    assert info['depth_type'] == 'camera_z_m'


def test_invalid_orientation_is_rejected():
    raw, anchors, _ = synthetic_data(corrupt=True)
    with pytest.raises(ValueError, match='positively correlated'):
        align_relative_da2_to_sfm(raw, anchors)


def test_spatial_holdout_rejects_insufficient_spread():
    raw, anchors, _ = synthetic_data()
    anchors = LegacySfMAnchors(
        xy=np.repeat([[12., 12.]], 60, axis=0), depth_m=np.ones(60)*5,
        reprojection_error_px=np.zeros(60),
        image_shape=anchors.image_shape, sfm_scale=22.)
    with pytest.raises(ValueError, match='spatial tiles'):
        align_relative_da2_to_sfm(raw, anchors)


def sample_colmap_model(tmp_path: Path, *, binary: bool):
    model = tmp_path / 'sparse'
    model.mkdir()
    camera = CameraGeometry(
        np.array([[100., 0, 79.], [0, 100., 59.], [0, 0, 1.]]),
        [], 160, 120)
    scale = 5.
    pts = []
    triplets = []
    for index in range(36):
        u = 15. + index % 9 * 14
        v = 12. + index // 9 * 23
        z = (3. + (u+v)/150.) / scale
        xyz = ((u-79.) / 100. * z, (v-59.) / 100. * z, z)
        pts.append((index + 100, xyz))
        triplets.append((u, v, index + 100))
    if binary:
        with (model / 'images.bin').open('wb') as f:
            f.write(struct.pack('<Q', 2))
            for image_id, name in [(1, '000105.png'), (2, '000106.png')]:
                f.write(struct.pack('<i4d3di', image_id, 1, 0, 0, 0, 0, 0, 0, 1))
                f.write(name.encode() + b'\0')
                f.write(struct.pack('<Q', len(triplets)))
                for u, v, pid in triplets:
                    f.write(struct.pack('<ddq', u, v, pid))
        with (model / 'points3D.bin').open('wb') as f:
            f.write(struct.pack('<Q', len(pts)))
            for idx, (pid, xyz) in enumerate(pts):
                f.write(struct.pack('<QdddBBBdQ', pid, *xyz, 255, 0, 0, .1, 2))
                f.write(struct.pack('<iiii', 1, idx, 2, idx))
    else:
        with (model / 'images.txt').open('w') as f:
            for image_id, name in [(1, '000105.png'), (2, '000106.png')]:
                f.write(f'{image_id} 1 0 0 0 0 0 0 1 {name}\n')
                f.write(' '.join(f'{u} {v} {pid}' for u, v, pid in triplets) + '\n')
        with (model / 'points3D.txt').open('w') as f:
            for idx, (pid, xyz) in enumerate(pts):
                x,y,z = xyz
                f.write(f'{pid} {x} {y} {z} 255 0 0 0.1 1 {idx} 2 {idx}\n')
    return model, camera, scale


@pytest.mark.parametrize('binary', [False, True])
def test_colmap_sparse_anchors_text_and_binary(tmp_path, binary):
    model, camera, scale = sample_colmap_model(tmp_path, binary=binary)
    arrays, info = build_sfm_metric_anchors(
        model, '000105.png', camera, scale)
    assert len(arrays['depth_m']) == 36
    assert np.min(arrays['depth_m']) > 3.
    assert np.max(arrays['reprojection_error_px']) < 1e-7
    assert info['accepted_metric_anchors'] == 36
    assert info['gt_used_for_fitting'] is False
    np.testing.assert_allclose(arrays['sfm_scale'], scale)


def test_align_depth_cli_generates_metric_for_prepare_ground(tmp_path):
    raw, anchors, truth = synthetic_data()
    src = tmp_path / 'relative.npy'
    np.save(src, raw)
    anchor_file = tmp_path / 'anchors.npz'
    np.savez(anchor_file, xy=anchors.xy, depth_m=anchors.depth_m,
             reprojection_error_px=anchors.reprojection_error_px,
             image_shape=np.array(anchors.image_shape), sfm_scale=22.)
    he = tmp_path / 'handeye.json'
    he.write_text(json.dumps({'handeye': {
        'convention': 'T_camera_lidar (multiply lidar point to camera frame)',
        'sfm_scale_to_metric': 22.}}))
    out = tmp_path / 'metric.npy'
    assert main(['align-depth', '--relative-depth', str(src),
                 '--anchors', str(anchor_file),
                 '--handeye', str(he), '--output', str(out)]) == 0
    metric = np.load(out, allow_pickle=False)
    assert np.nanmedian(np.abs(metric - truth)) < .06
    report = json.loads(out.with_suffix('.json').read_text())
    assert report['depth_type'] == 'camera_z_m'
    # Reject an anchor file independently scaled by a different SfM reconstruction.
    np.savez(anchor_file, xy=anchors.xy, depth_m=anchors.depth_m,
             reprojection_error_px=anchors.reprojection_error_px,
             image_shape=np.array(anchors.image_shape), sfm_scale=13.)
    assert main(['align-depth', '--relative-depth', str(src),
                 '--anchors', str(anchor_file),
                 '--handeye', str(he), '--output', str(out)]) == 2


def test_train_only_fit_resists_small_outlier_fraction():
    raw, anchors, truth = synthetic_data()
    corrupt = anchors.depth_m.copy()
    corrupt[::34] = corrupt[::34] * 1.6
    noisy = LegacySfMAnchors(
        anchors.xy, corrupt, anchors.reprojection_error_px,
        anchors.image_shape, anchors.sfm_scale)
    metric, info = align_relative_da2_to_sfm(raw, noisy)
    assert info['train_anchor_count'] > 40
    assert np.nanmedian(np.abs(metric - truth)) < .2
