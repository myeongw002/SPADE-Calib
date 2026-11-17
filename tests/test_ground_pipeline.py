"""Synthetic ROI-free Camera/LiDAR ground extraction and correspondence checks.

Creates a fully metric image (optical Z) and independent 3-D LiDAR samples
of the SAME horizontal ground. No KITTI GT transform is loaded by fitting.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('cv2')

from spade_calib.camera.planes import (CameraGeometry, CameraPlaneSettings,
                                       dense_metric_xyz, extract_camera_planes)
from spade_calib.lidar.ground_planes import LidarGroundSettings, extract_lidar_ground
from spade_calib.correspondence.ground_matcher import match_ground
from spade_calib.calibration.ground_pipeline import prepare_ground_inputs
from spade_calib.evaluation.kitti import evaluate_camera_lidar


def build_ground_example(*, seed: int = 10):
    height, width = 120, 160
    K = np.array([[105., 0., 79.5], [0., 105., 54.5], [0., 0., 1.]])
    camera = CameraGeometry(K, [], width, height)
    # For our LiDAR->Camera rotation, LiDAR (z=-1.5) maps to camera y=+1.5.
    R = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
    T = np.eye(4)
    T[:3, :3] = R
    v = np.arange(height)[:, None].astype(float)
    zray_y = (v - camera.K[1, 2]) / camera.K[1, 1]
    values = np.where(zray_y > .09, 1.5 / np.maximum(zray_y, .09), np.nan)
    depth = np.broadcast_to(values, (height, width)).copy().astype(np.float32)
    # Include a few depth-invalid pixel holes, but keep enough metric plane support.
    depth[90:94, 36:44] = np.nan

    rng = np.random.default_rng(seed)
    n = 4000
    ground = np.column_stack([
        rng.uniform(4.0, 17., n),
        rng.uniform(-3.4, 3.4, n),
        -1.5 + rng.normal(0., .008, n),
    ])
    non_ground = np.column_stack([
        rng.uniform(4., 17., 700),
        rng.uniform(-3., 3., 700),
        rng.uniform(-.2, 2., 700),
    ])
    return camera, depth, np.concatenate([ground, non_ground]), T


def test_metric_depth_convention_and_camera_plane_extraction():
    camera, depth, _, T = build_ground_example()
    xyz, good, _ = dense_metric_xyz(depth, camera, width=160, height=120)
    assert good.any()
    np.testing.assert_allclose(xyz[88, 80, 1], 1.5, atol=.05)
    planes, report = extract_camera_planes(
        depth, camera,
        CameraPlaneSettings(processing_width=160, processing_height=120,
                            min_region_pixels=70),
    )
    assert report['camera_plane_count'] >= 1
    expected = T[:3, :3] @ np.array([0., 0., 1.])
    assert any(abs(c.plane.normal @ expected) > .98 for c in planes)
    assert all(c.image_shape == (120, 160) for c in planes)


def test_lidar_ground_ransac_and_camera_guided_matching():
    camera, depth, cloud, T = build_ground_example()
    cam, _ = extract_camera_planes(
        depth, camera,
        CameraPlaneSettings(processing_width=160, processing_height=120,
                            min_region_pixels=70),
    )
    lidar, info = extract_lidar_ground(
        cloud, LidarGroundSettings(iterations=280, min_inliers=110,
                                   max_sampled_points=6000),
    )
    assert info['final_inliers'] > 1500
    assert lidar.plane.offset_m == pytest.approx(1.5, abs=.08)
    match, meta = match_ground(cam, lidar, depth, camera, T)
    assert match is not None, meta
    assert meta['selected_camera_plane'] is not None
    assert meta['correspondence_status'] == 'provisional_requires_independent_review'
    assert np.linalg.norm(match.lidar.normal - np.array([0, 0, 1])) < .1
    # Only a geometry alignment test; not evidence of real extrinsic recovery.
    assert abs(match.lidar.offset_m - 1.5) < .10
    assert abs(abs(match.camera.offset_m) - 1.5) < .10


def test_pipeline_requires_matching_camera_and_lidar_reference_frame(tmp_path):
    camera, depth, cloud, T = build_ground_example()
    handeye = tmp_path / 'handeye.json'
    handeye.write_text(json.dumps({
        'handeye': {
            'convention': 'T_camera_lidar (multiply lidar point to camera frame)',
            'T_camera_lidar': T.tolist(), 'sfm_scale_to_metric': 10.,
        }
    }))
    camera_json = tmp_path / 'camera.json'
    camera_json.write_text(json.dumps({
        'K': camera.K.tolist(), 'distortion': [], 'width': camera.width,
        'height': camera.height, 'depth_type': 'camera_z_m',
    }))
    depth_path = tmp_path / 'depth.npy'
    np.save(depth_path, depth)
    cloud_path = tmp_path / 'cloud.npz'
    np.savez(cloud_path, xyz=cloud, frame_id=np.str_('kitti_lidar_frame_3'),
             reference_timestamp_ns=np.int64(300_000_000),
             frame_ids=np.array([2, 3, 4]), motion_model=np.str_('rigid'))
    times = tmp_path / 'times.txt'
    times.write_text('0.0\n0.1\n0.2\n0.3\n0.4\n')
    args = SimpleNamespace(
        handeye=handeye, camera_json=camera_json, metric_depth=depth_path,
        cloud=cloud_path, kitti_times=times, frame_index=3,
        depth_frame_index=4, processing_width=160, processing_height=120,
        min_depth_m=1., max_depth_m=55., max_lidar_points=6000,
        lidar_ransac_iterations=280, min_lidar_inliers=110,
        min_ground_height_m=.35, max_ground_height_m=3.5,
        mask_dilation_px=10, min_guided_points=35,
    )
    with pytest.raises(ValueError, match='SAME frame'):
        prepare_ground_inputs(args)
    args.depth_frame_index = 3
    ground, diagnostics = prepare_ground_inputs(args)
    assert ground is not None
    assert ground['verified_by_independent_review'] is False
    assert diagnostics['accepted_ground_hypotheses'] == 1
    assert diagnostics['frame_alignment']['target_lidar_frame'] == 'kitti_lidar_frame_3'
    # Cloud has the correct name but incorrect time: refuse silent temporal mismatch.
    np.savez(cloud_path, xyz=cloud, frame_id=np.str_('kitti_lidar_frame_3'),
             reference_timestamp_ns=np.int64(400_000_000),
             frame_ids=np.array([2, 3, 4]), motion_model=np.str_('rigid'))
    with pytest.raises(ValueError, match='timestamp differs'):
        prepare_ground_inputs(args)


def test_camera_geometry_rejects_unscaled_depth_schema(tmp_path):
    bad = tmp_path / 'bad_camera.json'
    bad.write_text(json.dumps({'K': np.eye(3).tolist(), 'distortion': [],
                               'width': 160, 'height': 120,
                               'depth_type': 'raw_da2_inverse_depth'}))
    with pytest.raises(ValueError, match='camera_z_m'):
        CameraGeometry.from_json(bad)
