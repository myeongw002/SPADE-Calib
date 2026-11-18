"""Regressions for all-voxel candidate search + full-scan finite-patch refit."""
import numpy as np
import pytest

from spade_calib.correspondence.sfm_lidar_planes import PatchSettings, extract_lidar_patches


def cloud_with_dense_planar_returns(seed=23):
    rng = np.random.default_rng(seed)
    uv = np.array([(x, y) for x in np.linspace(1.0, 3.6, 25)
                   for y in np.linspace(-1.7, 1.7, 25)])
    exact = np.column_stack([uv, np.full(len(uv), 2.7)])
    # Each independently occupied voxel contributes several original returns.
    cloud = np.repeat(exact, 6, axis=0)
    cloud += rng.normal(0, .003, cloud.shape)
    scan_id = np.tile(np.arange(6), len(exact))
    return cloud, scan_id


def config(**extra):
    kw = dict(voxel_size_m=.08, min_range_m=.1, max_range_m=20.,
              distance_threshold_m=.04, min_inliers=70, max_planes=2,
              ransac_iterations=75, connectivity_radius_m=.3,
              orientation_aware=False)
    kw.update(extra)
    return PatchSettings(**kw)


def test_uncapped_voxel_search_and_original_scan_refinement():
    xyz, sid = cloud_with_dense_planar_returns()
    patches, report = extract_lidar_patches(xyz, sid, config())
    assert patches
    p = patches[0]
    assert report['random_sampling_applied'] is False
    assert report['voxel_points'] == report['sampled_points']
    assert report['sampled_points'] < len(xyz)
    assert len(p.support_xyz_lidar) > 3 * report['sampled_points']
    assert len(p.support_xyz_lidar) > 1000
    assert len(p.scan_ids) == 6
    assert np.median(np.abs(p.support_xyz_lidar @ p.normal + p.offset_m)) < .02
    assert abs(p.normal @ [0, 0, 1]) > .99
    assert report['refinement_full_resolution'] is True
    assert report['full_resolution_patch_points'][0] == len(p.support_xyz_lidar)


def test_optional_cap_and_voxel_only_ablation():
    xyz, sid = cloud_with_dense_planar_returns()
    old, meta = extract_lidar_patches(xyz, sid, config(
        max_sampled_points=300, refine_with_full_points=False))
    assert meta['sampled_points'] == 300
    assert meta['random_sampling_applied'] is True
    assert not meta['refinement_full_resolution']
    assert old
    assert all(len(p.support_xyz_lidar) <= 300 for p in old)


def test_refit_keeps_disconnected_coplanar_regions_separate():
    xyz, sid = cloud_with_dense_planar_returns()
    # Same plane equation but disconnected at a distance > 1 m.
    shifted = xyz.copy()
    shifted[:, 0] += 15
    combined = np.vstack([xyz, shifted])
    scans = np.tile(sid, 2)
    patches, meta = extract_lidar_patches(combined, scans, config(
        max_planes=1, connectivity_radius_m=.3))
    assert patches
    patch = patches[0]
    extent = np.ptp(patch.support_xyz_lidar[:, 0])
    assert extent < 4.0, 'distant coplanar surface must not be merged in full refit'


def test_invalid_negative_max_sampled_points():
    xyz, sid = cloud_with_dense_planar_returns()
    with pytest.raises(ValueError):
        extract_lidar_patches(xyz, sid, config(max_sampled_points=-1))