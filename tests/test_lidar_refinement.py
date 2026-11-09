"""Deterministic synthetic checks and opt-in D1 shape/numeric regression."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pytest
from spade_calib.lidar import (GlimTrajectory, LidarScan, accumulate_glim,
                                AccumulatedCloud, FreeSpaceSettings, target_free_space_mask,
                                filter_cloud, load_d1_scan_zip)


def motion_trajectory():
    return GlimTrajectory.from_array(np.array([
        [10.0, 0, 0, 0, 0, 0, 0, 1],
        [12.0, 2, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float64))


def scan(scan_id, timestamp_ns, points, dts):
    points=np.asarray(points, dtype=np.float64)
    n=len(points)
    return LidarScan(scan_id, timestamp_ns, points, np.array(dts, dtype=np.float64),
                     np.zeros(n, dtype=np.uint16), np.arange(n, dtype=np.int64))


def cloud(points, ids):
    points=np.array(points, dtype=np.float32)
    n=len(points)
    return AccumulatedCloud(points, np.asarray(ids,dtype=np.uint16),
                            np.zeros(n,dtype=np.float32),
                            np.zeros(n,dtype=np.uint16),np.arange(n,dtype=np.int64),
                            10_000_000_000)


def test_glim_linear_pose_and_no_extrapolation():
    t=motion_trajectory()
    r,p=t.at([10.,11.,12.])
    np.testing.assert_allclose(p[:,0],[0,1,2])
    np.testing.assert_allclose(r,np.repeat(np.eye(3)[None,...],3,axis=0))
    with pytest.raises(ValueError,match='extrapolate'):
        t.at(12.01)


def test_deskew_and_rigid_use_different_per_point_times():
    t=motion_trajectory()
    # Sensor moves +x 1m/s; second return happened 1 second later.
    s=scan(0,11_000_000_000,[[3,0,0],[3,0,0]],[0.,1.])
    a=accumulate_glim([s],t,target_timestamp_ns=11_000_000_000,mode='deskew')
    b=accumulate_glim([s],t,target_timestamp_ns=11_000_000_000,mode='rigid')
    np.testing.assert_allclose(a.xyz[:,0],[3,4],atol=1e-6)
    np.testing.assert_allclose(b.xyz[:,0],[3,3],atol=1e-6)
    np.testing.assert_array_equal(a.source_return_index,[0,1])
    np.testing.assert_array_equal(a.scan_id,[0,0])


def test_trajectory_rejects_bad_quaternion_and_duplicates():
    m=np.array([[10,0,0,0,0,0,0,1],[10,0,0,0,0,0,0,1]],dtype=float)
    with pytest.raises(ValueError,match='strictly'):
        GlimTrajectory.from_array(m)
    m[1,0]=11
    m[1,4:8]=0
    with pytest.raises(ValueError,match='zero-norm'):
        GlimTrajectory.from_array(m)


def test_target_free_space_keeps_target_and_rejects_in_front_ghost():
    # Target sees at 10m at azimuth=0, elevation=0. Other scans see 6m
    # (physically inconsistent free space), 10.2m (consistent), 12m (occluded).
    c=cloud([[10,0,0],[6,0,0],[10.2,0,0],[12,0,0],[5,5,0]],[5,2,3,4,2])
    keep=target_free_space_mask(c,target_scan_id=5)
    assert keep.tolist()==[True,False,True,True,True]
    out=filter_cloud(c,keep)
    np.testing.assert_array_equal(out.source_return_index,[0,2,3,4])
    assert len(out.xyz)==4


def test_free_space_target_needed_and_no_cross_frame_guess():
    c=cloud([[1,0,0],[2,0,0]],[0,0])
    with pytest.raises(ValueError,match='Target scan'):
        target_free_space_mask(c,target_scan_id=5)
    with pytest.raises(ValueError,match='resolution'):
        FreeSpaceSettings(azimuth_resolution_deg=0)
    with pytest.raises(ValueError,match='shape'):
        filter_cloud(c,np.array([True]))




def test_reference_recovery_accepts_only_joint_pgo():
    from spade_calib.lidar.refinement import SUPPORTED_METHODS, recover_reference_scan_corrections
    assert SUPPORTED_METHODS == ('Joint', 'PGO')
    c = cloud([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
               [2, 0, 0], [3, 0, 0], [2, 1, 0], [2, 0, 1]],
              [0, 0, 0, 0, 1, 1, 1, 1])
    xyz = c.xyz.copy()
    xyz[c.scan_id == 1] += np.array([0.1, -0.2, 0.3])
    r = recover_reference_scan_corrections(c, xyz, method='Joint', anchor_scan_id=0)
    assert r.max_residual_m < 1e-6
    assert r.scan_ids.tolist() == [0, 1]
    np.testing.assert_allclose(r.corrections_target_aligned[1, :3, 3], [0.1, -0.2, 0.3], atol=1e-6)
    from spade_calib.lidar import apply_scan_corrections
    replay = apply_scan_corrections(c, r.scan_ids, r.corrections_target_aligned)
    np.testing.assert_allclose(replay.xyz, xyz, atol=1e-6)
    with pytest.raises(ValueError, match='Only Joint and PGO'):
        recover_reference_scan_corrections(c, xyz, method='C1', anchor_scan_id=0)


def test_reference_zip_loader_rejects_non_selected_method(tmp_path):
    from spade_calib.lidar import load_archived_refined_xyz
    from zipfile import ZipFile
    from io import BytesIO
    z = tmp_path / 'reference.zip'
    with ZipFile(z, 'w') as f:
        raw = BytesIO()
        np.save(raw, np.array([[1., 2., 3.]], dtype=np.float32))
        f.writestr('D1/inputs/Joint_xyz.npy', raw.getvalue())
    loaded = load_archived_refined_xyz(z, 'Joint')
    np.testing.assert_array_equal(loaded, [[1., 2., 3.]])
    with pytest.raises(ValueError, match='Unsupported refinement'):
        load_archived_refined_xyz(z, 'B')
    with pytest.raises(ValueError, match='Expected exactly one'):
        load_archived_refined_xyz(z, 'PGO')


def test_recovery_requires_matching_reference_count():
    from spade_calib.lidar import recover_reference_scan_corrections
    c = cloud([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [0, 0, 0])
    with pytest.raises(ValueError, match='match original'):
        recover_reference_scan_corrections(c, np.ones((4,3)), method='PGO', anchor_scan_id=0)