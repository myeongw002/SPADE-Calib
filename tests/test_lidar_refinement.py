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


@pytest.mark.skipif(not os.getenv('SPADE_D1_RAW_ZIP') or not os.getenv('SPADE_D1_A_NPZ'),
                    reason='D1 original raw ZIP and reference A point cloud not configured')
def test_d1_original_a_numeric_regression():
    scans,t,m=load_d1_scan_zip(os.environ['SPADE_D1_RAW_ZIP'])
    a=accumulate_glim(scans,t,target_timestamp_ns=m['lidar_anchor_timestamp_ns'])
    with np.load(os.environ['SPADE_D1_A_NPZ'],allow_pickle=False) as original:
        expected=original['xyz']
    assert a.xyz.shape==expected.shape==(215504,3)
    # Different epoch subtraction order creates <1e-5m float rounding changes.
    assert np.max(np.linalg.norm(a.xyz-expected,axis=1)) < 2e-5
