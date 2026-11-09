"""Opt-in original D1 numerical tests: no private archive in public CI."""
from __future__ import annotations
import csv
import os
import numpy as np
import pytest
from spade_calib.lidar import load_d1_scan_zip, accumulate_glim


@pytest.mark.skipif(not os.getenv('SPADE_D1_RAW_ZIP') or not os.getenv('SPADE_D1_A_NPZ'),
                    reason='D1 inputs are private')
def test_d1_a_exact_reproduction():
    scans,traj,m=load_d1_scan_zip(os.environ['SPADE_D1_RAW_ZIP'])
    a=accumulate_glim(scans,traj,target_timestamp_ns=m['lidar_anchor_timestamp_ns'])
    with np.load(os.environ['SPADE_D1_A_NPZ'],allow_pickle=False) as f:
        expected=f['xyz']
    assert a.xyz.shape==expected.shape==(215504,3)
    # The source experiment's timestamp arithmetic matches our at_offset method.
    np.testing.assert_array_equal(a.xyz, expected)


@pytest.mark.skipif(not os.getenv('SPADE_D1_RAW_ZIP') or not os.getenv('SPADE_D1_B_CSV'),
                    reason='B original correction CSV not configured')
def test_d1_b_numerical_corrections():
    from spade_calib.lidar.pose_refinement import refine_d1_b
    scans,traj,m=load_d1_scan_zip(os.environ['SPADE_D1_RAW_ZIP'])
    a=accumulate_glim(scans,traj,target_timestamp_ns=m['lidar_anchor_timestamp_ns'])
    result=refine_d1_b(a)
    with open(os.environ['SPADE_D1_B_CSV'],encoding='utf-8',newline='') as f:
        ref=list(csv.DictReader(f))
    old=np.asarray([[float(r['tx_m']),float(r['ty_m']),float(r['tz_m']),
                     np.deg2rad(float(r['rx_deg'])),np.deg2rad(float(r['ry_deg'])),
                     np.deg2rad(float(r['rz_deg']))] for r in ref])
    assert old.shape==(10,6)
    np.testing.assert_allclose(result.corrections[:,:3],old[:,:3],atol=1e-4,rtol=0)
    np.testing.assert_allclose(result.corrections[:,3:],old[:,3:],atol=1e-5,rtol=0)
    assert all(step['pairs']==22 for step in result.history)
