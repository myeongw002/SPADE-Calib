"""Opt-in original D1 checks. No private data in public CI."""
from __future__ import annotations
import os
import numpy as np
import pytest
from spade_calib.lidar import (SUPPORTED_METHODS, load_d1_scan_zip,
                              accumulate_glim, load_archived_refined_xyz,
                              recover_reference_scan_corrections)


@pytest.mark.skipif(not os.getenv('SPADE_D1_RAW_ZIP') or not os.getenv('SPADE_D1_A_NPZ'),
                    reason='Original D1 raw ZIP and A reference unavailable')
def test_d1_glim_per_return_deskew():
    scans, traj, manifest = load_d1_scan_zip(os.environ['SPADE_D1_RAW_ZIP'])
    a = accumulate_glim(scans, traj, target_timestamp_ns=int(manifest['lidar_anchor_timestamp_ns']))
    with np.load(os.environ['SPADE_D1_A_NPZ'], allow_pickle=False) as data:
        expected = data['xyz']
    np.testing.assert_array_equal(a.xyz, expected)
    assert len(a.xyz) == 215504


@pytest.mark.skipif(not os.getenv('SPADE_D1_RAW_ZIP') or not os.getenv('SPADE_D1_REFERENCE_ZIP'),
                    reason='Private D1 raw/refined reference ZIPs unavailable')
@pytest.mark.parametrize('method', SUPPORTED_METHODS)
def test_selected_d1_original_outputs_are_scanwise_rigid(method):
    scans, traj, manifest = load_d1_scan_zip(os.environ['SPADE_D1_RAW_ZIP'])
    a = accumulate_glim(scans, traj, target_timestamp_ns=int(manifest['lidar_anchor_timestamp_ns']))
    ref = load_archived_refined_xyz(os.environ['SPADE_D1_REFERENCE_ZIP'], method)
    recovered = recover_reference_scan_corrections(a, ref, method=method, anchor_scan_id=5)
    assert recovered.max_residual_m < 2e-5  # micrometer-scale float rounding
    assert recovered.scan_ids.tolist() == list(range(10))
    assert len(ref) == len(a.xyz) == 215504