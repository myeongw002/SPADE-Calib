"""Opt-in real D1 pointwise reproducibility; not run on public CI."""
from __future__ import annotations
import os
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import pytest
from spade_calib.lidar import load_d1_scan_zip, accumulate_glim, solve_joint, solve_pgo

RAW = os.getenv('SPADE_D1_RAW_ZIP')
CAL = os.getenv('SPADE_D1_REFINEMENT_REFERENCE_ZIP')


@pytest.mark.skipif(not RAW or not CAL, reason='D1 raw and reference ZIPs not configured')
@pytest.mark.parametrize('method', ['Joint', 'PGO'])
def test_d1_joint_pgo_numeric_refinement_from_raw(method):
    scans, trajectory, manifest = load_d1_scan_zip(RAW)
    cloud = accumulate_glim(scans, trajectory, target_timestamp_ns=int(manifest['lidar_anchor_timestamp_ns']))
    assert cloud.xyz.shape == (215504, 3)
    output = solve_joint(cloud) if method == 'Joint' else solve_pgo(cloud)
    assert output.xyz.shape == cloud.xyz.shape
    assert output.iterations[0].get('pairs', output.iterations[0].get('pair_edges')) == 22
    assert np.allclose(output.transforms[5], np.eye(4), atol=1e-12)
    with ZipFile(CAL) as z:
        name = next(n for n in z.namelist() if n.endswith(f'/inputs/{method}_xyz.npy'))
        ref = np.load(BytesIO(z.read(name)), allow_pickle=False)
    residual = np.linalg.norm(output.xyz.astype(float) - ref.astype(float), axis=1)
    # Threshold chosen for reproducing historical software output, not physical accuracy.
    assert np.max(residual) < .001
    assert np.percentile(residual, 95) < .0002