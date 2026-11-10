"""Independent solver synthetic smoke tests (no reference PCD needed)."""
import numpy as np
import pytest
from spade_calib.lidar import AccumulatedCloud, JointSettings, PGOSettings, solve_joint, solve_pgo


def synthetic(n_per_plane=450):
    rng = np.random.default_rng(7)
    # Three orthogonal planes give a locally observable 6-DoF problem.
    v = rng.uniform(-2, 2, (n_per_plane, 2))
    p = np.concatenate([
        np.column_stack([v[:, 0], v[:, 1], np.zeros(len(v))]),
        np.column_stack([np.zeros(len(v)) + 3., v[:, 0], v[:, 1]]),
        np.column_stack([v[:, 0], np.zeros(len(v)) - 3., v[:, 1]]),
    ])
    p += rng.normal(scale=.002, size=p.shape)
    # Each scan sees the same geometry under a different small rigid bias.
    src = np.concatenate([p, p + [0.025, -.02, .018], p + [-.02, .025, -.015]]).astype(np.float32)
    n = len(p)
    cloud = AccumulatedCloud(src, np.repeat(np.arange(3, dtype=np.uint16), n),
                             np.zeros(3*n, dtype=np.float32), np.zeros(3*n, dtype=np.uint16),
                             np.arange(3*n, dtype=np.int64), 10_000_000_000)
    return cloud


@pytest.mark.parametrize('method', ['Joint', 'PGO'])
def test_refine_without_saved_reference_and_preserve_anchor(method):
    cloud = synthetic()
    settings = JointSettings(anchor_scan_id=0, d1_roi=False, iterations=3,
                             max_samples_per_scan=3000)
    result = (solve_joint(cloud, settings) if method == 'Joint'
              else solve_pgo(cloud, PGOSettings(pairs=settings)))
    assert result.method == method
    assert result.xyz.shape == cloud.xyz.shape
    assert np.isfinite(result.xyz).all()
    assert result.transforms.shape == (3,4,4)
    np.testing.assert_array_equal(result.xyz[cloud.scan_id == 0], cloud.xyz[cloud.scan_id == 0])
    np.testing.assert_allclose(result.transforms[0], np.eye(4), atol=1e-12)
    assert len(result.iterations) >= 1
    # Pairwise discrepancy decreases on the exact shared synthetic surfaces.
    n=len(cloud.xyz)//3
    initial = np.mean(np.linalg.norm(cloud.xyz[n:2*n].astype(float)-cloud.xyz[:n], axis=1))
    final = np.mean(np.linalg.norm(result.xyz[n:2*n].astype(float)-result.xyz[:n], axis=1))
    assert final < initial, (method, initial, final)


def test_only_two_methods():
    from spade_calib.lidar import SUPPORTED_METHODS
    assert SUPPORTED_METHODS == ('Joint', 'PGO')
    with pytest.raises(ValueError):
        JointSettings(iterations=0)
    with pytest.raises(ValueError):
        PGOSettings(iterations=0)