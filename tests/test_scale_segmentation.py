"""Synthetic tests for segmentation, common pairs and multi-scale Hand-Eye."""
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spade_calib.selection.trajectory import MotionPairs
from spade_calib.selection.scale_segmentation import (
    ScaleSegmentationSettings, ScalePartition, adaptive_partition,
    fixed_partition, estimate_pair_scales, shared_pair_mask,
)
from spade_calib.handeye.multiscale import fit_segmented_motion_pairs
from spade_calib.handeye.solver import HandEyeSettings


def synthetic():
    rng = np.random.default_rng(42)
    n = 200
    i = np.arange(n - 10)
    j = i + 10
    X = np.eye(4)
    X[:3, :3] = Rotation.from_rotvec([.15, -.08, .2]).as_matrix()
    X[:3, 3] = [.2, -.15, .3]
    R = X[:3, :3]
    Ra = Rotation.from_rotvec(rng.normal(0, .18, (len(i), 3))).as_matrix()
    Rb = R @ Ra @ R.T
    tb = rng.normal(0, .08, (len(i), 3)) + [.3, .2, .1]
    s = np.where(i < 100, 5., 12.)
    ta = (np.einsum("nij,j->ni", Rb - np.eye(3), X[:3, 3]) + tb * s[:, None]) @ R
    pairs = MotionPairs(Ra, ta, Rb, tb, np.column_stack((i, j)), i.astype(np.int64))
    return pairs, X, n


def test_adaptive_recovers_step_and_excludes_crossing_pairs():
    pairs, X, n = synthetic()
    cfg = ScaleSegmentationSettings(min_segment_frames=40, max_segments=4)
    part, candidates = adaptive_partition(pairs, n, X, cfg)
    assert part.boundaries == (0, 100, 200)
    assert len(candidates) >= 2
    g = part.groups(pairs)
    assert np.all(g[90:100] == -1)
    assert np.all(g[:90] == 0)
    assert np.all(g[100:] == 1)


def test_multiscale_recovers_shared_extrinsic_and_scales():
    pairs, X, n = synthetic()
    initial = X.copy()
    initial[:3, :3] = Rotation.from_rotvec([.02, -.01, .01]).as_matrix() @ X[:3, :3]
    initial[:3, 3] += [.03, -.02, .04]
    result = fit_segmented_motion_pairs(pairs, fixed_partition(n, 2), initial)
    np.testing.assert_allclose(result.T_camera_lidar, X, atol=1e-7)
    np.testing.assert_allclose(result.scales, [5., 12.], atol=1e-7)
    assert result.diagnostics["full_rank"]
    assert result.diagnostics["parameter_count"] == 8
    report = json.loads(json.dumps(result.to_dict()))
    assert report["segments"][1]["stop_index_exclusive"] == 200


def test_same_pair_comparison_and_fixed_extrinsic():
    pairs, X, n = synthetic()
    a, b = fixed_partition(n, 2), fixed_partition(n, 3)
    ev = estimate_pair_scales(pairs, X)
    mask = shared_pair_mask(pairs, ev, a, b)
    for p in (ScalePartition((0, n)), a, b):
        result = fit_segmented_motion_pairs(pairs, p, X, pair_mask=mask,
                                            fix_rotation=True, fix_translation=True)
        np.testing.assert_array_equal(result.pair_mask, mask)
        np.testing.assert_allclose(result.T_camera_lidar, X)
        assert result.diagnostics["parameter_count"] == p.count


def test_invalid_inputs_and_empty_segments():
    pairs, X, n = synthetic()
    with pytest.raises(ValueError):
        ScalePartition((0, 100, 100, n))
    with pytest.raises(ValueError):
        fit_segmented_motion_pairs(pairs, ScalePartition((0, 5, n)), X)
    with pytest.raises(ValueError):
        estimate_pair_scales(pairs, np.zeros((4, 4)))
    with pytest.raises(ValueError):
        fit_segmented_motion_pairs(pairs, fixed_partition(n, 2), X,
                                   pair_mask=np.zeros(pairs.count, dtype=bool))
    with pytest.raises(ValueError):
        adaptive_partition(pairs, n, X, ScaleSegmentationSettings(min_segment_frames=201))


def test_degenerate_motion_is_reported_not_hidden():
    pairs, X, n = synthetic()
    eye = np.broadcast_to(np.eye(3), pairs.Ra.shape).copy()
    scales = np.where(pairs.indices[:, 0] < 100, 5., 12.)
    ta = (pairs.tb * scales[:, None]) @ X[:3, :3]
    p = MotionPairs(eye, ta, eye, pairs.tb, pairs.indices, pairs.starts_ns)
    result = fit_segmented_motion_pairs(p, fixed_partition(n, 2), X,
                                        HandEyeSettings(robust_loss="linear"))
    assert not result.diagnostics["full_rank"]
    assert result.diagnostics["normalized_jacobian_rank"] < 8
