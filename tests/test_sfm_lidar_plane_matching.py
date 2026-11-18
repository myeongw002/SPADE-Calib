"""Deterministic synthetic tests for finite patches, matching and joint refinement."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spade_calib.correspondence.sfm_lidar_planes import (
    LidarPatch, MatchSettings, PatchSettings, SfMPlaneMatches,
    extract_lidar_patches, match_sfm_to_patches,
)
from spade_calib.calibration.sfm_plane_solver import (
    SfMPlaneRefinementSettings, point_plane_residual, refine_joint_7dof,
)
from spade_calib.selection.trajectory import MotionPairs


def _patch(i, points, normal, offset):
    return LidarPatch(i, np.asarray(normal, float), float(offset),
                      np.asarray(points, float), 0.01, np.array([0, 1]))


def test_finite_patch_rejects_distant_coplanar_surface():
    grid = np.array([(x, y) for x in np.linspace(-1., 1., 10)
                     for y in np.linspace(-1., 1., 10)])
    plane_z = np.column_stack((grid, np.full(len(grid), 5.)))
    wall = np.column_stack((np.full(len(grid), 7.),
                            grid[:, 0] + 5., grid[:, 1] + 5.))
    patches = [_patch(0, plane_z, [0, 0, 1], -5.),
               _patch(1, wall, [1, 0, 0], -7.)]
    # Last point lies on z=5 infinite plane but outside its observed support.
    points = np.vstack((plane_z[:15], wall[:15], [[20, 20, 5]]))
    xy = np.zeros((len(points), 2))
    matches, report = match_sfm_to_patches(
        points, xy, patches, np.eye(4), 1.,
        MatchSettings(max_normal_distance_m=0.3, max_support_distance_m=0.8,
                      min_matches_per_patch=4))
    assert matches is not None
    assert matches.count == 30
    assert 30 not in matches.feature_index
    assert report['matched_patch_count'] == 2
    assert set(matches.patch_id.tolist()) == {0, 1}


def test_synthetic_lidar_patch_detection():
    rng = np.random.default_rng(91)
    uv = np.array([(a, b) for a in np.linspace(-2, 2, 25)
                   for b in np.linspace(-2, 2, 25)])
    ground = np.column_stack((uv, np.zeros(len(uv))))
    wall = np.column_stack((np.full(len(uv), 5.), uv))
    xyz = np.vstack([ground, wall]) + rng.normal(0, 0.008, (2*len(uv), 3))
    patches, report = extract_lidar_patches(
        xyz, np.r_[np.zeros(len(uv)), np.ones(len(uv))].astype(int),
        PatchSettings(voxel_size_m=0.06, max_sampled_points=5000, max_planes=4,
                      ransac_iterations=140, min_inliers=100,
                      distance_threshold_m=0.045, connectivity_radius_m=0.37))
    assert len(patches) >= 2, report
    assert any(abs(p.normal @ [0, 0, 1]) > 0.98 for p in patches)
    assert any(abs(p.normal @ [1, 0, 0]) > 0.98 for p in patches)
    assert all(len(p.support_xyz_lidar) >= 100 for p in patches)


def _synthetic_motion_pairs(T, scale, seed=11):
    rng = np.random.default_rng(seed)
    R = T[:3, :3]
    t = T[:3, 3]
    RA, TA, RB, TB = [], [], [], []
    for _ in range(35):
        a = Rotation.from_rotvec(rng.normal(0, 0.11, 3)).as_matrix()
        shift = rng.normal(0, 0.6, 3)
        b = R @ a @ R.T
        bt = (R @ shift - (b-np.eye(3)) @ t) / scale
        RA.append(a)
        TA.append(shift)
        RB.append(b)
        TB.append(bt)
    n = len(RA)
    return MotionPairs(np.stack(RA), np.stack(TA), np.stack(RB), np.stack(TB),
                       np.column_stack([np.arange(n), np.arange(n)+1]),
                       np.arange(n, dtype=np.int64) * 1000000000)


def test_joint_solver_recovers_7dof_with_correct_correspondences():
    rng = np.random.default_rng(2)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', [2, -6, 8], degrees=True).as_matrix()
    T[:3, 3] = [0.3, -0.1, 0.22]
    scale = 13.
    points_lidar = []
    normals = []
    offsets = []
    ids = []
    for i, (normal, off) in enumerate((([1., 0, 0], -5.),
                                       ([0., 1, 0], -3.),
                                       ([0., 0, 1], -7.))):
        for _ in range(70):
            p = rng.uniform(-2, 2, 3)
            p[i] = -off
            points_lidar.append(p)
            normals.append(normal)
            offsets.append(off)
            ids.append(i)
    p_l = np.asarray(points_lidar)
    p_sfm = (p_l @ T[:3, :3].T + T[:3, 3]) / scale
    m = SfMPlaneMatches(p_sfm, np.zeros((len(p_l), 2)), np.asarray(normals, float),
                        np.asarray(offsets, float), np.asarray(ids, np.int32),
                        np.arange(len(p_l), dtype=np.int64))
    m.validate()
    T0 = T.copy()
    T0[:3, :3] = Rotation.from_rotvec([0.03, -0.02, 0.018]).as_matrix() @ T[:3, :3]
    T0[:3, 3] += [0.2, -0.25, 0.12]
    s0 = scale * 1.08
    pairs = _synthetic_motion_pairs(T, scale)
    result = refine_joint_7dof(T0, s0, pairs, m,
                               SfMPlaneRefinementSettings(robust_loss='linear'))
    estimated = np.array(result['T_camera_lidar'])
    angle_deg = np.rad2deg(Rotation.from_matrix(
        estimated[:3, :3] @ T[:3, :3].T).magnitude())
    assert result['optimizer']['success']
    assert result['local_observability']['full_rank']
    assert angle_deg < 0.001
    assert np.linalg.norm(estimated[:3, 3] - T[:3, 3]) < 0.001
    assert abs(result['sfm_scale_to_metric']/scale - 1) < 1e-5
    assert result['optimizer']['plane_median_abs_m_final'] < 1e-5


def test_rejects_duplicate_feature_matches():
    points = np.tile([1, 2, 3], (4, 1)).astype(float)
    m = SfMPlaneMatches(points, np.zeros((4, 2)),
                        np.tile([0, 0, 1], (4, 1)), np.zeros(4),
                        np.zeros(4, np.int32), np.array([0, 1, 1, 3]))
    with pytest.raises(ValueError, match='at most one'):
        m.validate()