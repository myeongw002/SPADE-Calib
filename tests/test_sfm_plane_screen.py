"""Plane-normal scene selection regression tests, no COLMAP executable required."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spade_calib.selection.sfm_planes import (
    ImageGeometry, PlaneObservation, PlanePoint, PlaneScreenSettings,
    screen_window_planes,
)


def _points(orientations: tuple[str, ...], observations_per_point: int = 3,
            outside: bool = False) -> tuple[list[PlanePoint], dict[int, ImageGeometry]]:
    """Three spatially separated orthogonal planes with known normals."""
    rng = np.random.default_rng(20261116)
    ims = {i: ImageGeometry(1000, 800, np.eye(3)) for i in range(3)}
    points = []
    for orientation in orientations:
        uv = rng.uniform(-2., 2., size=(230, 2))
        noise = rng.normal(scale=.004, size=230)
        if orientation == 'z':
            xyz = np.column_stack((uv[:, 0], uv[:, 1], noise + 1.))
        elif orientation == 'x':
            xyz = np.column_stack((noise + 8., uv[:, 0], uv[:, 1] + 2.))
        elif orientation == 'y':
            xyz = np.column_stack((uv[:, 0], noise + 8., uv[:, 1] + 2.))
        else:
            raise ValueError(orientation)
        for p in xyz:
            if orientation == 'z':
                xx, yy = p[0], p[1]
            elif orientation == 'x':
                xx, yy = p[1], p[2] - 2
            else:
                xx, yy = p[0], p[2] - 2
            xy = (500 + xx * 125, 400 + yy * 95)
            obs = tuple(PlaneObservation(i, xy[0] + i * 2, xy[1] + i * 3)
                        for i in range(observations_per_point))
            points.append(PlanePoint(np.asarray(p), obs, .2, 3.0, outside))
    return points, ims


def _settings(**kwargs) -> PlaneScreenSettings:
    return replace(PlaneScreenSettings(
        threshold_m=.045,
        min_inliers=55,
        max_planes=5,
        max_points=1200,
        ransac_iters=330,
        min_image_observations=8,
        min_visible_frames=2,
        min_width_ratio=.07,
        seed=20261116,
    ), **kwargs)


def test_three_orthogonal_planes_have_normal_rank_three():
    points, images = _points(('x', 'y', 'z'))
    report, planes = screen_window_planes(points, images, _settings())
    assert report['plane_count'] >= 3, report
    assert report['plane_normal_rank'] == 3, report
    assert report['plane_translation_normal_proxy_sufficient'] is True
    assert report['plane_normal_min_eigen_fraction'] > .02
    assert all(p['inlier_count'] >= 55 for p in planes)
    assert all(p['visible_image_count'] >= 2 for p in planes)
    assert all(p['plane_fit_rmse_m'] < .045 for p in planes)
    assert report['plane_criterion'] == (
        'world_normal_diversity_translation_proxy_not_6d_calibration')


def test_single_plane_can_gain_logdet_but_cannot_pass_sufficiency():
    points, images = _points(('x',))
    report, planes = screen_window_planes(points, images, _settings())
    assert len(planes) >= 1
    assert report['plane_score_heuristic'] > 0
    assert report['plane_normal_rank'] <= 1
    assert report['plane_translation_normal_proxy_sufficient'] is False
    assert report['plane_normal_min_eigen_fraction'] < 1e-3


def test_unobserved_plane_is_not_qualified_even_if_xyz_is_dense():
    points, images = _points(('x', 'y', 'z'), observations_per_point=1)
    report, planes = screen_window_planes(points, images, _settings())
    assert report['plane_count'] == 0
    assert report['plane_translation_normal_proxy_sufficient'] is False
    assert planes == []


def test_global_external_support_is_reported_not_mistaken_for_local_retriangulation():
    points, images = _points(('x', 'y', 'z'), outside=True)
    report, planes = screen_window_planes(points, images, _settings())
    assert report['plane_count'] >= 3
    assert all(p['window_only_triangulation'] is False for p in planes)
    assert all(p['candidate_points_with_outside_observations'] == p['inlier_count']
               for p in planes)
    assert report['plane_external_xyz_point_count'] > 0


def test_normal_rank_is_invariant_under_global_frame_rotation():
    points, images = _points(('x', 'y', 'z'))
    transform = Rotation.from_euler('xyz', [15, -8, 20], degrees=True).as_matrix()
    rotated_points = [replace(point, xyz=transform @ point.xyz) for point in points]
    rotated_images = {
        key: replace(img, R_cw=img.R_cw @ transform.T)
        for key, img in images.items()
    }
    baseline, _ = screen_window_planes(points, images, _settings())
    rotated, _ = screen_window_planes(rotated_points, rotated_images, _settings())
    assert baseline['plane_normal_rank'] == rotated['plane_normal_rank'] == 3
    np.testing.assert_allclose(
        baseline['plane_normal_eigenvalues'],
        rotated['plane_normal_eigenvalues'], atol=.005)


@pytest.mark.parametrize('kwargs', [
    {'threshold_m': -1.},
    {'min_inliers': 2},
    {'max_points': 1},
    {'min_normal_rank': 4},
    {'min_normal_eigen_fraction': .9},
])
def test_invalid_screening_config(kwargs):
    with pytest.raises(ValueError):
        replace(_settings(), **kwargs)
