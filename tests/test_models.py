"""Data-free geometry tests for the initial interoperability models."""

import numpy as np
import pytest

from spade_calib.io import CameraModel, Plane, PointCloud, RigidTransform


def test_transform_inverse_round_trip() -> None:
    transform = RigidTransform.from_rt(
        np.eye(3), np.array([0.2, -0.1, 0.4]), "camera", "lidar"
    )
    points = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, -1.0]])
    np.testing.assert_allclose(transform.inverse().apply(transform.apply(points)), points)


def test_transform_rejects_reflection() -> None:
    reflection = np.diag([-1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="SO"):
        RigidTransform.from_rt(reflection, np.zeros(3), "camera", "lidar")


def test_cloud_provenance_and_shape() -> None:
    cloud = PointCloud(np.zeros((2, 3)), "lidar_reference", scan_id=np.array([1, 2]))
    assert cloud.point_count == 2
    assert cloud.source_point_id is None
    with pytest.raises(ValueError, match="length-N"):
        PointCloud(np.zeros((2, 3)), "lidar_reference", scan_id=np.array([1]))


def test_plane_normalization_keeps_equation() -> None:
    plane = Plane(np.array([0, 0, 2.0]), -4.0, "camera")
    np.testing.assert_allclose(plane.normal, [0.0, 0.0, 1.0])
    assert plane.offset_m == pytest.approx(-2.0)


def test_camera_model_validation() -> None:
    camera = CameraModel(
        width=1280, height=720, K=np.diag([600.0, 600.0, 1.0]),
        distortion=np.zeros(5),
    )
    assert camera.frame_id == "camera"
    with pytest.raises(ValueError, match="focal"):
        CameraModel(1280, 720, np.eye(3) * -1, np.zeros(5))
