"""Synthetic regression tests for the recovered D1 on-disk field names."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import yaml

from spade_calib.io.legacy import (
    load_camera_planes,
    load_depth,
    load_handeye,
    load_sfm_anchors,
    load_xyz_cloud,
)


def test_xyz_loader_preserves_order_but_not_invented_provenance(tmp_path) -> None:
    path = tmp_path / "Joint_xyz.npy"
    expected = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]], dtype=np.float32)
    np.save(path, expected)
    cloud = load_xyz_cloud(path, frame_id="lidar_reference")
    np.testing.assert_array_equal(cloud.xyz, expected)
    assert cloud.frame_id == "lidar_reference"
    assert cloud.source_point_id is None
    with pytest.raises(ValueError, match="finite"):
        np.save(path, np.array([[float("nan"), 2, 3]]))
        load_xyz_cloud(path, frame_id="lidar_reference")


def test_depth_preserves_nonfinite_values_and_exposes_validity(tmp_path) -> None:
    path = tmp_path / "metric.npy"
    np.save(path, np.array([[1.0, np.nan, np.inf], [0.0, -1.0, 2.0]]))
    result = load_depth(path)
    assert result.shape == (2, 3)
    assert result.valid.sum() == 2
    assert np.isnan(result.values[0, 1])
    assert np.isposinf(result.values[0, 2])
    assert result.valid_fraction == pytest.approx(1 / 3)


def test_handeye_txt_and_yaml_equivalent(tmp_path) -> None:
    transform = np.eye(4)
    transform[:3, 3] = [0.1, 0.2, 0.3]
    txt = tmp_path / "T_handeye_only.txt"
    np.savetxt(txt, transform, delimiter=",")
    yaml_path = tmp_path / "handeye_original.yaml"
    yaml_path.write_text(
        yaml.safe_dump({
            "sfm_scale_to_metric": 9.229601422,
            "T_C_L": {
                "rotation_matrix": np.eye(3).tolist(),
                "translation_m": [0.1, 0.2, 0.3],
            },
        }),
        encoding="utf-8",
    )
    a = load_handeye(txt, source_frame="lidar", target_frame="camera")
    b = load_handeye(yaml_path, source_frame="lidar", target_frame="camera")
    np.testing.assert_allclose(a.extrinsic.matrix, b.extrinsic.matrix)
    assert a.sfm_scale is None
    assert b.sfm_scale == pytest.approx(9.229601422)


def test_camera_plane_csv_mask_contract(tmp_path) -> None:
    csv_path = tmp_path / "camera_planes.csv"
    mask_path = tmp_path / "camera_masks_320x180.npz"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id", "pixels", "plane_rmse_cm", "normal_x", "normal_y",
                "normal_z", "offset",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "id": 4, "pixels": 2, "plane_rmse_cm": 1.5, "normal_x": 0,
            "normal_y": 0, "normal_z": 1, "offset": -2,
        })
    np.savez(mask_path, plane_4=np.array([[0, 1], [1, 0]], dtype=np.uint8))
    planes = load_camera_planes(csv_path, mask_path, frame_id="camera")
    assert len(planes) == 1
    assert planes[0].plane_id == 4
    assert planes[0].fitting_rmse_m == pytest.approx(0.015)
    assert planes[0].pixel_count == 2
    np.testing.assert_array_equal(planes[0].plane.normal, [0, 0, 1])
    np.savez(mask_path, plane_4=np.ones((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="mask pixels"):
        load_camera_planes(csv_path, mask_path, frame_id="camera")


def test_sfm_anchor_legacy_field_names_and_image_grid(tmp_path) -> None:
    path = tmp_path / "d1_sfm_sparse_metric.npz"
    np.savez(
        path,
        xy=np.array([[100.0, 200.0], [50.5, 40.0]], dtype=np.float32),
        depth_m=np.array([10.0, 5.0], dtype=np.float32),
        reprojection_error_px=np.array([0.3, 4.0], dtype=np.float32),
        image_shape=np.array([720, 1280], dtype=np.int64),
        sfm_scale=np.array(9.2),
    )
    anchors = load_sfm_anchors(path)
    assert anchors.count == 2
    assert anchors.image_shape == (720, 1280)
    assert anchors.quality_mask().tolist() == [True, False]
    assert anchors.sfm_scale == pytest.approx(9.2)
