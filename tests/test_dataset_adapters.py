"""Regression tests for KITTI frame joins and custom timestamp metadata."""
from __future__ import annotations

import csv

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from spade_calib.io.colmap import RegisteredImage
from spade_calib.io.kitti import (
    frame_id_from_name, kitti_timestamp_map, load_kitti_trajectory_data,
    read_kitti_times_ns,
)
from spade_calib.lidar.trajectory import GlimTrajectory
from spade_calib.selection.trajectory import match_trajectories
from spade_calib.handeye.solver import fit_full_trajectory


def _synthetic_kitti(tmp_path, *, n=211):
    seconds = np.arange(n, dtype=np.float64) * 0.1038
    times_file = tmp_path / "times.txt"
    times_file.write_text("".join(f"{t:.9f}\n" for t in seconds))
    # Deliberately *synthetic* GLIM time stamps, not KITTI timestamps.
    t = np.arange(n, dtype=float) * 0.1
    lr = Rotation.from_euler("xyz", np.column_stack((
        0.21 * np.sin(0.2 * t),
        0.29 * np.sin(0.18 * t),
        0.24 * np.cos(0.25 * t)))).as_matrix()
    lp = np.column_stack((0.35 * t, 0.8 * np.sin(0.3 * t),
                          0.5 * np.cos(0.33 * t)))
    true_r = Rotation.from_euler("xyz", [0.13, -0.1, 0.17]).as_matrix()
    true_t = np.array([0.12, -0.06, 0.19])
    true_s = 8.2
    cr = lr @ true_r
    cp = (lp + np.einsum("nij,j->ni", lr, true_t)) / true_s + [2.0, -1.0, 0.5]
    lidar = np.column_stack((1.0 + t, lp, Rotation.from_matrix(lr).as_quat()))
    glim_path = tmp_path / "odom_lidar.txt"
    np.savetxt(glim_path, lidar, fmt="%.12f")
    csv_path = tmp_path / "camera_poses.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id", "image_name", "camera_id",
                         "qw", "qx", "qy", "qz", "tx", "ty", "tz"])
        # Represent unregistered camera images by absent frame IDs.
        for i in range(n):
            if i % 17 == 0 and i not in (0, n - 1):
                continue
            Rcw = cr[i].T
            tcw = -Rcw @ cp[i]
            q = Rotation.from_matrix(Rcw).as_quat()
            writer.writerow([i + 1, f"{i:06d}.png", 1, q[3], *q[:3], *tcw])
    return glim_path, csv_path, times_file, true_r, true_t, true_s


def test_kitti_adapter_matches_by_frame_despite_independent_pose_time(tmp_path):
    glim, csv_path, times, R, t, scale = _synthetic_kitti(tmp_path)
    matched = load_kitti_trajectory_data(glim, csv_path, times)
    assert matched.total_registered == 199  # 211 frames, 12 omitted
    assert len(matched.timestamps_ns) == matched.total_registered
    assert matched.timestamps_ns[0] == 0
    assert matched.timestamps_ns[1] == 103_800_000
    # Both sources are correctly registered by *frame*, not synthetic time.
    first = matched.image_names.index("000001.png")
    assert np.linalg.norm(matched.lidar_t_world_m[first]) > 0
    fit = fit_full_trajectory(matched)
    assert fit.diagnostics["converged"]
    assert Rotation.from_matrix(fit.rotation_lidar_camera @ R.T).magnitude() < 1e-5
    np.testing.assert_allclose(fit.translation_lidar_camera_m, t, atol=1e-4)
    assert fit.sfm_scale_to_metric == pytest.approx(scale, rel=1e-5)


def test_kitti_legacy_tum_only_for_dense_frame_export(tmp_path):
    glim, _, times, _, _, _ = _synthetic_kitti(tmp_path, n=25)
    raw = np.loadtxt(glim)
    camera_tum = tmp_path / "camera_trajectory_tum.txt"
    np.savetxt(camera_tum, raw)  # Here camera data are intentionally arbitrary.
    matched = load_kitti_trajectory_data(glim, camera_tum, times)
    assert len(matched.timestamps_ns) == 25
    np.testing.assert_allclose(matched.camera_t_world_sfm[0], raw[0, 1:4])
    assert matched.image_names[12] == "000012.png"


def test_kitti_times_and_frame_validation(tmp_path):
    file = tmp_path / "times.txt"
    file.write_text("0\n0.1038752\n0.2078941\n")
    assert read_kitti_times_ns(file).tolist() == [0, 103875200, 207894100]
    assert frame_id_from_name("image_2/000001.png") == 1
    assert kitti_timestamp_map(file, ["000000.png", "000002.png"]) == {
        "000000.png": 0, "000002.png": 207894100,
    }
    with pytest.raises(ValueError, match="Duplicate"):
        kitti_timestamp_map(file, ["image_2/000001.png", "image_3/000001.png"])
    with pytest.raises(ValueError, match="outside"):
        kitti_timestamp_map(file, ["000003.png"])
    file.write_text("0\n0.2\n0.2\n")
    with pytest.raises(ValueError, match="not increasing"):
        read_kitti_times_ns(file)


def test_custom_timestamp_map_allows_non_numeric_colmap_names():
    base = 1_791_439_637_016_960_512
    n = 10
    # GLIM epochs should be parsed in seconds, with 10Hz pose timestamps.
    stamps = base + np.arange(n, dtype=np.int64) * 100_000_000
    poses = np.column_stack((stamps * 1e-9,
                             np.zeros((n, 3)),
                             np.tile([0, 0, 0, 1.0], (n, 1))))
    glim = GlimTrajectory.from_array(poses)
    camera = [
        RegisteredImage(i + 1, f"camera/frame_{i}.png", 1, (1, 0, 0, 0), (0, 0, 0))
        for i in range(n)
    ]
    table = {rec.name: int(stamps[i]) for i, rec in enumerate(camera)}
    result = match_trajectories(glim, camera, image_timestamp_map=table)
    assert len(result.timestamps_ns) >= 8
    assert all(name.startswith("camera/frame_") for name in result.image_names)
    table.pop(camera[5].name)
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        match_trajectories(glim, camera, image_timestamp_map=table)


def test_kitti_rejects_invalid_glim_start_frame(tmp_path):
    glim, csv_path, times, *_ = _synthetic_kitti(tmp_path, n=25)
    with pytest.raises(ValueError, match="nonnegative"):
        load_kitti_trajectory_data(glim, csv_path, times, glim_first_frame=-1)
    with pytest.raises(ValueError, match="exceeds"):
        load_kitti_trajectory_data(glim, csv_path, times, glim_first_frame=1)
