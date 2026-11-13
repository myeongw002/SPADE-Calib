"""KITTI 4xfloat32 scans: no fabricated per-return timestamps or ring IDs."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import yaml

from spade_calib.cli import main
from spade_calib.io.kitti import accumulate_kitti_rigid_window
from spade_calib.io.kitti_bin import load_kitti_bin_scan
from spade_calib.io.recording import load_recording_config, index_recording
from spade_calib.lidar import accumulate_glim, GlimTrajectory, filter_cloud


def _write_bin(path, rows):
    np.asarray(rows, dtype='<f4').reshape(-1, 4).tofile(path)


def _make_indexed_fixture(tmp_path):
    times = tmp_path / "times.txt"
    times.write_text("0.0\n0.10384\n0.20768\n")
    scans = tmp_path / "velodyne"
    scans.mkdir()
    _write_bin(scans / "000000.bin", [[5, 0, 0, .25], [0, 0, 0, .0]])
    _write_bin(scans / "000001.bin", [[4, 0, 0, .75]])
    _write_bin(scans / "000002.bin", [[3, 0, 0, .95]])
    # GLIM has synthetic 0.1s timestamps but pose ROWS align to KITTI frames.
    traj = tmp_path / "odom_lidar.txt"
    np.savetxt(traj, [[1.0, 0, 0, 0, 0, 0, 0, 1],
                      [1.1, 1, 0, 0, 0, 0, 0, 1],
                      [1.2, 2, 0, 0, 0, 0, 0, 1]], fmt="%.10f")
    return scans, times, traj


def test_bin_reader_preserves_reflectance_and_original_indices(tmp_path):
    p = tmp_path / "000000.bin"
    _write_bin(p, [[2, 0, 0, .25], [np.nan, 0, 0, .1],
                   [0.1, 0, 0, .3], [3, 4, 0, .9],
                   [1000, 0, 0, .6], [2.1, 0, 0, np.inf]])
    scan = load_kitti_bin_scan(p, scan_id=4, timestamp_ns=103840000,
                               min_range_m=.5, max_range_m=10)
    assert scan.scan_id == 4 and scan.timestamp_ns == 103840000
    assert scan.time_offset_s is None and scan.ring is None
    assert not scan.has_per_return_time and not scan.has_ring
    np.testing.assert_array_equal(scan.source_return_index, [0, 3])
    np.testing.assert_allclose(scan.xyz, [[2, 0, 0], [3, 4, 0]])
    np.testing.assert_allclose(scan.intensity, [.25, .9], rtol=1e-6)


def test_bin_reader_rejects_bad_payload(tmp_path):
    path = tmp_path / "000000.bin"
    path.write_bytes(b"123")
    with pytest.raises(ValueError, match="16 bytes"):
        load_kitti_bin_scan(path, scan_id=0, timestamp_ns=0)
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="nonempty"):
        load_kitti_bin_scan(path, scan_id=0, timestamp_ns=0)
    _write_bin(path, [[2, 0, 0, 1]])
    with pytest.raises(ValueError, match="range"):
        load_kitti_bin_scan(path, scan_id=0, timestamp_ns=0, min_range_m=10, max_range_m=2)
    with pytest.raises(ValueError, match="timestamp_ns"):
        load_kitti_bin_scan(path, scan_id=0, timestamp_ns=-1)
    with pytest.raises(ValueError, match="scan_id"):
        load_kitti_bin_scan(path, scan_id=65536, timestamp_ns=0)


def test_kitti_rigid_accumulation_uses_pose_row_not_fake_timestamps(tmp_path):
    scans, times, odom = _make_indexed_fixture(tmp_path)
    result = accumulate_kitti_rigid_window(
        scans, odom, times, [0, 1, 2], target_frame_id=0)
    assert result.motion_model == "rigid"
    assert result.time_offset_s is None
    assert result.ring is None
    assert result.frame_id == "kitti_lidar_frame_0"
    assert result.reference_timestamp_ns == 0
    np.testing.assert_allclose(result.xyz, [[5, 0, 0]] * 3, atol=1e-6)
    np.testing.assert_array_equal(result.scan_id, [0, 1, 2])
    np.testing.assert_array_equal(result.source_return_index, [0, 0, 0])
    np.testing.assert_allclose(result.intensity, [.25, .75, .95], atol=1e-6)
    mask = np.array([True, False, True])
    filtered = filter_cloud(result, mask)
    assert filtered.motion_model == 'rigid'
    assert filtered.time_offset_s is None and filtered.ring is None
    np.testing.assert_allclose(filtered.intensity, [.25, .95], atol=1e-6)


def test_kitti_rigid_window_rejects_misaligned_frames(tmp_path):
    scans, times, odom = _make_indexed_fixture(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        accumulate_kitti_rigid_window(scans, odom, times, [0, 3],
                                      target_frame_id=0)
    with pytest.raises(ValueError, match="increasing"):
        accumulate_kitti_rigid_window(scans, odom, times, [1, 0],
                                      target_frame_id=0)
    with pytest.raises(ValueError, match="Duplicate|unique"):
        accumulate_kitti_rigid_window(scans, odom, times, [1, 1],
                                      target_frame_id=0)
    (scans / "000001.bin").unlink()
    with pytest.raises(ValueError, match="Missing KITTI"):
        accumulate_kitti_rigid_window(scans, odom, times, [0, 1],
                                      target_frame_id=0)


def test_explicit_non_deskew_mode_rejects_untimed_scan(tmp_path):
    path = tmp_path / "000000.bin"
    _write_bin(path, [[5, 0, 0, .5]])
    scan = load_kitti_bin_scan(path, scan_id=0, timestamp_ns=11_000_000_000)
    traj = GlimTrajectory.from_array(np.array([
        [10.0, 0, 0, 0, 0, 0, 0, 1], [12.0, 2, 0, 0, 0, 0, 0, 1]], dtype=float))
    with pytest.raises(ValueError, match="Cannot deskew"):
        accumulate_glim([scan], traj, target_timestamp_ns=11_000_000_000)
    result = accumulate_glim(
        [scan], traj, target_timestamp_ns=11_000_000_000, mode="rigid")
    assert result.motion_model == "rigid"
    assert result.time_offset_s is None and result.ring is None
    np.testing.assert_allclose(result.xyz, [[5, 0, 0]], atol=1e-6)
    np.testing.assert_allclose(result.intensity, [.5], atol=1e-6)


def test_kitti_accumulate_cli_npz_without_fake_point_metadata(tmp_path):
    scans, times, odom = _make_indexed_fixture(tmp_path)
    out = tmp_path / "out" / "rigid.npz"
    assert main([
        "accumulate-kitti", "--lidar-dir", str(scans), "--glim", str(odom),
        "--kitti-times", str(times),
        "--start-frame", "0", "--stop-frame", "2", "--target-frame", "0",
        "--output", str(out)]) == 0
    with np.load(out, allow_pickle=False) as archive:
        assert "time_offset_s" not in archive.files
        assert "ring" not in archive.files
        assert archive["motion_model"].item() == "rigid"
        assert not bool(archive["deskew_applied"])
        np.testing.assert_array_equal(archive["frame_ids"], [0, 1, 2])
        np.testing.assert_array_equal(archive["scan_timestamps_ns"],
                                      [0, 103840000, 207680000])
        np.testing.assert_allclose(archive["xyz"], [[5, 0, 0]] * 3, atol=1e-6)
        np.testing.assert_allclose(archive["intensity"], [.25, .75, .95], atol=1e-6)
    assert main([
        "accumulate-kitti", "--lidar-dir", str(scans), "--glim", str(odom),
        "--kitti-times", str(times), "--start-frame", "2", "--stop-frame", "1",
        "--target-frame", "0", "--output", str(out)]) == 2


def test_recording_index_loads_bin_with_explicit_timestamps(tmp_path):
    root = tmp_path / "raw"
    img = root / "images"
    scans = root / "velodyne"
    sfm = root / "sfm"
    for d in (img, scans, sfm):
        d.mkdir(parents=True)
    base = 1_800_000_000_000_000_000
    for i in (0, 1):
        (img / f"{i:06d}.png").write_bytes(b"jpeg")
        _write_bin(scans / f"{i:06d}.bin", [[3 + i, 0, 0, 0.8]])
    (sfm / "camera_poses.csv").write_text(
        "image_id,image_name,camera_id,qw,qx,qy,qz,tx,ty,tz\n"
        "1,000000.png,1,1,0,0,0,0,0,0\n"
        "2,000001.png,1,1,0,0,0,0,0,0\n")
    def write_map(name, ext):
        with (root / name).open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file", "timestamp_ns"])
            writer.writerows([[f"{i:06d}.{ext}", base+i*100_000_000] for i in (0, 1)])
    write_map("image_stamps.csv", "png")
    write_map("scan_stamps.csv", "bin")
    np.savetxt(root / "odom_lidar.txt",
               [[base*1e-9-1, 0, 0, 0, 0, 0, 0, 1],
                [base*1e-9+1, 1, 0, 0, 0, 0, 0, 1]], fmt="%.9f")
    cfg = {
        "schema_version": 2, "dataset": {"name": "test", "root": "raw"},
        "inputs": {
            "images": "images", "lidar_scans": "velodyne", "sfm": "sfm",
            "glim": "odom_lidar.txt", "image_timestamps": "image_stamps.csv",
            "lidar_timestamps": "scan_stamps.csv"},
        "pcd": {"range_m": [.5, 100]},
    }
    config = tmp_path / "recording.yaml"
    config.write_text(yaml.safe_dump(cfg))
    index = index_recording(load_recording_config(config))
    assert len(index.lidar_scans) == 2
    loaded = index.load_lidar_scan(0)
    assert loaded.time_offset_s is None and loaded.ring is None
    np.testing.assert_allclose(loaded.intensity, [.8])
    window = index.load_window_scans(index.lidar_window(base, base+100_000_000))
    assert [s.scan_id for s in window] == [0, 1]
    manifest = index.to_dict()
    assert manifest["summary"]["kitti_bin_scans_without_per_return_time"] == 2
    assert any("rigid" in x for x in manifest["warnings"])
