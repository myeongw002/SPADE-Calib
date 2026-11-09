"""Configuration, provenance and read-only manifest tests."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import yaml

from spade_calib.cli import main
from spade_calib.io.dataset import inspect_dataset, load_dataset_config, sha256_file


def make_fixture(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    np.save(root / "Joint_xyz.npy", np.array([[1, 2, 3]], dtype=np.float32))
    np.save(root / "metric.npy", np.array([[3.0, np.nan], [1.0, 2.0]]))
    transform = np.eye(4)
    np.savetxt(root / "T_handeye_only.txt", transform, delimiter=",")
    document = {
        "schema_version": 1,
        "dataset": {"name": "d1_synthetic", "root": "inputs"},
        "frames": {"lidar": "lidar_reference", "camera": "camera"},
        "inputs": {
            "joint_cloud": "Joint_xyz.npy",
            "metric_depth": "metric.npy",
            "handeye": "T_handeye_only.txt",
        },
    }
    config = tmp_path / "dataset.yaml"
    config.write_text(yaml.safe_dump(document), encoding="utf-8")
    return config, root


def test_config_resolution_and_inspection(tmp_path) -> None:
    config_path, root = make_fixture(tmp_path)
    config = load_dataset_config(config_path)
    assert config.require("joint_cloud") == root / "Joint_xyz.npy"
    report = inspect_dataset(config)
    assert report["dataset"] == "d1_synthetic"
    assert report["artifacts"]["joint_cloud"]["point_count"] == 1
    assert report["artifacts"]["joint_cloud"]["original_point_ids_available"] is False
    assert report["artifacts"]["metric_depth"]["invalid_pixels"] == 1
    assert report["artifacts"]["metric_depth"]["depth_semantics"] == "camera_z_m"
    assert report["artifacts"]["handeye"]["transform"] == "T_camera_lidar_reference"


def test_sha256_matches_known_content(tmp_path) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"hello")
    assert sha256_file(path) == hashlib.sha256(b"hello").hexdigest()


def test_unknown_config_input_rejected(tmp_path) -> None:
    config_path, _ = make_fixture(tmp_path)
    document = yaml.safe_load(config_path.read_text())
    document["inputs"]["unknown_feature"] = "bogus.txt"
    config_path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="Unsupported legacy input"):
        load_dataset_config(config_path)


def test_missing_input_is_error(tmp_path) -> None:
    config_path, root = make_fixture(tmp_path)
    (root / "Joint_xyz.npy").unlink()
    config = load_dataset_config(config_path)
    with pytest.raises(FileNotFoundError, match="joint_cloud"):
        inspect_dataset(config)


def test_cli_outputs_json_manifest(tmp_path) -> None:
    config_path, _ = make_fixture(tmp_path)
    output = tmp_path / "manifest.json"
    status = main(["inspect", "--config", str(config_path), "--output", str(output)])
    assert status == 0
    manifest = json.loads(output.read_text())
    assert manifest["artifacts"]["metric_depth"]["finite_positive_pixels"] == 3
    assert len(manifest["artifacts"]["joint_cloud"]["sha256"]) == 64


def test_cli_reports_bad_config_without_traceback(tmp_path, capsys) -> None:
    status = main(["inspect", "--config", str(tmp_path / "absent.yaml")])
    assert status == 2
    assert "spade-calib:" in capsys.readouterr().err
