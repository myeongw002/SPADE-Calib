"""Configuration and inspection of extracted legacy D1 inputs.

No code in this module assumes a fixed sequence ID or a known local folder.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from spade_calib.io.legacy import (
    load_camera_planes,
    load_depth,
    load_handeye,
    load_sfm_anchors,
    load_xyz_cloud,
)


SUPPORTED_INPUTS = frozenset({
    "joint_cloud", "pgo_cloud", "handeye", "camera_planes", "camera_masks",
    "sfm_anchors", "raw_depth", "metric_depth",
})
CLOUD_INPUTS = frozenset({"joint_cloud", "pgo_cloud"})
DEPTH_INPUTS = frozenset({"raw_depth", "metric_depth"})


@dataclass(frozen=True)
class LegacyDatasetConfig:
    """Resolved paths and named frames for a single extracted scene."""

    name: str
    root: Path
    camera_frame: str
    lidar_frame: str
    inputs: Mapping[str, Path]

    def require(self, key: str) -> Path:
        if key not in SUPPORTED_INPUTS:
            raise KeyError(f"Unknown legacy input: {key}")
        if key not in self.inputs:
            raise KeyError(f"Input not configured: {key}")
        path = self.inputs[key]
        if not path.is_file():
            raise FileNotFoundError(f"Configured input {key} not found: {path}")
        return path


def load_dataset_config(path: str | Path) -> LegacyDatasetConfig:
    """Load an explicitly typed YAML config; paths are relative to dataset.root.

    Relative dataset roots are resolved relative to the config file directory.
    """
    config_path = Path(path).expanduser().resolve(strict=True)
    with config_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError("Dataset config must be a YAML mapping")
    if document.get("schema_version") != 1:
        raise ValueError("Dataset schema_version must be integer 1")

    dataset = document.get("dataset")
    frames = document.get("frames")
    inputs = document.get("inputs")
    if not all(isinstance(value, dict) for value in (dataset, frames, inputs)):
        raise ValueError("Config needs dataset, frames and inputs mappings")
    name = dataset.get("name")
    root_string = dataset.get("root")
    lidar_frame = frames.get("lidar")
    camera_frame = frames.get("camera")
    for value, field in (
        (name, "dataset.name"), (root_string, "dataset.root"),
        (lidar_frame, "frames.lidar"), (camera_frame, "frames.camera")
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a nonempty string")

    root = Path(root_string).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {root}")

    if not inputs:
        raise ValueError("Config inputs mapping must not be empty")
    unexpected = set(inputs) - SUPPORTED_INPUTS
    if unexpected:
        raise ValueError(f"Unsupported legacy input keys: {sorted(unexpected)}")
    if ("camera_planes" in inputs) != ("camera_masks" in inputs):
        raise ValueError("camera_planes and camera_masks must be configured together")

    resolved: dict[str, Path] = {}
    for key, input_path in inputs.items():
        if not isinstance(input_path, str) or not input_path.strip():
            raise ValueError(f"inputs.{key} must be a nonempty path string")
        candidate = Path(input_path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved[key] = candidate.resolve(strict=False)

    return LegacyDatasetConfig(
        name=name, root=root, camera_frame=camera_frame,
        lidar_frame=lidar_frame, inputs=resolved,
    )


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash file bytes in bounded memory; useful for exact-input provenance."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_dataset(config: LegacyDatasetConfig, *, include_hashes: bool = True) -> dict:
    """Inspect the configured legacy artifacts without printing private point values.

    Loading and validation are read-only. The resulting report contains only
    shapes, counts, source-file basenames, hashes and named frames.
    """
    report: dict = {
        "schema_version": 1,
        "dataset": config.name,
        "frames": {"lidar": config.lidar_frame, "camera": config.camera_frame},
        "artifacts": {},
    }
    for key in sorted(config.inputs):
        path = config.require(key)
        item: dict = {"filename": path.name, "size_bytes": path.stat().st_size}
        if include_hashes:
            item["sha256"] = sha256_file(path)
        if key in CLOUD_INPUTS:
            cloud = load_xyz_cloud(path, frame_id=config.lidar_frame)
            item["point_count"] = cloud.point_count
            item["shape"] = list(cloud.xyz.shape)
            item["original_point_ids_available"] = cloud.source_point_id is not None
        elif key in DEPTH_INPUTS:
            depth = load_depth(path)
            item["shape"] = list(depth.shape)
            item["finite_positive_pixels"] = int(depth.valid.sum())
            item["invalid_pixels"] = int(depth.valid.size - depth.valid.sum())
            item["depth_semantics"] = "raw_model_output" if key == "raw_depth" else "camera_z_m"
        elif key == "sfm_anchors":
            sfm = load_sfm_anchors(path)
            item["anchor_count"] = sfm.count
            item["image_shape"] = list(sfm.image_shape)
            item["sfm_scale"] = sfm.sfm_scale
        elif key == "handeye":
            handeye = load_handeye(
                path, source_frame=config.lidar_frame, target_frame=config.camera_frame
            )
            item["transform"] = f"T_{config.camera_frame}_{config.lidar_frame}"
            item["sfm_scale"] = handeye.sfm_scale
        report["artifacts"][key] = item

    if "camera_planes" in config.inputs:
        planes = load_camera_planes(
            config.require("camera_planes"),
            config.require("camera_masks"),
            frame_id=config.camera_frame,
        )
        report["artifacts"]["camera_planes"]["plane_count"] = len(planes)
        report["artifacts"]["camera_masks"]["mask_shape"] = list(planes[0].mask.shape)
        report["artifacts"]["camera_masks"]["plane_count"] = len(planes)

    if "sfm_anchors" in config.inputs and "metric_depth" in config.inputs:
        sfm_shape = report["artifacts"]["sfm_anchors"]["image_shape"]
        depth_shape = report["artifacts"]["metric_depth"]["shape"]
        if sfm_shape != depth_shape:
            raise ValueError(
                f"SfM image shape {sfm_shape} differs from metric depth {depth_shape}"
            )

    return report
