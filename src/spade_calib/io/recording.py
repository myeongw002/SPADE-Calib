"""Whole-recording dataset loader and deterministic timestamp/trajectory index.

The index is metadata-only: images and PCD/KITTI .bin points are read lazily.
No scene segmentation, calibration, or automatic time-offset estimation occurs.
"""
from __future__ import annotations

import bisect
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from spade_calib.io.colmap import RegisteredImage, load_registered_images, resolve_colmap_images
from spade_calib.io.pcd import load_pcd_scan
from spade_calib.io.kitti_bin import load_kitti_bin_scan
from spade_calib.lidar.scan_io import LidarScan
from spade_calib.lidar.trajectory import GlimTrajectory

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
_LIDAR_EXTENSIONS = frozenset({".pcd", ".bin"})
_NS_MIN_DIGITS = 17  # reject short integer frame numbers as nanosecond timestamps
_NS_MAX_DIGITS = 20
_INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True)
class RecordingConfig:
    name: str
    root: Path
    images: Path
    lidar_scans: Path
    glim_trajectory: Path
    sfm: Path
    camera_calibration: Path | None
    image_timestamps: Path | None
    lidar_timestamps: Path | None
    camera_to_lidar_offset_ns: int
    max_pair_delta_ns: int
    lidar_time_field: str
    lidar_time_unit: str
    lidar_ring_field: str
    min_range_m: float
    max_range_m: float
    recursive: bool
    source_config: Path


@dataclass(frozen=True)
class TimedFile:
    timestamp_ns: int
    path: Path
    relative_path: str


@dataclass(frozen=True)
class CameraAssociation:
    image: TimedFile
    lidar: TimedFile | None
    delta_ns: int | None  # lidar_ts - (camera_ts + offset)
    sfm: RegisteredImage | None
    glim_covered: bool  # frame-level time check, NOT per-point deskew coverage


@dataclass(frozen=True)
class RecordingIndex:
    config: RecordingConfig
    images: tuple[TimedFile, ...]
    lidar_scans: tuple[TimedFile, ...]
    associations: tuple[CameraAssociation, ...]
    sfm_images: dict[str, RegisteredImage]
    glim_trajectory: GlimTrajectory
    sfm_source: Path

    def _read_lidar(self, record: TimedFile, *, scan_id: int) -> LidarScan:
        """Select reader by file type; never fabricate .bin time/ring fields."""
        if record.path.suffix.lower() == '.bin':
            return load_kitti_bin_scan(
                record.path, scan_id=scan_id, timestamp_ns=record.timestamp_ns,
                min_range_m=self.config.min_range_m, max_range_m=self.config.max_range_m,
            )
        if record.path.suffix.lower() == '.pcd':
            return load_pcd_scan(
                record.path, scan_id=scan_id, timestamp_ns=record.timestamp_ns,
                time_field=self.config.lidar_time_field, time_unit=self.config.lidar_time_unit,
                ring_field=self.config.lidar_ring_field,
                min_range_m=self.config.min_range_m, max_range_m=self.config.max_range_m,
            )
        raise ValueError(f'Unsupported LiDAR scan extension: {record.path.suffix}')

    def load_lidar_scan(self, scan: int | TimedFile) -> LidarScan:
        """Lazily load PCD or KITTI .bin with explicit scan timestamp."""
        if isinstance(scan, int):
            record = self.lidar_scans[scan]
            scan_id = scan
        else:
            try:
                scan_id = self.lidar_scans.index(scan)
            except ValueError as exc:
                raise ValueError("Scan record does not belong to this index") from exc
            record = scan
        return self._read_lidar(record, scan_id=scan_id)

    def lidar_window(self, start_ns: int, stop_ns: int) -> tuple[TimedFile, ...]:
        """Return chronological scans in inclusive timestamp bounds, without choosing a scene."""
        if stop_ns < start_ns:
            raise ValueError("stop_ns must be >= start_ns")
        timestamps = [record.timestamp_ns for record in self.lidar_scans]
        lo = bisect.bisect_left(timestamps, start_ns)
        hi = bisect.bisect_right(timestamps, stop_ns)
        return self.lidar_scans[lo:hi]

    def load_window_scans(self, records: Sequence[TimedFile]) -> tuple[LidarScan, ...]:
        """Load a selected window with local 0..K-1 scan IDs for the deskew solver.

        Caller keeps `records` as the explicit mapping from local IDs to original
        timestamps/paths. All point-level source_return_index values remain intact.
        """
        source = {r.path: r for r in self.lidar_scans}
        stamps = [r.timestamp_ns for r in records]
        if len(set(stamps)) != len(stamps) or stamps != sorted(stamps):
            raise ValueError("Selected scan records must be unique and chronological")
        for record in records:
            if source.get(record.path) != record:
                raise ValueError("Selected record not present in this recording")
        if len(records) > 65536:
            raise ValueError('Accumulation scan IDs must fit uint16')
        return tuple(self._read_lidar(record, scan_id=i)
                     for i, record in enumerate(records))

    def to_dict(self) -> dict[str, Any]:
        def file_row(record: TimedFile) -> dict[str, Any]:
            return {"timestamp_ns": record.timestamp_ns,
                    "path": str(record.path), "relative_path": record.relative_path}

        matched = sum(x.lidar is not None for x in self.associations)
        sfm_matched = sum(x.sfm is not None for x in self.associations)
        glim_scans = sum(_glim_covers(self.glim_trajectory, s.timestamp_ns) for s in self.lidar_scans)
        data = {
            "schema_version": 2,
            "dataset": {"name": self.config.name, "root": str(self.config.root)},
            "inputs": {
                "images": str(self.config.images), "lidar_scans": str(self.config.lidar_scans),
                "glim_trajectory": str(self.config.glim_trajectory),
                "sfm_registered_images": str(self.sfm_source),
                "camera_calibration": str(self.config.camera_calibration) if self.config.camera_calibration else None,
            },
            "matching_policy": {
                "camera_to_lidar_offset_ns": self.config.camera_to_lidar_offset_ns,
                "max_pair_delta_ns": self.config.max_pair_delta_ns,
                "timestamp_policy": "filename_ns_or_explicit_csv",
                "nearest_lidar_reuse_allowed": True,
                "sfm_lookup": "exact_relative_name_or_unique_basename",
            },
            "summary": {
                "images": len(self.images), "lidar_scans": len(self.lidar_scans),
                "sfm_registered_images": len(self.sfm_images),
                "images_with_sfm_pose": sfm_matched,
                "camera_lidar_matched_images": matched,
                "unpaired_images": len(self.images) - matched,
                "lidar_scans_within_glim_range": glim_scans,
                "kitti_bin_scans_without_per_return_time": sum(
                    r.path.suffix.lower() == '.bin' for r in self.lidar_scans),
                "glim_pose_count": len(self.glim_trajectory.timestamps),
                "glim_range_ns_approx": [
                    int(round(float(self.glim_trajectory.timestamps[0]) * 1e9)),
                    int(round(float(self.glim_trajectory.timestamps[-1]) * 1e9)),
                ],
            },
            "images": [file_row(x) for x in self.images],
            "lidar_scans": [file_row(x) for x in self.lidar_scans],
            "camera_associations": [{
                "image": a.image.relative_path,
                "image_timestamp_ns": a.image.timestamp_ns,
                "lidar": a.lidar.relative_path if a.lidar else None,
                "lidar_timestamp_ns": a.lidar.timestamp_ns if a.lidar else None,
                "delta_ns": a.delta_ns,
                "sfm_image_id": a.sfm.image_id if a.sfm else None,
                "sfm_camera_id": a.sfm.camera_id if a.sfm else None,
                "sfm_qvec_wxyz": list(a.sfm.qvec_wxyz) if a.sfm else None,
                "sfm_tvec": list(a.sfm.tvec) if a.sfm else None,
                "glim_covered": a.glim_covered,
            } for a in self.associations],
            "warnings": [],
        }
        if sfm_matched < len(self.images):
            data["warnings"].append("Some images are not registered in SfM; no poses were invented")
        if glim_scans < len(self.lidar_scans):
            data["warnings"].append("Some scan timestamps fall outside GLIM trajectory coverage")
        if matched < len(self.images):
            data["warnings"].append("Some camera frames have no LiDAR frame within the configured tolerance")
        if any(s.path.suffix.lower() == ".bin" for s in self.lidar_scans):
            data["warnings"].append(
                "KITTI .bin contains no per-return time/ring; use rigid scan accumulation, "
                "never per-return deskew. Explicit scan timestamps must match the GLIM clock.")
        if not self.config.camera_calibration:
            data["warnings"].append("Camera intrinsics/distortion not configured; future projection requires calibration")
        return data


def _path_under(root: Path, path_str: str, label: str) -> Path:
    if not isinstance(path_str, str) or not path_str.strip():
        raise ValueError(f"{label} must be a nonempty string path")
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=True)


def _optional_path(root: Path, value: Any, label: str) -> Path | None:
    return None if value is None else _path_under(root, value, label)


def _resolve_glim_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise ValueError("GLIM trajectory path is neither file nor directory")
    matches = sorted(p for p in path.rglob("*.txt") if p.name in {
        "traj_lidar.txt", "glim_trajectory.txt", "trajectory.txt"})
    if len(matches) != 1:
        raise ValueError("GLIM directory must contain exactly one known trajectory file; select file explicitly")
    return matches[0]


def load_recording_config(path: str | Path) -> RecordingConfig:
    """Read schema_version=2 without changing legacy schema_version=1 loaders."""
    source = Path(path).expanduser().resolve(strict=True)
    with source.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict) or cfg.get("schema_version") != 2:
        raise ValueError("Recording config requires schema_version: 2")
    dataset, inputs = cfg.get("dataset"), cfg.get("inputs")
    if not isinstance(dataset, dict) or not isinstance(inputs, dict):
        raise ValueError("Recording config requires dataset and inputs mappings")
    name = dataset.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("dataset.name must be nonempty")
    root = _path_under(source.parent, dataset.get("root"), "dataset.root")
    if not root.is_dir():
        raise ValueError("dataset.root must be a directory")
    images = _path_under(root, inputs.get("images"), "inputs.images")
    lidar = _path_under(root, inputs.get("lidar_scans"), "inputs.lidar_scans")
    for folder, label in ((images, "images"), (lidar, "lidar_scans")):
        if not folder.is_dir():
            raise ValueError(f"inputs.{label} must be a directory")
    sfm = _path_under(root, inputs.get("sfm"), "inputs.sfm")
    glim = _resolve_glim_path(_path_under(root, inputs.get("glim"), "inputs.glim"))
    optional = {key: _optional_path(root, inputs.get(key), f"inputs.{key}")
                for key in ("camera_calibration", "image_timestamps", "lidar_timestamps")}
    for key, val in optional.items():
        if val is not None and not val.is_file():
            raise ValueError(f"inputs.{key} must be a file")
    match = cfg.get("matching", {})
    pcd = cfg.get("pcd", {})
    if not isinstance(match, dict) or not isinstance(pcd, dict):
        raise ValueError("matching and pcd must be mappings")
    offset = match.get("camera_to_lidar_offset_ns", 0)
    delta_ms = match.get("max_pair_delta_ms", 100.0)
    if type(offset) is not int or abs(offset) > _INT64_MAX:
        raise ValueError("camera_to_lidar_offset_ns must be an int64 integer")
    if not isinstance(delta_ms, (int, float)) or not np.isfinite(delta_ms) or delta_ms < 0:
        raise ValueError("max_pair_delta_ms must be nonnegative and finite")
    delta_ns = round(float(delta_ms) * 1e6)
    if delta_ns > _INT64_MAX:
        raise ValueError("max_pair_delta_ms is too large")
    min_range, max_range = pcd.get("range_m", [0.5, 100.0])
    if not (0 <= min_range < max_range and np.isfinite(min_range) and np.isfinite(max_range)):
        raise ValueError("Invalid pcd.range_m")
    time_unit = pcd.get("time_unit", "s")
    if time_unit not in {"s", "ms", "us", "ns"}:
        raise ValueError("pcd.time_unit must be s/ms/us/ns")
    recursive = cfg.get("recursive", True)
    if type(recursive) is not bool:
        raise ValueError("recursive must be boolean")
    return RecordingConfig(
        name=name, root=root, images=images, lidar_scans=lidar, sfm=sfm,
        glim_trajectory=glim, camera_calibration=optional["camera_calibration"],
        image_timestamps=optional["image_timestamps"],
        lidar_timestamps=optional["lidar_timestamps"],
        camera_to_lidar_offset_ns=offset, max_pair_delta_ns=delta_ns,
        lidar_time_field=str(pcd.get("time_field", "time")),
        lidar_ring_field=str(pcd.get("ring_field", "ring")),
        lidar_time_unit=time_unit, min_range_m=float(min_range), max_range_m=float(max_range),
        recursive=recursive, source_config=source,
    )


def read_timestamp_csv(path: Path | None) -> dict[str, int] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "timestamp_ns" not in reader.fieldnames or not (
            "file" in reader.fieldnames or "filename" in reader.fieldnames
        ):
            raise ValueError(f"Timestamp CSV requires file(or filename),timestamp_ns: {path}")
        name_key = "file" if "file" in reader.fieldnames else "filename"
        stamps = {}
        for row in reader:
            key = row[name_key].replace("\\", "/").removeprefix("./")
            stamp = _validate_timestamp(row["timestamp_ns"], path)
            if key in stamps or not key or Path(key).is_absolute() or ".." in Path(key).parts:
                raise ValueError(f"Invalid or duplicate timestamp CSV filename: {key!r}")
            stamps[key] = stamp
    if not stamps:
        raise ValueError(f"Empty timestamp CSV: {path}")
    return stamps



# Backward compatible internal alias for existing recording index paths.
_read_timestamp_csv = read_timestamp_csv


def _validate_timestamp(value: str, context: object) -> int:
    text = str(value).strip()
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError(f"Expected integer nanosecond timestamp, got {value!r} in {context}")
    timestamp = int(text)
    if timestamp > _INT64_MAX:
        raise ValueError(f"Timestamp outside int64 range in {context}")
    return timestamp


def _enumerate_timed(folder: Path, extensions: frozenset[str], *, recursive: bool,
                     timestamps: dict[str, int] | None) -> tuple[TimedFile, ...]:
    paths = sorted(p for p in (folder.rglob("*") if recursive else folder.iterdir())
                   if p.is_file() and p.suffix.lower() in extensions)
    if not paths:
        raise FileNotFoundError(f"No supported files under {folder}; extensions={sorted(extensions)}")
    basename_counts: dict[str, int] = {}
    for p in paths:
        basename_counts[p.name] = basename_counts.get(p.name, 0) + 1
    records = []
    used_keys: set[str] = set()
    for path in paths:
        rel = path.relative_to(folder).as_posix()
        if timestamps is None:
            stem = path.stem
            if not stem.isdigit() or not _NS_MIN_DIGITS <= len(stem) <= _NS_MAX_DIGITS:
                raise ValueError(f"Missing nanosecond timestamp in filename {path.name}; supply CSV")
            stamp = _validate_timestamp(stem, path)
        else:
            if rel in timestamps:
                key = rel
            elif basename_counts[path.name] == 1 and path.name in timestamps:
                key = path.name
            else:
                raise ValueError(f"No unique timestamp CSV entry for {rel}")
            used_keys.add(key)
            stamp = timestamps[key]
        records.append(TimedFile(stamp, path, rel))
    if timestamps is not None and set(timestamps) - used_keys:
        raise ValueError(f"Timestamp CSV contains files not indexed: {sorted(set(timestamps) - used_keys)[:5]}")
    records.sort(key=lambda row: (row.timestamp_ns, row.relative_path))
    duplicates = [(a.relative_path, b.relative_path) for a, b in zip(records, records[1:])
                  if a.timestamp_ns == b.timestamp_ns]
    if duplicates:
        raise ValueError(f"Duplicate timestamps for a single sensor stream: {duplicates[:3]}")
    return tuple(records)


def _glim_covers(trajectory: GlimTrajectory, timestamp_ns: int) -> bool:
    # Conversion to float seconds can introduce ~100 ns error at epoch 2026;
    # tolerance is only a numeric precision allowance, not a sync allowance.
    sec = timestamp_ns * 1e-9
    return bool(trajectory.timestamps[0] - 1e-6 <= sec <= trajectory.timestamps[-1] + 1e-6)


def _matching_sfm(images: tuple[TimedFile, ...], poses: dict[str, RegisteredImage]) -> dict[str, RegisteredImage | None]:
    image_by_name = {r.relative_path: r for r in images}
    by_basename: dict[str, list[RegisteredImage]] = {}
    for name, pose in poses.items():
        by_basename.setdefault(Path(name).name, []).append(pose)
    img_basename_count: dict[str, int] = {}
    for row in images:
        img_basename_count[row.path.name] = img_basename_count.get(row.path.name, 0) + 1
    matched = {}
    for row in images:
        pose = poses.get(row.relative_path)
        if pose is None:
            options = by_basename.get(row.path.name, [])
            if len(options) == 1 and img_basename_count[row.path.name] == 1:
                pose = options[0]
            elif options and img_basename_count[row.path.name] > 1:
                raise ValueError(f"Ambiguous SfM basename {row.path.name!r}; fix relative paths")
            elif len(options) > 1:
                raise ValueError(f"Ambiguous SfM basename {row.path.name!r}; fix paths")
        matched[row.relative_path] = pose
    return matched


def index_recording(config: RecordingConfig) -> RecordingIndex:
    """Scan an entire recording and associate, but do not segment or optimize."""
    images = _enumerate_timed(config.images, _IMAGE_EXTENSIONS, recursive=config.recursive,
                             timestamps=_read_timestamp_csv(config.image_timestamps))
    scans = _enumerate_timed(config.lidar_scans, _LIDAR_EXTENSIONS, recursive=config.recursive,
                            timestamps=_read_timestamp_csv(config.lidar_timestamps))
    sfm_path = resolve_colmap_images(config.sfm)
    poses = load_registered_images(sfm_path)
    sfm_match = _matching_sfm(images, poses)
    trajectory = GlimTrajectory.from_array(np.loadtxt(config.glim_trajectory, dtype=np.float64, ndmin=2))
    scan_times = [x.timestamp_ns for x in scans]
    associations = []
    for img in images:
        target = img.timestamp_ns + config.camera_to_lidar_offset_ns
        if not 0 <= target <= _INT64_MAX:
            raise ValueError(f"Camera time offset overflow: {img.relative_path}")
        at = bisect.bisect_left(scan_times, target)
        options = [i for i in (at - 1, at) if 0 <= i < len(scans)]
        best = min(options, key=lambda i: (abs(scan_times[i] - target), scan_times[i]))
        diff = scan_times[best] - target
        scan = scans[best] if abs(diff) <= config.max_pair_delta_ns else None
        associations.append(CameraAssociation(
            image=img, lidar=scan, delta_ns=diff if scan else None,
            sfm=sfm_match[img.relative_path],
            glim_covered=_glim_covers(trajectory, target),
        ))
    return RecordingIndex(config, images, scans, tuple(associations), poses, trajectory, sfm_path)


def write_recording_index(index: RecordingIndex, path: str | Path) -> dict[str, Any]:
    """Atomically write metadata manifest; never touch source files."""
    dest = Path(path).expanduser().resolve()
    source_files = {record.path for record in (*index.images, *index.lidar_scans)}
    source_files.update({index.config.glim_trajectory, index.sfm_source, index.config.source_config})
    source_files.update(x for x in (index.config.camera_calibration,
                                   index.config.image_timestamps,
                                   index.config.lidar_timestamps) if x is not None)
    if dest in source_files:
        raise ValueError("Refusing to overwrite an indexed input file")
    document = index.to_dict()
    # Fingerprint is metadata-only, NOT hashes of all input bytes.
    summary = json.dumps({
        "dataset": index.config.name,
        "images": [(r.relative_path, r.timestamp_ns, r.path.stat().st_size, r.path.stat().st_mtime_ns)
                   for r in index.images],
        "lidar": [(r.relative_path, r.timestamp_ns, r.path.stat().st_size, r.path.stat().st_mtime_ns)
                  for r in index.lidar_scans],
        "glim": (str(index.config.glim_trajectory),
                 index.config.glim_trajectory.stat().st_size,
                 index.config.glim_trajectory.stat().st_mtime_ns),
        "sfm": (str(index.sfm_source), index.sfm_source.stat().st_size,
                index.sfm_source.stat().st_mtime_ns),
        "config_sha256": hashlib.sha256(index.config.source_config.read_bytes()).hexdigest(),
        "offset_ns": index.config.camera_to_lidar_offset_ns,
        "threshold_ns": index.config.max_pair_delta_ns,
        "pcd_time_field": index.config.lidar_time_field,
        "pcd_time_unit": index.config.lidar_time_unit,
        "pcd_range_m": [index.config.min_range_m, index.config.max_range_m],
    }, sort_keys=True)
    document["metadata_fingerprint_sha256"] = hashlib.sha256(summary.encode()).hexdigest()
    document["fingerprint_note"] = "metadata-only; not content hashes of PCD/images"
    dest.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=dest.parent, prefix=".recording_",
                                         suffix=".tmp", encoding="utf-8", delete=False) as handle:
            tmp = Path(handle.name)
            json.dump(document, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, dest)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return document