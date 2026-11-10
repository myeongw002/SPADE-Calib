"""Metadata-only COLMAP/pycolmap registered-image readers.

Images store T_camera_from_sfm_world, with *unknown global metric scale*
unless an independent alignment has been performed. No file contents are
executed and observations are not loaded into memory.
"""
from __future__ import annotations

import csv
import math
import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegisteredImage:
    image_id: int
    name: str
    camera_id: int
    qvec_wxyz: tuple[float, float, float, float]
    tvec: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.image_id <= 0 or self.camera_id <= 0 or not self.name.strip():
            raise ValueError("Invalid COLMAP image ID, camera ID or image name")
        if len(self.qvec_wxyz) != 4 or len(self.tvec) != 3 or not all(
            math.isfinite(float(v)) for v in (*self.qvec_wxyz, *self.tvec)
        ):
            raise ValueError("Invalid COLMAP pose")
        norm = math.sqrt(sum(float(v) ** 2 for v in self.qvec_wxyz))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError("COLMAP quaternion must have unit norm")


def resolve_colmap_images(source: Path) -> Path:
    """Identify one registered-image table without guessing a sparse model."""
    source = Path(source).expanduser().resolve(strict=True)
    if source.is_file():
        if source.name not in {"images.bin", "images.txt", "camera_poses.csv"}:
            raise ValueError("SfM input file must be images.bin, images.txt or camera_poses.csv")
        return source
    if not source.is_dir():
        raise ValueError(f"Invalid SfM path: {source}")

    # Each COLMAP sparse model has images.bin/txt. Multiple sparse models are
    # intentionally rejected; sorting and taking model 0 could be wrong.
    candidates = []
    for filename in ("images.bin", "images.txt", "camera_poses.csv"):
        candidates.extend(p for p in source.rglob(filename) if p.is_file())
    if not candidates:
        raise FileNotFoundError(f"No COLMAP registered-image file found under {source}")
    # Binary is preferred over text/CSV *within the same model directory*.
    per_folder: dict[Path, Path] = {}
    for path in candidates:
        per_folder.setdefault(path.parent, path)
        current = per_folder[path.parent]
        priority = {"images.bin": 0, "images.txt": 1, "camera_poses.csv": 2}
        if priority[path.name] < priority[current.name]:
            per_folder[path.parent] = path
    if len(per_folder) != 1:
        raise ValueError(
            "Multiple SfM reconstructions found; specify the exact sparse model directory: "
            + ", ".join(str(p) for p in sorted(per_folder))
        )
    return next(iter(per_folder.values()))


def load_registered_images(source: str | Path) -> dict[str, RegisteredImage]:
    """Return registered images keyed by normalized COLMAP image name."""
    path = resolve_colmap_images(Path(source))
    if path.suffix == ".bin":
        records = _load_binary(path)
    elif path.suffix == ".txt":
        records = _load_text(path)
    else:
        records = _load_csv(path)
    names: dict[str, RegisteredImage] = {}
    ids: set[int] = set()
    for record in records:
        name = record.name.replace("\\", "/").removeprefix("./")
        if name in names or record.image_id in ids:
            raise ValueError(f"Duplicate COLMAP name/ID: {name}")
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"COLMAP image name must be a relative path: {name}")
        names[name] = record
        ids.add(record.image_id)
    if not names:
        raise ValueError(f"No registered SfM images in {path}")
    return names


def _load_csv(path: Path) -> list[RegisteredImage]:
    result = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        cols = {"image_id", "image_name", "camera_id", "qw", "qx", "qy", "qz", "tx", "ty", "tz"}
        if not reader.fieldnames or not cols.issubset(reader.fieldnames):
            raise ValueError(f"COLMAP CSV requires columns: {sorted(cols)}")
        for row in reader:
            result.append(RegisteredImage(
                image_id=int(row["image_id"]), name=row["image_name"],
                camera_id=int(row["camera_id"]),
                qvec_wxyz=tuple(float(row[k]) for k in ("qw", "qx", "qy", "qz")),
                tvec=tuple(float(row[k]) for k in ("tx", "ty", "tz")),
            ))
    return result


def _load_text(path: Path) -> list[RegisteredImage]:
    result = []
    with path.open("r", encoding="utf-8") as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.strip().split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError("Malformed COLMAP images.txt image line")
            result.append(RegisteredImage(
                image_id=int(fields[0]), name=fields[9], camera_id=int(fields[8]),
                qvec_wxyz=tuple(float(v) for v in fields[1:5]),
                tvec=tuple(float(v) for v in fields[5:8]),
            ))
            if not stream.readline():
                raise ValueError("COLMAP images.txt missing POINTS2D line")
    return result


def _read_exact(stream, count: int) -> bytes:
    b = stream.read(count)
    if len(b) != count:
        raise ValueError("Truncated COLMAP images.bin")
    return b


def _load_binary(path: Path) -> list[RegisteredImage]:
    fixed = struct.Struct("<i4d3di")
    result = []
    with path.open("rb") as stream:
        num = struct.unpack("<Q", _read_exact(stream, 8))[0]
        if num > 10_000_000:
            raise ValueError("Unreasonably large COLMAP image count")
        size = path.stat().st_size
        for _ in range(num):
            data = fixed.unpack(_read_exact(stream, fixed.size))
            chars = bytearray()
            while True:
                char = _read_exact(stream, 1)
                if char == b"\x00":
                    break
                if len(chars) >= 8192:
                    raise ValueError("COLMAP image name too long")
                chars.extend(char)
            points2d = struct.unpack("<Q", _read_exact(stream, 8))[0]
            if points2d > (size - stream.tell()) // 24:
                raise ValueError("Truncated COLMAP 2D feature observations")
            stream.seek(points2d * 24, 1)  # x(double), y(double), point3D_id(int64)
            result.append(RegisteredImage(
                image_id=data[0], name=chars.decode("utf-8"), camera_id=data[8],
                qvec_wxyz=tuple(data[1:5]), tvec=tuple(data[5:8]),
            ))
    return result