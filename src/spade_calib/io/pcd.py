"""General PCD scanner for raw LiDAR returns (binary and ASCII).

The indexed path is opened lazily. The time field MUST represent an offset
relative to the scan timestamp; absolute-per-point timestamps need a separate
explicit adapter. No deskew is performed in this module.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from spade_calib.lidar.scan_io import LidarScan

_MAX_HEADER = 64 * 1024
_MAX_POINTS = 20_000_000
_PCD_DTYPES = {
    ("F", 4): "<f4", ("F", 8): "<f8",
    ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
    ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
}
_TIME_FACTORS = {"s": 1., "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


def load_pcd_scan(
    path: str | Path,
    *,
    scan_id: int,
    timestamp_ns: int,
    time_field: str = "time",
    time_unit: str = "s",
    ring_field: str = "ring",
    min_range_m: float = 0.5,
    max_range_m: float = 100.0,
) -> LidarScan:
    """Read one scan with preserved *source* return indices.

    Missing ring fields receive uint16(65535), explicitly marking unknown;
    a missing time field is rejected, never silently treated as zero.
    """
    if time_unit not in _TIME_FACTORS:
        raise ValueError("time_unit must be s, ms, us or ns")
    if not (np.isfinite(min_range_m) and np.isfinite(max_range_m)
            and 0 <= min_range_m < max_range_m):
        raise ValueError("Invalid range bounds")
    if scan_id < 0 or timestamp_ns < 0:
        raise ValueError("Invalid scan ID or timestamp")
    path = Path(path)
    with path.open("rb") as stream:
        hdr: dict[str, str] = {}
        bytes_read = 0
        while True:
            line = stream.readline()
            bytes_read += len(line)
            if not line or bytes_read > _MAX_HEADER:
                raise ValueError("Missing or oversized PCD header")
            text = line.decode("ascii", errors="strict").strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Malformed PCD header line: {text!r}")
            key, value = parts
            hdr[key.upper()] = value.strip()
            if key.upper() == "DATA":
                break
        fields = hdr.get("FIELDS", "").split()
        if len(fields) != len(set(fields)) or not fields:
            raise ValueError("PCD FIELDS must be nonempty and unique")
        sizes = [int(x) for x in hdr.get("SIZE", "").split()]
        types = hdr.get("TYPE", "").split()
        counts = [int(x) for x in hdr.get("COUNT", " ".join("1" for _ in fields)).split()]
        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise ValueError("PCD FIELDS/SIZE/TYPE/COUNT lengths differ")
        if any(n < 1 for n in counts):
            raise ValueError("PCD COUNT values must be positive")
        spec = []
        for field, size, type_, count in zip(fields, sizes, types, counts):
            if (type_, size) not in _PCD_DTYPES:
                raise ValueError(f"Unsupported PCD data type: {type_}{size}")
            spec.append((field, _PCD_DTYPES[type_, size], (count,)) if count != 1
                        else (field, _PCD_DTYPES[type_, size]))
        dtype = np.dtype(spec)
        n = int(hdr.get("POINTS", "-1"))
        if n < 0 or n > _MAX_POINTS:
            raise ValueError("Invalid PCD point count")
        if "WIDTH" in hdr and "HEIGHT" in hdr and int(hdr["WIDTH"]) * int(hdr["HEIGHT"]) != n:
            raise ValueError("PCD WIDTH*HEIGHT differs from POINTS")
        mode = hdr["DATA"].lower()
        if mode == "binary":
            raw = stream.read(n * dtype.itemsize + 1)
            if len(raw) != n * dtype.itemsize:
                raise ValueError("PCD binary payload length differs from header")
            data = np.frombuffer(raw, dtype=dtype, count=n)
        elif mode == "ascii":
            # Structured ASCII parsing with COUNT>1 requires a positional map.
            matrix = np.loadtxt(stream, dtype=np.float64, ndmin=2)
            if matrix.shape != (n, sum(counts)):
                raise ValueError("PCD ASCII payload shape differs from header")
            data = np.empty(n, dtype=dtype)
            cursor = 0
            for field, count in zip(fields, counts):
                data[field] = matrix[:, cursor:cursor + count].reshape(-1) if count == 1 \
                    else matrix[:, cursor:cursor + count]
                cursor += count
        else:
            raise ValueError(f"Unsupported PCD DATA mode: {mode} (binary_compressed not supported)")

    for field in ("x", "y", "z", time_field):
        if field not in fields or np.asarray(data[field]).ndim != 1:
            raise ValueError(f"PCD requires scalar field {field!r} for per-point deskew")
    xyz = np.stack([data[f] for f in ("x", "y", "z")], axis=1).astype(np.float64)
    times = np.asarray(data[time_field], dtype=np.float64) * _TIME_FACTORS[time_unit]
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(times)
    radius = np.linalg.norm(xyz, axis=1)
    valid &= (radius > min_range_m) & (radius < max_range_m)
    chosen = np.flatnonzero(valid)
    if ring_field in fields:
        original_ring = np.asarray(data[ring_field])
        if original_ring.ndim != 1 or not np.isfinite(original_ring).all():
            raise ValueError("Ring field must contain finite scalar values")
        if (original_ring < 0).any() or (original_ring > np.iinfo(np.uint16).max).any():
            raise ValueError("Ring ID must fit uint16")
        rings = original_ring[chosen].astype(np.uint16)
    else:
        rings = np.full(len(chosen), np.iinfo(np.uint16).max, dtype=np.uint16)
    return LidarScan(
        scan_id=int(scan_id), timestamp_ns=int(timestamp_ns), xyz=xyz[chosen],
        time_offset_s=times[chosen], ring=rings, source_return_index=chosen.astype(np.int64),
    )