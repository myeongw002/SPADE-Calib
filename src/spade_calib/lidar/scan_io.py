"""Read D1 raw VLP16 scans with exact source-return identity and time offsets."""
from __future__ import annotations
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from numpy.typing import NDArray
from .trajectory import GlimTrajectory

# Specific historical VLP-16 PCD exported schema: 22 bytes/point.
D1_POINT_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'),
    ('ring', '<u2'), ('time', '<f4'),
])


@dataclass(frozen=True)
class LidarScan:
    scan_id: int
    timestamp_ns: int
    xyz: NDArray[np.float64]
    # None means that per-return timing/ring metadata is absent, NOT zero.
    time_offset_s: NDArray[np.float64] | None
    ring: NDArray[np.uint16] | None
    source_return_index: NDArray[np.int64]
    # Optional measured reflectance; left untouched (no invented normalization).
    intensity: NDArray[np.float64] | None = None

    def __post_init__(self):
        if type(self.scan_id) is not int or not (0 <= self.scan_id < 65536):
            raise ValueError('scan_id must fit the accumulated uint16 ID')
        if type(self.timestamp_ns) is not int or self.timestamp_ns < 0:
            raise ValueError('timestamp_ns must be a nonnegative integer')
        n = len(self.xyz)
        required = (self.source_return_index,)
        optional = (self.time_offset_s, self.ring, self.intensity)
        if self.xyz.shape != (n, 3) or any(np.asarray(a).shape != (n,) for a in required):
            raise ValueError('LiDAR scan has incompatible array shapes')
        if any(np.asarray(a).shape != (n,) for a in optional if a is not None):
            raise ValueError('LiDAR scan metadata has incompatible array shapes')
        if not np.isfinite(self.xyz).all() or any(
            not np.isfinite(a).all() for a in (self.time_offset_s, self.intensity)
            if a is not None
        ):
            raise ValueError('LiDAR scan contains non-finite coordinates/times/intensity')
        if np.any(self.source_return_index < 0) or len(np.unique(self.source_return_index)) != n:
            raise ValueError('Invalid or duplicate source return indices')

    @property
    def has_per_return_time(self) -> bool:
        return self.time_offset_s is not None

    @property
    def has_ring(self) -> bool:
        return self.ring is not None


def _decode_binary_pcd(raw: bytes, *, scan_id: int, timestamp_ns: int,
                       min_range_m: float, max_range_m: float) -> LidarScan:
    if min_range_m < 0 or max_range_m <= min_range_m:
        raise ValueError('Invalid LiDAR range bounds')
    end = raw.find(b'DATA binary\n')
    if end < 0 or end > 8192:
        raise ValueError('Expected binary PCD with DATA binary header')
    header = raw[:end].decode('ascii', errors='strict')
    schema = {
        'FIELDS': 'x y z intensity ring time',
        'SIZE': '4 4 4 4 2 4',
        'TYPE': 'F F F F U F',
        'COUNT': '1 1 1 1 1 1',
    }
    lines = dict((line.split(maxsplit=1) for line in header.splitlines()
                  if line and not line.startswith('#') and len(line.split(maxsplit=1)) == 2))
    for key, value in schema.items():
        if lines.get(key) != value:
            raise ValueError(f'Unexpected PCD {key}: {lines.get(key)}')
    count = int(lines.get('POINTS', '-1'))
    offset = end + len(b'DATA binary\n')
    if count < 0 or len(raw) - offset != count * D1_POINT_DTYPE.itemsize:
        raise ValueError('PCD binary payload size does not match POINTS')
    cloud = np.frombuffer(raw, dtype=D1_POINT_DTYPE, count=count, offset=offset)
    xyz = np.stack([cloud[k] for k in ('x', 'y', 'z')], axis=-1).astype(np.float64)
    good = np.isfinite(xyz).all(axis=1) & np.isfinite(cloud['intensity'])
    r = np.linalg.norm(xyz[good], axis=1)
    chosen = np.flatnonzero(good)[(r > min_range_m) & (r < max_range_m)]
    return LidarScan(scan_id, timestamp_ns, xyz[chosen],
                     cloud['time'][chosen].astype(np.float64), cloud['ring'][chosen].copy(),
                     chosen.astype(np.int64),
                     intensity=cloud['intensity'][chosen].astype(np.float64))


def load_d1_scan_zip(path: str | Path, *, min_range_m: float = 0.5,
                     max_range_m: float = 100.0) -> tuple[list[LidarScan], GlimTrajectory, dict]:
    """Read the user's D1 'distance' zip, sorted by its manifest scan order.

    Source-return indices identify positions in each unfiltered raw PCD.
    The archive is data only; executable code and unsafe names are rejected.
    """
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        manifests = [n for n in names if re.fullmatch(r'[^/]+/manifest\.json', n)]
        if len(manifests) != 1:
            raise ValueError('ZIP must contain exactly one scene/manifest.json')
        prefix = manifests[0].rsplit('/', 1)[0] + '/'
        manifest = json.loads(archive.read(prefix + 'manifest.json'))
        entries = manifest.get('scans')
        if not isinstance(entries, list) or not entries:
            raise ValueError('D1 manifest has no scans')
        trajectory_name = manifest.get('trajectory_file', 'glim_trajectory.txt')
        if trajectory_name != 'glim_trajectory.txt':
            raise ValueError('Unexpected trajectory path in D1 manifest')
        traj = GlimTrajectory.from_array(np.loadtxt(io.BytesIO(archive.read(prefix + trajectory_name))))
        scans = []
        for i, entry in enumerate(entries):
            file_name = entry['file']
            if Path(file_name).name != file_name or not file_name.endswith('.pcd'):
                raise ValueError(f'Unexpected scan filename: {file_name!r}')
            stamp = int(entry['timestamp_ns'])
            if file_name != f'{stamp}.pcd' or prefix + file_name not in names:
                raise ValueError('PCD filename/timestamp mismatch')
            scans.append(_decode_binary_pcd(archive.read(prefix + file_name), scan_id=i,
                              timestamp_ns=stamp, min_range_m=min_range_m,
                              max_range_m=max_range_m))
        if int(manifest.get('scan_count', -1)) != len(scans):
            raise ValueError('D1 manifest scan count mismatch')
        if len({s.timestamp_ns for s in scans}) != len(scans):
            raise ValueError('D1 scan timestamps are not unique')
    return scans, traj, manifest
