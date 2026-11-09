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
    time_offset_s: NDArray[np.float64]
    ring: NDArray[np.uint16]
    source_return_index: NDArray[np.int64]

    def __post_init__(self):
        n = len(self.xyz)
        if self.xyz.shape != (n, 3) or any(np.asarray(a).shape != (n,) for a in
            (self.time_offset_s, self.ring, self.source_return_index)):
            raise ValueError('LiDAR scan has incompatible array shapes')
        if not np.isfinite(self.xyz).all() or not np.isfinite(self.time_offset_s).all():
            raise ValueError('LiDAR scan contains non-finite coordinates/times')


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
                     chosen.astype(np.int64))


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
