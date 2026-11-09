"""Recover/replay the two selected historical D1 LiDAR refinement outputs.

IMPORTANT: recovering rigid per-scan transforms from already-refined XYZ is
NOT the original Joint 6-DoF or Point-to-Plane PGO optimizer. Those optimizer
sources are still missing. No B/C1 or GICP refinement is implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

import numpy as np
from numpy.typing import NDArray

from .deskew import AccumulatedCloud

RefinementMethod = Literal['Joint', 'PGO']
SUPPORTED_METHODS: tuple[RefinementMethod, ...] = ('Joint', 'PGO')
MAX_REFERENCE_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class RecoveredRefinement:
    """Reconstructed scan corrections and residuals for an archived method.

    corrections_target_aligned: (S,4,4), with column-vector convention:
        x_refined = R @ x_glim_deskewed + t.
    These are POST-HOC corrections inferred from known final point locations.
    """
    method: RefinementMethod
    corrections_target_aligned: NDArray[np.float64]
    scan_ids: NDArray[np.int64]
    max_residual_m: float
    p95_residual_m: float
    rms_residual_m: float
    anchor_scan_id: int


def load_archived_refined_xyz(archive_path: str | Path, method: RefinementMethod) -> NDArray[np.float64]:
    """Load one of the two FINAL D1 reference refined point clouds from ZIP.

    This consumes the historical calibration input, not a solver output.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f'Unsupported refinement method: {method!r}; expected Joint or PGO')
    with ZipFile(archive_path) as archive:
        suffix = f'/inputs/{method}_xyz.npy'
        entries = [info for info in archive.infolist() if info.filename.endswith(suffix)]
        if len(entries) != 1:
            raise ValueError(f'Expected exactly one {suffix} in reference archive')
        info = entries[0]
        if info.file_size > MAX_REFERENCE_BYTES:
            raise ValueError('Refined cloud exceeds configured size limit')
        values = np.load(BytesIO(archive.read(info)), allow_pickle=False)
    if values.ndim != 2 or values.shape[1] != 3 or values.dtype.kind not in 'f':
        raise ValueError(f'{method}: expected floating-point Nx3 reference PCD')
    if not np.isfinite(values).all():
        raise ValueError(f'{method}: non-finite refined XYZ')
    return np.asarray(values, dtype=np.float64)


def _rigid_fit_row(source: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(source) < 3:
        raise ValueError('Rigid fit needs >=3 points per scan')
    src_mean, tgt_mean = source.mean(axis=0), target.mean(axis=0)
    if np.linalg.matrix_rank(source - src_mean, tol=1e-8) < 2:
        raise ValueError('Insufficient scan geometry for rigid transform')
    u, singular, vh = np.linalg.svd((source - src_mean).T @ (target - tgt_mean))
    diag = np.eye(3)
    diag[-1, -1] = np.sign(np.linalg.det(u @ vh))
    row_rotation = u @ diag @ vh  # row convention x_refined=x_source@row_rotation+t
    translation = tgt_mean - src_mean @ row_rotation
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = row_rotation.T
    t[:3, 3] = translation
    return t


def apply_scan_corrections(
    cloud: AccumulatedCloud,
    scan_ids: NDArray[np.int64],
    corrections_target_aligned: NDArray[np.float64],
) -> AccumulatedCloud:
    """Apply one target-LiDAR-frame rigid correction per source scan."""
    scan_ids = np.asarray(scan_ids, dtype=np.int64)
    corrections = np.asarray(corrections_target_aligned, dtype=np.float64)
    if scan_ids.ndim != 1 or len(np.unique(scan_ids)) != len(scan_ids):
        raise ValueError('scan_ids must be a unique 1D list')
    if corrections.shape != (len(scan_ids), 4, 4) or not np.isfinite(corrections).all():
        raise ValueError('Invalid correction matrix array shape/values')
    if not np.allclose(corrections[:, 3, :], [0, 0, 0, 1], atol=1e-10):
        raise ValueError('Correction matrices must have valid last row')
    if not np.array_equal(np.unique(cloud.scan_id), np.sort(scan_ids)):
        raise ValueError('Correction scan IDs do not match source cloud')
    xyz = np.empty_like(cloud.xyz)
    for scan, transform in zip(scan_ids, corrections):
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or np.linalg.det(rotation) < 0.999999:
            raise ValueError('Corrections must contain proper 3D rotations')
        selected = cloud.scan_id == scan
        xyz[selected] = (np.asarray(cloud.xyz[selected], dtype=np.float64) @ rotation.T + transform[:3, 3]).astype(np.float32)
    return AccumulatedCloud(
        xyz, cloud.scan_id.copy(), cloud.time_offset_s.copy(), cloud.ring.copy(),
        cloud.source_return_index.copy(), cloud.reference_timestamp_ns, cloud.frame_id,
    )


def recover_reference_scan_corrections(
    cloud: AccumulatedCloud,
    reference_xyz: NDArray[np.float64],
    *, method: RefinementMethod,
    anchor_scan_id: int,
) -> RecoveredRefinement:
    """Post-hoc fit scan transforms to stored Joint/PGO D1 XYZ reference.

    Assumes identical point ordering, as verified separately against the D1
    recording. Does not estimate unseen-scene transforms and is not Joint/PGO.
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError('Only Joint and PGO are supported')
    original = np.asarray(cloud.xyz, dtype=np.float64)
    target = np.asarray(reference_xyz, dtype=np.float64)
    if original.shape != target.shape or not np.isfinite(target).all():
        raise ValueError('Reference must be finite, Nx3 and match original point order/count')
    scan_ids = np.unique(cloud.scan_id).astype(np.int64)
    if anchor_scan_id not in scan_ids:
        raise ValueError('Anchor scan absent from source cloud')
    corrections = np.repeat(np.eye(4, dtype=np.float64)[None], len(scan_ids), axis=0)
    all_errors = np.empty(len(original), dtype=np.float64)
    for i, scan in enumerate(scan_ids):
        indices = np.flatnonzero(cloud.scan_id == scan)
        transform = _rigid_fit_row(original[indices], target[indices])
        corrections[i] = transform
        projected = original[indices] @ transform[:3, :3].T + transform[:3, 3]
        all_errors[indices] = np.linalg.norm(projected - target[indices], axis=1)
    return RecoveredRefinement(
        method=method, corrections_target_aligned=corrections, scan_ids=scan_ids,
        max_residual_m=float(np.max(all_errors)),
        p95_residual_m=float(np.percentile(all_errors, 95)),
        rms_residual_m=float(np.sqrt(np.mean(all_errors**2))),
        anchor_scan_id=int(anchor_scan_id),
    )