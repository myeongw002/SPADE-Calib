#!/usr/bin/env python3
"""D1 selected methods: GLIM Deskew + archived Joint/P2Plane-PGO results.

This does NOT re-run the original refinement optimization. It validates the
historical outputs and recovers their per-scan rigid corrections by SVD for
provenance/diagnostics; their original optimizers are not yet available.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from spade_calib.lidar import (
    SUPPORTED_METHODS, load_d1_scan_zip, accumulate_glim,
    load_archived_refined_xyz, recover_reference_scan_corrections,
    target_free_space_mask, filter_cloud, FreeSpaceSettings,
)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def run(
    raw_archive: Path, reference_archive: Path, output_dir: Path, *,
    method: str, ghost_filter: bool = False, anchor_scan_id: int = 5,
    max_replay_error_m: float = 2e-5,
) -> dict:
    if method not in SUPPORTED_METHODS:
        raise ValueError('Only the selected Joint and PGO methods are supported')
    if max_replay_error_m <= 0 or not np.isfinite(max_replay_error_m):
        raise ValueError('max_replay_error_m must be finite and positive')
    scans, trajectory, manifest = load_d1_scan_zip(raw_archive)
    timestamp = int(manifest['lidar_anchor_timestamp_ns'])
    if not 0 <= anchor_scan_id < len(scans):
        raise ValueError('Invalid anchor scan ID')
    if scans[anchor_scan_id].timestamp_ns != timestamp:
        raise ValueError('Anchor scan must coincide with target frame for Free-space Filtering')
    deskewed = accumulate_glim(scans, trajectory, target_timestamp_ns=timestamp)
    xyz_reference = load_archived_refined_xyz(reference_archive, method)
    if xyz_reference.shape != deskewed.xyz.shape:
        raise ValueError('Reference point counts/shapes mismatch; do not align by index')
    recovered = recover_reference_scan_corrections(
        deskewed, xyz_reference, method=method, anchor_scan_id=anchor_scan_id,
    )
    if recovered.max_residual_m > max_replay_error_m:
        raise ValueError(
            f'Original {method} is not representable by scan rigid corrections at '
            f'{max_replay_error_m:g} m: max={recovered.max_residual_m:g} m; '
            'do not assume matching point identity or use the recovered poses'
        )
    # Preserve the *exact saved reference XYZ* (not the approximate SVD replay).
    refined = type(deskewed)(
        xyz_reference.astype(np.float32), deskewed.scan_id.copy(),
        deskewed.time_offset_s.copy(), deskewed.ring.copy(),
        deskewed.source_return_index.copy(), deskewed.reference_timestamp_ns,
        deskewed.frame_id,
    )
    dropped = 0
    if ghost_filter:
        mask = target_free_space_mask(refined, target_scan_id=anchor_scan_id, settings=FreeSpaceSettings())
        dropped = int(np.count_nonzero(~mask))
        refined = filter_cloud(refined, mask)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f'd1_{method}_refined.npz', xyz=refined.xyz, scan_id=refined.scan_id,
        time_offset_s=refined.time_offset_s, ring=refined.ring,
        source_return_index=refined.source_return_index,
        reference_timestamp_ns=np.int64(refined.reference_timestamp_ns),
    )
    np.savez_compressed(
        output_dir / f'd1_{method}_recovered_poses.npz',
        scan_ids=recovered.scan_ids,
        T_target_refined_from_target_deskewed=recovered.corrections_target_aligned,
    )
    report = {
        'method': method,
        'data_source': 'historical_refined_xyz_not_recomputed_by_optimizer',
        'original_joint_pgo_solver_reproduced': False,
        'raw_archive_sha256': file_sha256(raw_archive),
        'reference_archive_sha256': file_sha256(reference_archive),
        'input_scans': len(scans),
        'input_points': len(deskewed.xyz),
        'anchor_scan_id': anchor_scan_id,
        'target_timestamp_ns': timestamp,
        'rigid_fit_max_residual_m': recovered.max_residual_m,
        'rigid_fit_p95_residual_m': recovered.p95_residual_m,
        'rigid_fit_rms_residual_m': recovered.rms_residual_m,
        'ghost_filter_enabled': ghost_filter,
        'ghost_filter_rejected_points': dropped,
        'output_points': len(refined.xyz),
        'notes': (
            'Recovered corrections are fit to the archived output (post-hoc); '
            'they do not reproduce the optimizer. Free-space filtering is optional '
            'and rejected points are NOT verified ground-truth ghosts.'
        ),
    }
    (output_dir / f'd1_{method}_report.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8',
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-archive', required=True, type=Path)
    parser.add_argument('--reference-archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--method', required=True, choices=SUPPORTED_METHODS)
    parser.add_argument('--ghost-filter', action='store_true')
    parser.add_argument('--anchor-scan-id', type=int, default=5)
    args = parser.parse_args()
    result = run(
        args.raw_archive, args.reference_archive, args.output,
        method=args.method, ghost_filter=args.ghost_filter,
        anchor_scan_id=args.anchor_scan_id,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()