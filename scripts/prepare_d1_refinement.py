#!/usr/bin/env python3
"""Run D1 GLIM deskew + selected Joint or Point-to-Plane PGO refinement.

Independent optimization: the historical refined-cloud archive is NEVER used
by the solver. It may only be supplied for after-the-fact regression metrics.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
from spade_calib.lidar import (
    SUPPORTED_METHODS, load_d1_scan_zip, accumulate_glim,
    target_free_space_mask, filter_cloud, FreeSpaceSettings,
)
from spade_calib.lidar.joint_solver import JointSettings, solve_joint
from spade_calib.lidar.pgo_solver import PGOSettings, solve_pgo

LOG = logging.getLogger('spade_calib.refinement')


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(raw_archive: Path, output_dir: Path, *, method: str,
        reference_archive: Path | None = None, ghost_filter: bool = False,
        anchor_scan_id: int = 5, d1_roi: bool = True) -> dict:
    """Refine using raw scans only; reference is read after optimization."""
    if method not in SUPPORTED_METHODS:
        raise ValueError('Only the selected Joint and PGO methods are supported')
    scans, trajectory, manifest = load_d1_scan_zip(raw_archive)
    timestamp = int(manifest['lidar_anchor_timestamp_ns'])
    if not 0 <= anchor_scan_id < len(scans) or scans[anchor_scan_id].timestamp_ns != timestamp:
        raise ValueError('Anchor scan must correspond to the target LiDAR frame')
    deskewed = accumulate_glim(scans, trajectory, target_timestamp_ns=timestamp)
    joint_cfg = JointSettings(anchor_scan_id=anchor_scan_id, d1_roi=d1_roi)
    if method == 'Joint':
        solved = solve_joint(deskewed, joint_cfg)
    else:
        solved = solve_pgo(deskewed, PGOSettings(pairs=joint_cfg))
    if solved.xyz.shape != deskewed.xyz.shape or not np.isfinite(solved.xyz).all():
        raise RuntimeError('Refiner returned invalid point cloud')
    if not np.allclose(solved.transforms[anchor_scan_id], np.eye(4), atol=1e-12):
        raise RuntimeError('Reference scan pose must remain fixed')

    # The optional comparison happens AFTER optimization (never solver input).
    comparison = None
    if reference_archive is not None:
        from spade_calib.lidar.refinement import load_archived_refined_xyz
        reference = load_archived_refined_xyz(reference_archive, method)
        if reference.shape != solved.xyz.shape:
            raise ValueError('Reference cloud must have the same point count/order')
        residual = np.linalg.norm(solved.xyz.astype(np.float64) - reference, axis=1)
        comparison = {
            'max_euclidean_error_m': float(np.max(residual)),
            'mean_euclidean_error_m': float(np.mean(residual)),
            'p95_euclidean_error_m': float(np.percentile(residual, 95)),
            'reference_archive_sha256': file_sha256(reference_archive),
            'note': 'same-index point comparison; valid only when original scan/return order is identical',
        }

    refined = type(deskewed)(
        solved.xyz, deskewed.scan_id.copy(), deskewed.time_offset_s.copy(),
        deskewed.ring.copy(), deskewed.source_return_index.copy(),
        deskewed.reference_timestamp_ns, deskewed.frame_id,
    )
    rejected = 0
    if ghost_filter:
        keep = target_free_space_mask(refined, target_scan_id=anchor_scan_id,
                                      settings=FreeSpaceSettings())
        rejected = int(np.count_nonzero(~keep))
        refined = filter_cloud(refined, keep)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f'd1_{method}_refined.npz',
        xyz=refined.xyz, scan_id=refined.scan_id,
        time_offset_s=refined.time_offset_s, ring=refined.ring,
        source_return_index=refined.source_return_index,
        reference_timestamp_ns=np.int64(refined.reference_timestamp_ns),
    )
    np.savez_compressed(
        output_dir / f'd1_{method}_solved_poses.npz',
        scan_ids=solved.scan_ids, corrections=solved.corrections,
        centers=solved.centers,
        T_target_refined_from_target_deskewed=solved.transforms,
    )
    report = {
        'method': method,
        'data_source': 'raw_scan_GLIM_plus_independent_optimizer',
        'joint_or_pgo_optimization_executed': True,
        'original_internal_GLIM_optimizer_reproduced': False,
        'raw_archive_sha256': file_sha256(raw_archive),
        'input_scans': len(scans),
        'input_points': len(deskewed.xyz),
        'anchor_scan_id': anchor_scan_id,
        'target_timestamp_ns': timestamp,
        'd1_reproduction_roi_enabled': d1_roi,
        'iteration_reports': list(solved.iterations),
        'reference_comparison': comparison,
        'ghost_filter_enabled': ghost_filter,
        'ghost_filter_rejected_points': rejected,
        'output_points': len(refined.xyz),
        'notes': ('Rigid scan-level corrections on per-point deskewed clouds; '
                  'PGO is an anchored small-angle graph solve. '
                  'Optional ghost filtering detects target-scan free-space contradictions, '
                  'not verified ground-truth dynamic or ghost points.'),
    }
    (output_dir / f'd1_{method}_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-archive', required=True, type=Path)
    parser.add_argument('--reference-archive', type=Path,
                        help='Optional D1 output reference, compared only after solver execution')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--method', required=True, choices=SUPPORTED_METHODS)
    parser.add_argument('--ghost-filter', action='store_true')
    parser.add_argument('--anchor-scan-id', type=int, default=5)
    parser.add_argument('--disable-d1-roi', action='store_true',
                        help='Disable D1-specific ROI/wall exclusion (changes original experiment)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format='[%(levelname)s] %(message)s')
    result = run(args.raw_archive, args.output, method=args.method,
                 reference_archive=args.reference_archive, ghost_filter=args.ghost_filter,
                 anchor_scan_id=args.anchor_scan_id, d1_roi=not args.disable_d1_roi)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()