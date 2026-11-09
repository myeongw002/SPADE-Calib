#!/usr/bin/env python3
"""D1: raw VLP16 + GLIM -> deskew A -> optional B/C1/target free-space filtering.

No private data is tracked in Git. The historical Joint/PGO trajectory is
NOT recovered by this runner; B/C1 is the archived separate ablation method.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from spade_calib.lidar import (load_d1_scan_zip, accumulate_glim,
                              target_free_space_mask, filter_cloud, FreeSpaceSettings)


def run(archive: Path, output_dir: Path, *, variant: str, ghost_filter: bool,
        target_scan_id: int = 5, max_points: int = 0) -> dict:
    if variant not in {'A', 'B', 'C1'}:
        raise ValueError('variant must be A, B or C1')
    scans, trajectory, manifest = load_d1_scan_zip(archive)
    anchor_ns = int(manifest['lidar_anchor_timestamp_ns'])
    if not 0 <= target_scan_id < len(scans):
        raise ValueError('Invalid target scan id')
    # Ghost-free-space reference must be the actual target scan's LiDAR frame.
    if ghost_filter and scans[target_scan_id].timestamp_ns != anchor_ns:
        raise ValueError('Ghost filtering requires target_scan_id corresponding to anchor time')
    cloud = accumulate_glim(scans, trajectory, target_timestamp_ns=anchor_ns, mode='deskew')
    info = {'variant': variant, 'scans': len(scans), 'raw_points': len(cloud.xyz),
            'target_timestamp_ns': anchor_ns, 'ghost_filter_enabled': ghost_filter,
            'source_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    if variant != 'A':
        from spade_calib.lidar.pose_refinement import refine_d1_b, apply_d1_corrections
        refinement = refine_d1_b(cloud, verbose=True)
        transformed = apply_d1_corrections(cloud, scans, refinement, variant)
        # Preserve sample identity; only x,y,z change.
        cloud = type(cloud)(transformed.astype(np.float32), cloud.scan_id,
              cloud.time_offset_s, cloud.ring, cloud.source_return_index,
              cloud.reference_timestamp_ns, cloud.frame_id)
        info['corrections'] = refinement.corrections.tolist()
        info['history'] = list(refinement.history)
    if ghost_filter:
        keep = target_free_space_mask(cloud, target_scan_id=target_scan_id,
                          settings=FreeSpaceSettings())
        info['filtered_points'] = int(np.count_nonzero(~keep))
        cloud = filter_cloud(cloud, keep)
    info['output_points'] = len(cloud.xyz)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f'd1_{variant}_points.npz', xyz=cloud.xyz,
         scan_id=cloud.scan_id, time_offset_s=cloud.time_offset_s,
         ring=cloud.ring, source_return_index=cloud.source_return_index,
         reference_timestamp_ns=np.int64(cloud.reference_timestamp_ns))
    (output_dir / f'd1_{variant}_report.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--variant', choices=['A', 'B', 'C1'], default='A')
    parser.add_argument('--ghost-filter', action='store_true')
    parser.add_argument('--target-scan-id', type=int, default=5)
    args = parser.parse_args()
    result = run(args.archive, args.output, variant=args.variant,
                 ghost_filter=args.ghost_filter, target_scan_id=args.target_scan_id)
    print(json.dumps({k: v for k,v in result.items() if k not in {'corrections','history'}}, indent=2))


if __name__ == '__main__':
    main()
