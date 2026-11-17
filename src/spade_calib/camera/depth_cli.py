"""CLI adapters for Depth Anything V2 and SfM metric alignment."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from spade_calib.camera.depth_inference import _atomic_npy


def _write_report(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.depth_', suffix='.tmp',
                                         delete=False) as f:
            tmp = Path(f.name)
            json.dump(doc, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _handeye_scale(path: Path) -> float:
    with path.open('r', encoding='utf-8') as source:
        doc = json.load(source)
    h = doc.get('handeye')
    if not isinstance(h, dict) or h.get('convention') != (
            'T_camera_lidar (multiply lidar point to camera frame)'):
        raise ValueError('Expected a valid Hand-Eye result using T_camera_lidar convention')
    value = float(h['sfm_scale_to_metric'])
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Saved Hand-Eye SfM scale must be finite and positive')
    return value


def run_infer_depth(args: Any) -> int:
    from spade_calib.camera.depth_inference import infer_da2, save_inference
    predicted, info = infer_da2(
        args.image, args.checkpoint, encoder=args.encoder,
        input_size=args.input_size, device=args.device,
        da2_repo=args.da2_repo,
    )
    path, report = save_inference(args.output, predicted, info)
    print(json.dumps({'relative_depth_npy': str(path), 'report': str(report),
                      'shape': list(predicted.shape), 'device': info['device']}, indent=2))
    return 0


def run_align_depth(args: Any) -> int:
    from spade_calib.io.legacy import load_sfm_anchors, LegacySfMAnchors
    from spade_calib.camera.metric_alignment import (MetricAlignmentSettings,
                                                     align_relative_da2_to_sfm)
    from spade_calib.camera.planes import CameraGeometry
    from spade_calib.camera.sfm_anchors import build_sfm_metric_anchors

    raw_path = Path(args.relative_depth).expanduser().resolve(strict=True)
    raw = np.load(raw_path, allow_pickle=False)
    if not isinstance(raw, np.ndarray) or raw.ndim != 2:
        raise ValueError('Relative depth must be an HxW .npy numeric array')
    meta = raw_path.with_suffix('.json')
    if meta.is_file():
        info = json.loads(meta.read_text(encoding='utf-8'))
        if info.get('output_kind') not in (None, 'relative_inverse_depth_proxy_not_metric'):
            raise ValueError('Input depth provenance is not raw DA2 relative prediction')
    sfm_scale = _handeye_scale(args.handeye)
    anchor_source = {}
    if args.anchors is not None:
        anchors = load_sfm_anchors(args.anchors)
        anchor_source = {'anchors_npz': str(args.anchors.resolve())}
        if anchors.sfm_scale is not None and not np.isclose(
                anchors.sfm_scale, sfm_scale, rtol=1e-4, atol=0.):
            raise ValueError('Input SfM anchors scale differs from Hand-Eye estimated scale')
    else:
        if not args.image_name or args.camera_json is None:
            raise ValueError('--colmap-model requires --image-name and --camera-json')
        geometry = CameraGeometry.from_json(args.camera_json)
        if raw.shape != (geometry.height, geometry.width):
            raise ValueError('CameraGeometry must match raw DA2 HxW dimensions')
        fields, anchor_report = build_sfm_metric_anchors(
            args.colmap_model, args.image_name, geometry, sfm_scale,
            max_reprojection_error_px=args.max_reprojection_error_px,
        )
        anchors = LegacySfMAnchors(
            xy=fields['xy'], depth_m=fields['depth_m'],
            reprojection_error_px=fields['reprojection_error_px'],
            image_shape=tuple(map(int, fields['image_shape'])),
            sfm_scale=float(fields['sfm_scale']),
        )
        anchor_source = {'colmap_model': str(args.colmap_model.resolve()),
                         'colmap_image': args.image_name,
                         'colmap_anchors': anchor_report}

    config = MetricAlignmentSettings(
        max_reprojection_error_px=args.max_reprojection_error_px,
        max_holdout_median_relative_error=args.max_holdout_median_relative_error,
        min_train_anchors=args.min_train_anchors,
        min_holdout_anchors=args.min_holdout_anchors,
    )
    metric, report = align_relative_da2_to_sfm(raw, anchors, config)
    destination = Path(args.output).expanduser().resolve()
    if destination.suffix.lower() != '.npy':
        raise ValueError('Metric depth output must have .npy extension')
    report.update({
        'source_relative_da2': str(raw_path),
        'source_handeye': str(args.handeye.resolve()),
        'source_sfm_scale_to_metric': sfm_scale,
        'source_anchor': anchor_source,
        'metric_depth_output': str(destination),
    })
    _atomic_npy(destination, metric)
    _write_report(destination.with_suffix('.json'), report)
    # Preserve the exact anchor observations when built from COLMAP.
    if args.colmap_model is not None:
        np.savez_compressed(
            destination.with_name(destination.stem + '_anchors.npz'),
            **fields,
        )
    print(json.dumps({
        'metric_depth_npy': str(destination),
        'report': str(destination.with_suffix('.json')),
        'sfm_scale_to_metric': sfm_scale,
        'train_anchors': report['train_anchor_count'],
        'holdout_anchors': report['heldout_anchor_count'],
        'heldout_median_relative_error': report['heldout']['median_relative_error'],
        'valid_fraction': report['metric_valid_fraction'],
    }, indent=2))
    return 0
