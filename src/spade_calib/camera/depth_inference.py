"""Depth Anything V2 relative-depth inference (official repository implementation).

The raw float32 HxW prediction is retained without visualization clipping,
per-image min/max normalization, inversion or an unsupported metric claim.
Official DA2 relative model predictions are treated as a *learned relative
inverse-depth proxy* only when subsequently fitted to metric SfM anchors.
External code and checkpoints must come from a trusted official checkout.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_npy(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.da2_',
                                         suffix='.npy', delete=False) as f:
            temp = Path(f.name)
            np.save(f, data, allow_pickle=False)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def infer_da2(image_path: str | Path, checkpoint: str | Path,
              *, encoder: str = 'vitl', input_size: int = 518,
              device: str = 'auto', da2_repo: str | Path | None = None,
              ) -> tuple[np.ndarray, dict[str, Any]]:
    """Run official DA2 relative model, no downloads and no metric conversion.

    da2_repo is an existing trusted checkout of Depth-Anything-V2.
    Checkpoints are locally loaded with weights_only=True.
    """
    if encoder not in _MODEL_CONFIGS:
        raise ValueError(f'encoder must be one of {sorted(_MODEL_CONFIGS)}')
    if type(input_size) is not int or input_size < 32 or input_size > 4096:
        raise ValueError('input_size must be an integer in [32,4096]')
    source, ckpt = Path(image_path).expanduser().resolve(strict=True), Path(
        checkpoint).expanduser().resolve(strict=True)
    if not source.is_file() or not ckpt.is_file():
        raise ValueError('Image and checkpoint must be existing regular files')
    if da2_repo is not None:
        root = Path(da2_repo).expanduser().resolve(strict=True)
        if not (root / 'depth_anything_v2' / 'dpt.py').is_file():
            raise ValueError('--da2-repo must be the official Depth-Anything-V2 checkout root')
        sys.path.insert(0, str(root))

    try:
        import cv2
        import torch
        DepthAnythingV2 = importlib.import_module('depth_anything_v2.dpt').DepthAnythingV2
    except ImportError as exc:
        raise RuntimeError(
            'Depth Anything V2 dependencies or official model source missing. '
            'Install the optional depth dependencies and provide --da2-repo '
            'pointing to a trusted Depth-Anything-V2 checkout.'
        ) from exc
    if device == 'auto':
        selected = ('cuda' if torch.cuda.is_available() else
                    'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        if device not in {'cpu', 'cuda', 'mps'}:
            raise ValueError('device must be auto, cpu, cuda or mps')
        selected = device
    if selected == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA selected but PyTorch cannot access a CUDA GPU')
    if selected == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS selected but unavailable')

    img = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if img is None or img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f'OpenCV could not decode an RGB/BGR image: {source}')
    model = DepthAnythingV2(**_MODEL_CONFIGS[encoder])
    try:
        weights = torch.load(str(ckpt), map_location='cpu', weights_only=True)
        model.load_state_dict(weights, strict=True)
    except (RuntimeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError('Checkpoint and encoder architecture are incompatible') from exc
    model = model.to(selected).eval()
    try:
        with torch.inference_mode():
            predicted = model.infer_image(img, input_size)
    finally:
        del model
    raw = np.asarray(predicted, dtype=np.float32)
    if raw.shape != img.shape[:2] or not np.isfinite(raw).all():
        raise ValueError('DA2 inference produced an invalid HxW depth array')
    if float(np.ptp(raw)) < 1e-8:
        raise ValueError('DA2 prediction is constant and cannot be metric-aligned')
    return raw, {
        'kind': 'depth_anything_v2_relative_prediction',
        'output_kind': 'relative_inverse_depth_proxy_not_metric',
        'encoder': encoder,
        'input_size': input_size,
        'device': selected,
        'image': str(source),
        'image_shape': [int(img.shape[0]), int(img.shape[1])],
        'checkpoint': str(ckpt),
        'checkpoint_sha256': _sha256(ckpt),
        'prediction_min': float(raw.min()),
        'prediction_max': float(raw.max()),
        'raw_prediction_unnormalized': True,
        'official_model_source': 'DepthAnything/Depth-Anything-V2',
        'note': 'DA2 relative checkpoint is not a metric camera-Z estimator.',
    }


def save_inference(output: str | Path, prediction: np.ndarray,
                   metadata: dict[str, Any]) -> tuple[Path, Path]:
    """Atomically save raw .npy prediction and its separate provenance JSON."""
    import json
    path = Path(output).expanduser().resolve()
    if path.suffix.lower() != '.npy':
        raise ValueError('DA2 output must be a .npy file')
    _atomic_npy(path, np.asarray(prediction, dtype=np.float32))
    report = path.with_suffix('.json')
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=report.parent,
                                         prefix='.da2_', suffix='.json',
                                         delete=False) as handle:
            tmp = Path(handle.name)
            json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(tmp, report)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return path, report
