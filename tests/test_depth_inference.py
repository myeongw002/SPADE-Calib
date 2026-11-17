"""DA2 official API adapter tested with a fake backend; no GPU or model download in CI."""
from __future__ import annotations

import contextlib
import json
import sys
import types

import numpy as np
import pytest

pytest.importorskip('cv2')


def test_raw_inference_preserves_float_predictions_and_metadata(monkeypatch, tmp_path):
    import cv2
    from spade_calib.camera.depth_inference import infer_da2, save_inference
    repo = tmp_path / 'official_source'
    (repo / 'depth_anything_v2').mkdir(parents=True)
    (repo / 'depth_anything_v2' / 'dpt.py').write_text('# trusted test stub')
    image = tmp_path / '000105.png'
    cv2.imwrite(str(image), np.ones((32, 48, 3), dtype=np.uint8) * 128)
    ckpt = tmp_path / 'weights.pth'
    ckpt.write_bytes(b'test-stub-only-not-a-real-model')

    captured = {}
    class FakeDA2:
        def __init__(self, **config):
            captured['model_config'] = config

        def load_state_dict(self, weights, strict):
            assert weights == {}
            assert strict

        def to(self, device):
            assert device == 'cpu'
            return self

        def eval(self):
            return self

        def infer_image(self, bgr, input_size):
            assert bgr.shape == (32, 48, 3)
            assert input_size == 518
            return np.linspace(.2, 9.8, 32*48).reshape(32, 48)
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch.load = lambda path, map_location, weights_only: {}
    torch.inference_mode = contextlib.nullcontext
    fake_source = types.ModuleType('depth_anything_v2.dpt')
    fake_source.DepthAnythingV2 = FakeDA2
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'depth_anything_v2.dpt', fake_source)
    relative, meta = infer_da2(
        image, ckpt, encoder='vits', da2_repo=repo, device='cpu')
    assert relative.dtype == np.float32
    assert relative.shape == (32, 48)
    assert float(relative.min()) == pytest.approx(.2)
    assert float(relative.max()) == pytest.approx(9.8)
    assert meta['raw_prediction_unnormalized'] is True
    assert 'not_metric' in meta['output_kind']
    output, report = save_inference(tmp_path / 'raw.npy', relative, meta)
    np.testing.assert_array_equal(np.load(output, allow_pickle=False), relative)
    loaded = json.loads(report.read_text())
    assert loaded['encoder'] == 'vits'
    assert loaded['checkpoint_sha256'] != ''


def test_invalid_config_and_missing_checkpoint_fail_early(tmp_path):
    from spade_calib.camera.depth_inference import infer_da2
    with pytest.raises(ValueError, match='encoder must'):
        infer_da2('x.png', 'x.pth', encoder='invalid')
    with pytest.raises(ValueError, match='input_size'):
        infer_da2('x.png', 'x.pth', encoder='vits', input_size=0)
    with pytest.raises(FileNotFoundError):
        infer_da2(tmp_path / 'image_missing.png', tmp_path / 'checkpoint_missing.pth')
