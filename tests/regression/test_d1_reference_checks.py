"""Fast synthetic tests: ensure baseline regression comparator detects changes."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "reproduce_d1.py"
SPEC = importlib.util.spec_from_file_location("spade_reproduce_d1", SOURCE)
assert SPEC is not None and SPEC.loader is not None
REPRO = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPRO
SPEC.loader.exec_module(REPRO)


def _write_csv(path: Path, data: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def test_compare_csv_allows_numerical_tolerance(tmp_path: Path) -> None:
    expected, actual = tmp_path / "expected.csv", tmp_path / "actual.csv"
    _write_csv(expected, [{"fold": 0, "method": "scene", "error": 0.1}])
    _write_csv(actual, [{"fold": 0, "method": "scene", "error": 0.10000001}])
    result = REPRO.compare_csv(expected, actual, atol=1e-6, rtol=1e-6)
    assert result["rows"] == 1
    assert result["max_abs_numeric_difference"] < 1e-6


def test_compare_csv_detects_method_identity_change(tmp_path: Path) -> None:
    expected, actual = tmp_path / "expected.csv", tmp_path / "actual.csv"
    _write_csv(expected, [{"fold": 0, "method": "plane_6d", "error": 0.1}])
    _write_csv(actual, [{"fold": 0, "method": "plane_8d", "error": 0.1}])
    with pytest.raises(REPRO.ReproductionError, match="method"):
        REPRO.compare_csv(expected, actual, atol=1e-6, rtol=1e-6)


def test_compare_csv_detects_extrinsic_error(tmp_path: Path) -> None:
    expected, actual = tmp_path / "expected.csv", tmp_path / "actual.csv"
    _write_csv(expected, [{"fold": 0, "translation_m": 0.5}])
    _write_csv(actual, [{"fold": 0, "translation_m": 0.52}])
    with pytest.raises(REPRO.ReproductionError, match="translation_m"):
        REPRO.compare_csv(expected, actual, atol=1e-6, rtol=1e-6)


def test_compare_npz_exact_support_ids(tmp_path: Path) -> None:
    expected, actual = tmp_path / "expected.npz", tmp_path / "actual.npz"
    np.savez(expected, C0_ids=np.array([1, 2, 7], dtype=np.int64))
    np.savez(actual, C0_ids=np.array([1, 3, 7], dtype=np.int64))
    with pytest.raises(REPRO.ReproductionError, match="IDs changed"):
        REPRO.compare_npz(expected, actual, atol=0, rtol=0, exact=True)


def test_compare_npz_extrinsic_tolerance(tmp_path: Path) -> None:
    expected, actual = tmp_path / "reference.npz", tmp_path / "actual.npz"
    original = np.eye(4)
    moved = original.copy()
    moved[0, 3] = 0.03
    np.savez(expected, Plane6D_fold0=original)
    np.savez(actual, Plane6D_fold0=moved)
    with pytest.raises(REPRO.ReproductionError, match="values changed"):
        REPRO.compare_npz(expected, actual, atol=1e-6, rtol=1e-6)


def test_archive_path_traversal_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as file:
        file.writestr("package/../../evil.py", "not safe")
    with pytest.raises(REPRO.ReproductionError, match="Unsafe ZIP member"):
        REPRO.unpack_trusted_archive(archive, tmp_path / "output")
    assert not (tmp_path / "evil.py").exists()
