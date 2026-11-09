"""Original D1 numerical regression (requires private experiment archive)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "reproduce_d1.py"
SPEC = importlib.util.spec_from_file_location("spade_reproduce_d1", SOURCE)
assert SPEC is not None and SPEC.loader is not None
REPRO = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPRO
SPEC.loader.exec_module(REPRO)


@pytest.mark.skipif(
    not os.environ.get("SPADE_D1_REPRO_ARCHIVE"),
    reason="SPADE_D1_REPRO_ARCHIVE not set: private D1 archive is not in public CI",
)
def test_original_d1_matching_and_calibration(tmp_path: Path) -> None:
    source = Path(os.environ["SPADE_D1_REPRO_ARCHIVE"])
    result = REPRO.reproduce(source, work_dir=tmp_path / "reproduction")
    assert result["status"] == "pass"
    assert result["methods"]["Joint"]["accepted_ids"] == [0, 6, 1, 5, 3, 19]
    assert result["methods"]["PGO"]["accepted_ids"] == [0, 6, 1, 5, 3]
    assert result["methods"]["Joint"]["calibration_csv"]["rows"] == 44
    assert result["methods"]["PGO"]["calibration_csv"]["rows"] == 40
