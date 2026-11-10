"""Serializable reports for the whole-recording Hand-Eye estimator."""
from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile
from typing import Any

from .solver import HandEyeResult
from .uncertainty import analyze_local_uncertainty, block_bootstrap_uncertainty


def report_handeye(result: HandEyeResult, *, bootstrap_resamples: int = 0,
                   bootstrap_block_s: float = 20., bootstrap_seed: int = 20261110,
                   source_description: str | None = None) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "kind": "full_trajectory_handeye_uncertainty_not_physical_ground_truth",
        "source_description": source_description,
        "handeye": result.to_dict(),
        "observability_and_model_covariance": analyze_local_uncertainty(result),
        "bootstrap": (block_bootstrap_uncertainty(result, n_resamples=bootstrap_resamples,
                      block_duration_s=bootstrap_block_s, seed=bootstrap_seed)
                      if bootstrap_resamples else {"available": False, "reason": "Not requested"}),
    }
    return report


def write_handeye_report(report: dict[str, Any], path: str | Path) -> Path:
    dest = Path(path).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=dest.parent,
                                         prefix=".handeye_", suffix=".tmp", delete=False) as handle:
            tmp = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(tmp, dest)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return dest