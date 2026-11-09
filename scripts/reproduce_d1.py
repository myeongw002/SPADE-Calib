#!/usr/bin/env python3
"""Reproduce original D1 plane matching and 6/8-DoF calibration from a private ZIP.

Executes original archived Python scripts, without modifying their algorithms.
Only use trusted research archives: this tool executes bundled source code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

import numpy as np

REQUIRED_SCRIPTS = ("run_camera_guided_refined.py", "run_calibration_refined.py")
DEFAULT_METHODS = ("Joint", "PGO")
MAX_ARCHIVE_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_ENTRY_BYTES = 256 * 1024 * 1024
DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-6


class ReproductionError(RuntimeError):
    """Legacy experiment cannot be reproduced or data are invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unpack_trusted_archive(archive_path: Path, destination: Path) -> None:
    """Extract one-root ZIP, disallowing traversal, symlinks and oversized archives."""
    with ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if not infos:
            raise ReproductionError("Empty archive")
        names = [entry.filename for entry in infos if not entry.is_dir()]
        prefixes = {Path(name).parts[0] for name in names if Path(name).parts}
        if len(prefixes) != 1 or any(len(Path(name).parts) < 2 for name in names):
            raise ReproductionError("Expected one top-level package directory")
        root_prefix = next(iter(prefixes))
        total_bytes = 0
        seen: set[str] = set()
        for info in infos:
            name = info.filename
            parts = Path(name).parts
            if (
                Path(name).is_absolute()
                or ".." in parts
                or "\\" in name
                or not parts
                or parts[0] != root_prefix
            ):
                raise ReproductionError(f"Unsafe ZIP member: {name!r}")
            entry_type = (info.external_attr >> 16) & 0o170000
            if entry_type == stat.S_IFLNK:
                raise ReproductionError(f"ZIP symlink rejected: {name!r}")
            if info.is_dir():
                continue
            relative = Path(*parts[1:])
            if not relative.parts or str(relative) in seen:
                raise ReproductionError(f"Duplicate ZIP member: {name!r}")
            seen.add(str(relative))
            total_bytes += info.file_size
            if total_bytes > MAX_ARCHIVE_TOTAL_BYTES or info.file_size > MAX_ENTRY_BYTES:
                raise ReproductionError("Archive exceeds unpack size limit")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
    missing = [name for name in REQUIRED_SCRIPTS if not (destination / name).is_file()]
    if missing:
        raise ReproductionError(f"Missing original scripts: {missing}")
    for method in DEFAULT_METHODS:
        for kind in ("matches", "calibration"):
            if not (destination / f"{kind}_{method}").is_dir():
                raise ReproductionError(f"Missing saved baseline: {kind}_{method}")


def _as_float(value: str) -> float | None:
    if not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _compare_numbers(
    expected: float, actual: float, *, atol: float, rtol: float, where: str
) -> float:
    if math.isnan(expected) and math.isnan(actual):
        return 0.0
    if not math.isclose(expected, actual, abs_tol=atol, rel_tol=rtol):
        raise ReproductionError(f"{where}: expected {expected!r}, actual {actual!r}")
    return abs(expected - actual)


def compare_csv(
    expected_path: Path, actual_path: Path, *, atol: float, rtol: float
) -> dict[str, Any]:
    with expected_path.open(newline="", encoding="utf-8") as source:
        expected = list(csv.DictReader(source))
    with actual_path.open(newline="", encoding="utf-8") as source:
        actual = list(csv.DictReader(source))
    if len(expected) != len(actual):
        raise ReproductionError(
            f"{expected_path.name}: row count {len(expected)} -> {len(actual)}"
        )
    if not expected:
        raise ReproductionError(f"{expected_path.name}: no reference data")
    if set(expected[0]) != set(actual[0]):
        raise ReproductionError(f"{expected_path.name}: CSV columns changed")
    max_delta = 0.0
    for row_index, (reference, reproduced) in enumerate(zip(expected, actual)):
        for key in reference:
            left, right = reference[key], reproduced[key]
            if left == right:
                continue
            left_num, right_num = _as_float(left), _as_float(right)
            if left_num is None or right_num is None:
                raise ReproductionError(
                    f"{expected_path.name} row={row_index}, {key}: {left!r} != {right!r}"
                )
            delta = _compare_numbers(
                left_num, right_num, atol=atol, rtol=rtol,
                where=f"{expected_path.name} row={row_index}, {key}",
            )
            max_delta = max(delta, max_delta)
    return {"rows": len(actual), "max_abs_numeric_difference": max_delta}


def compare_npz(
    expected_path: Path, actual_path: Path, *, atol: float, rtol: float,
    exact: bool = False,
) -> dict[str, Any]:
    max_delta = 0.0
    with np.load(expected_path, allow_pickle=False) as old, np.load(
        actual_path, allow_pickle=False
    ) as new:
        if set(old.files) != set(new.files):
            raise ReproductionError(f"{expected_path.name}: NPZ keys changed")
        for key in old.files:
            original, reproduced = old[key], new[key]
            if original.shape != reproduced.shape:
                raise ReproductionError(f"{expected_path.name}:{key}: array shape changed")
            if exact:
                if not np.array_equal(original, reproduced):
                    raise ReproductionError(f"{expected_path.name}:{key}: integer IDs changed")
            else:
                if not np.allclose(
                    original, reproduced, atol=atol, rtol=rtol, equal_nan=True
                ):
                    raise ReproductionError(f"{expected_path.name}:{key}: values changed")
                if original.size:
                    max_delta = max(
                        max_delta, float(np.nanmax(np.abs(original - reproduced)))
                    )
        count = len(old.files)
    return {"arrays": count, "max_abs_difference": max_delta}


def compare_match_summary(expected_path: Path, actual_path: Path) -> dict[str, Any]:
    reference = json.loads(expected_path.read_text(encoding="utf-8"))
    reproduced = json.loads(actual_path.read_text(encoding="utf-8"))
    fields = (
        "camera_planes", "lidar_cloud_size", "lidar_visible_points",
        "attempted_camera_guided_searches", "camera_guided_plane_found",
        "preliminary_gate_passes", "unique_accepted_matches", "accepted_ids",
    )
    for field in fields:
        if reference.get(field) != reproduced.get(field):
            raise ReproductionError(
                f"Matching summary changed: {field}={reference.get(field)!r} "
                f"-> {reproduced.get(field)!r}"
            )
    return {
        "camera_planes": reproduced["camera_planes"],
        "lidar_cloud_size": reproduced["lidar_cloud_size"],
        "accepted_ids": reproduced["accepted_ids"],
        "accepted_count": reproduced["unique_accepted_matches"],
    }


def compare_results(root: Path, method: str, *, atol: float, rtol: float) -> dict[str, Any]:
    ref_matching = root / f"reference_matches_{method}"
    matching = root / f"matches_{method}"
    ref_calib = root / f"reference_calibration_{method}"
    calib = root / f"calibration_{method}"
    summary = compare_match_summary(
        ref_matching / "summary.json", matching / "summary.json"
    )
    summary["matches_csv"] = compare_csv(
        ref_matching / "camera_guided_results.csv",
        matching / "camera_guided_results.csv", atol=atol, rtol=rtol,
    )
    summary["lidar_plane_parameters"] = compare_npz(
        ref_matching / "matched_lidar_plane_parameters.npz",
        matching / "matched_lidar_plane_parameters.npz", atol=atol, rtol=rtol,
    )
    summary["lidar_support_ids"] = compare_npz(
        ref_matching / "matched_lidar_support_indices.npz",
        matching / "matched_lidar_support_indices.npz",
        atol=0, rtol=0, exact=True,
    )
    summary["calibration_csv"] = compare_csv(
        ref_calib / "results.csv", calib / "results.csv", atol=atol, rtol=rtol,
    )
    summary["extrinsics"] = compare_npz(
        ref_calib / "extrinsics.npz", calib / "extrinsics.npz",
        atol=atol, rtol=rtol,
    )
    reference_status = json.loads((ref_calib / "status.json").read_text())
    actual_status = json.loads((calib / "status.json").read_text())
    for key in ("completed_folds", "num_rows"):
        if reference_status[key] != actual_status[key]:
            raise ReproductionError(f"{method}: calibration status {key} changed")
    summary["completed_folds"] = actual_status["completed_folds"]
    return summary


def _run_legacy(script: str, root: Path, method: str, *, timeout_sec: int) -> None:
    env = os.environ.copy()
    env.update({
        "REFINER": method,
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYTHONHASHSEED": "0",
    })
    log_path = root / f"reproduce_{method}_{Path(script).stem}.log"
    with log_path.open("w", encoding="utf-8") as log:
        try:
            result = subprocess.run(
                [sys.executable, script], cwd=root, env=env,
                stdout=log, stderr=subprocess.STDOUT,
                timeout=timeout_sec, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ReproductionError(
                f"{script} [{method}] timed out; see {log_path}"
            ) from exc
    if result.returncode != 0:
        tail = log_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-20:]
        raise ReproductionError(
            f"{script} [{method}] exited {result.returncode}; last log lines:\n"
            + "\n".join(tail)
        )


def reproduce(
    archive_path: Path, *, methods: Iterable[str] = DEFAULT_METHODS,
    work_dir: Path, atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL,
    timeout_sec: int = 180,
) -> dict[str, Any]:
    """Re-execute archived D1 scripts and compare against original ZIP outputs."""
    method_list = tuple(methods)
    if (
        not method_list
        or len(set(method_list)) != len(method_list)
        or (set(method_list) - set(DEFAULT_METHODS))
    ):
        raise ValueError("methods must be a nonempty subset of Joint, PGO")
    if atol < 0 or rtol < 0 or timeout_sec <= 0:
        raise ValueError("Invalid numerical tolerance or timeout")
    work_dir.mkdir(parents=True, exist_ok=True)
    if any(work_dir.iterdir()):
        raise ReproductionError(f"Working directory must be empty: {work_dir}")
    unpack_trusted_archive(archive_path, work_dir)
    for method in method_list:
        for kind in ("matches", "calibration"):
            (work_dir / f"{kind}_{method}").rename(
                work_dir / f"reference_{kind}_{method}"
            )
    results = {}
    for method in method_list:
        _run_legacy(
            "run_camera_guided_refined.py", work_dir, method,
            timeout_sec=timeout_sec,
        )
        _run_legacy(
            "run_calibration_refined.py", work_dir, method,
            timeout_sec=timeout_sec,
        )
        results[method] = compare_results(
            work_dir, method, atol=atol, rtol=rtol
        )
    import cv2
    import scipy
    return {
        "status": "pass",
        "archive_sha256": sha256_file(archive_path),
        "methods": results,
        "tolerance": {"absolute": atol, "relative": rtol},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
        "interpretation": (
            "Legacy code output reproducibility only; no extrinsic ground-truth "
            "accuracy or physical plane correspondence is established."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, required=True, help="Private original D1 ZIP"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("d1_reproduction_report.json")
    )
    parser.add_argument(
        "--methods", nargs="+", choices=DEFAULT_METHODS,
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument(
        "--keep-workdir", type=Path,
        help="Keep private data/logs in this NEW directory; default temporary",
    )
    args = parser.parse_args(argv)
    try:
        archive_path = args.archive.expanduser().resolve(strict=True)
        started = time.monotonic()
        if args.keep_workdir is not None:
            directory = args.keep_workdir.expanduser().resolve()
            if directory.exists():
                raise ReproductionError("--keep-workdir directory already exists")
            report = reproduce(
                archive_path, methods=args.methods, work_dir=directory,
                atol=args.atol, rtol=args.rtol, timeout_sec=args.timeout_sec,
            )
            report["private_workdir"] = str(directory)
        else:
            with tempfile.TemporaryDirectory(prefix="spade_d1_repro_") as temporary:
                report = reproduce(
                    archive_path, methods=args.methods,
                    work_dir=Path(temporary), atol=args.atol,
                    rtol=args.rtol, timeout_sec=args.timeout_sec,
                )
        report["runtime_sec"] = round(time.monotonic() - started, 3)
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, ReproductionError) as exc:
        print(f"D1 reproduction FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"D1 reproduction PASS: {args.output}")
    for method, result in report["methods"].items():
        print(
            f"  {method}: matches={result['accepted_ids']} "
            f"rows={result['calibration_csv']['rows']} "
            f"extrinsics={result['extrinsics']['arrays']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
