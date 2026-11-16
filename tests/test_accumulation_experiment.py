"""Real-backend smoke tests for the KITTI 07 accumulation pilot.

These tests use the repository's spade_calib modules, not a fake Joint solver.
They do NOT validate KITTI 07 real-world registration accuracy.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "experiments"
    / "run_accumulation_experiment.py"
)
spec = importlib.util.spec_from_file_location("spade_kitti07_accumulation_pilot", SCRIPT)
assert spec is not None and spec.loader is not None
pilot = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pilot
spec.loader.exec_module(pilot)


def _synthetic_recording(root: Path, n: int = 12) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(422)
    times = np.arange(n, dtype=float) * 0.1
    odom = np.zeros((n, 8), dtype=float)
    odom[:, 0] = times + 1.0  # Synthetic GLIM timestamps differ from KITTI times.
    odom[:, 1] = np.arange(n) * 0.08
    odom[:, 7] = 1.0
    times_file = root / "times.txt"
    odom_file = root / "odom.txt"
    np.savetxt(times_file, times, fmt="%.9f")
    np.savetxt(odom_file, odom, fmt="%.9f")
    scans = root / "velodyne"
    scans.mkdir()
    # Static world planes, viewed from a sensor translating +x.
    for i in range(n):
        ground = np.column_stack((
            rng.uniform(2, 12, size=700),
            rng.uniform(-7, 7, size=700),
            rng.normal(-1.3, 0.008, size=700),
        ))
        wall = np.column_stack((
            rng.uniform(2, 12, size=450),
            rng.normal(6.0, 0.008, size=450),
            rng.uniform(-1.0, 2.0, size=450),
        ))
        world = np.vstack((ground, wall))
        body_xyz = world - odom[i, 1:4]
        scan = np.column_stack((body_xyz, np.ones(len(body_xyz))))
        scan.astype("<f4").tofile(scans / f"{i:06d}.bin")
    return times_file, odom_file, scans


def _args(tmp_path: Path, times_file: Path, odom_file: Path) -> list[str]:
    return [
        "--times", str(times_file), "--odom", str(odom_file),
        "--window", "synthetic", "--start-s", "0.1", "--stop-s", "1.0",
        "--target-frame", "6", "--half-frames", "2",
        "--radii-m", "0.17", "--out", str(tmp_path / "results"),
    ]


def test_index_mapping_and_selection(tmp_path: Path) -> None:
    times_file, odom_file, _ = _synthetic_recording(tmp_path)
    times, trajectory, meta = pilot.load_kitti_glim(times_file, odom_file)
    assert meta["mapping"].startswith("GLIM row")
    assert trajectory.timestamps[0] == pytest.approx(0.0)
    assert meta["glim_original_first_timestamp_s"] == pytest.approx(1.0)
    parsed = pilot.build_parser().parse_args(_args(tmp_path, times_file, odom_file))
    plans = pilot.prepare_experiments(parsed, times, trajectory, 0)
    assert len(plans) == 2
    assert plans[0].selected == (4, 5, 6, 7, 8)
    assert plans[1].selected == (4, 5, 6, 7, 8)
    assert pilot.travel_distance(trajectory.positions, np.asarray(plans[0].selected), 0) == pytest.approx(0.32)


def test_frame_offset_and_coverage_rejection(tmp_path: Path) -> None:
    times_file, odom_file, _ = _synthetic_recording(tmp_path)
    odom = np.loadtxt(odom_file)
    np.savetxt(odom_file, odom[:7])
    times, trajectory, info = pilot.load_kitti_glim(times_file, odom_file, 2)
    assert info["first_mapped_frame"] == 2
    assert info["last_mapped_frame"] == 8
    with pytest.raises(ValueError):
        pilot.load_kitti_glim(times_file, odom_file, 6)
    with pytest.raises(ValueError):
        pilot.select_fixed_distance(np.array([4, 5, 6]), 7, 1.0,
                                    trajectory.positions, position_offset=2)


def test_same_scan_budget(tmp_path: Path) -> None:
    times_file, odom_file, _ = _synthetic_recording(tmp_path)
    times, trajectory, _ = pilot.load_kitti_glim(times_file, odom_file)
    args = _args(tmp_path, times_file, odom_file) + ["--match-scan-budget"]
    parsed = pilot.build_parser().parse_args(args)
    plans = pilot.prepare_experiments(parsed, times, trajectory, 0)
    by_method = {p.method: p for p in plans}
    assert set(by_method) == {"fixed_frame", "fixed_distance", "distance_budget"}
    assert len(by_method["fixed_frame"].selected) == len(by_method["distance_budget"].selected)
    assert 6 in by_method["distance_budget"].selected


def test_ground_wall_quality_on_synthetic_points(tmp_path: Path) -> None:
    times_file, odom_file, _ = _synthetic_recording(tmp_path)
    args = pilot.build_parser().parse_args(_args(tmp_path, times_file, odom_file))
    rng = np.random.default_rng(27)
    ground = np.column_stack((
        rng.uniform(2, 14, 1000),
        rng.uniform(-8, 8, 1000),
        rng.normal(-1.3, 0.012, 1000),
    ))
    wall = np.column_stack((
        rng.uniform(2, 14, 650),
        rng.normal(6.0, 0.012, 650),
        rng.uniform(-1.0, 2.0, 650),
    ))
    xyz = np.vstack((ground, wall))
    scan_labels = np.arange(len(xyz)) % 6
    planes = pilot.extract_planes(xyz, scan_labels, args, experiment="unit", stage="initial")
    assert {"ground", "non_ground"}.issubset({p["kind"] for p in planes})
    assert all(p["rmse_m"] < 0.05 for p in planes)
    json.dumps(planes, allow_nan=False)


def test_real_backend_rigid_accumulation_cli(tmp_path: Path) -> None:
    times_file, odom_file, scans = _synthetic_recording(tmp_path)
    args = (_args(tmp_path, times_file, odom_file)
            + ["--velodyne", str(scans), "--refiner", "none",
               "--no-preview", "--ransac-iters", "150",
               "--min-plane-points", "90", "--max-walls", "1"])
    assert pilot.main(args) == 0
    out = tmp_path / "results"
    with (out / "summary.csv").open(encoding="utf-8-sig") as fh:
        summary = list(csv.DictReader(fh))
    assert len(summary) == 2
    assert all(r["status"] == "ok" for r in summary)
    assert all(int(r["scan_count"]) == 5 for r in summary)
    for row in summary:
        folder = out / row["experiment"]
        assert (folder / "initial_scan_colors.ply").exists()
        assert (folder / "initial_sample.npz").exists()
        assert (folder / "initial_planes.json").exists()
        assert not (folder / "refinement.json").exists()
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["experiments"]) == 2
    with (out / "plane_quality.csv").open(encoding="utf-8-sig") as fh:
        quality = list(csv.DictReader(fh))
    assert any(p["kind"] == "ground" for p in quality)
    assert any(p["kind"] == "non_ground" for p in quality)


def test_real_backend_pgo_accumulation_cli(tmp_path: Path) -> None:
    """Existing pilot accepts PGO and exports its diagnostics without Joint."""
    times_file, odom_file, scans = _synthetic_recording(tmp_path)
    args = (_args(tmp_path, times_file, odom_file)
            + ["--velodyne", str(scans), "--refiner", "pgo",
               "--pgo-iterations", "2", "--pgo-pair-max-nfev", "15",
               "--refine-voxel-m", ".20", "--refine-max-samples", "1200",
               "--no-preview", "--ransac-iters", "30",
               "--min-plane-points", "90", "--max-walls", "1"])
    assert pilot.main(args) == 0
    out = tmp_path / "results"
    with (out / "summary.csv").open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    for row in rows:
        assert row['status'] == 'ok'
        folder = out / row['experiment']
        info = json.loads((folder / 'refinement.json').read_text())
        assert info['solver'] == 'spade_calib.lidar.pgo_solver.solve_pgo'
        assert info['settings']['pairs']['d1_roi'] is False
        assert info['iterations'][0]['pair_edges'] >= 1
        assert int(row['registration_pair_edges_first']) >= 1
        assert (folder / 'refined_transforms.npy').exists()
        assert (folder / 'refined_sample.npz').exists()
        matrices = np.load(folder / 'refined_transforms.npy')
        np.testing.assert_allclose(matrices[int(row['anchor_local_scan_id'])],
                                   np.eye(4), atol=1e-8)
