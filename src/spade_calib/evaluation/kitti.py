"""Evaluation-only KITTI Odometry extrinsic reference.

NEVER pass this module's ground-truth transform into Hand-Eye fitting.
The KITTI Odometry calibration 'Tr:' maps Velodyne -> rectified Camera 0.
P0/P1/P2/P3 are rectified-camera projection matrices K[I | t].
Each rectified camera is assumed to have the same axes (KITTI Odometry).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def parse_kitti_odom_calib(path: str | Path) -> dict[str, NDArray[np.float64]]:
    """Read 3x4 P* and Tr matrices with explicit shape and finite validation."""
    tables: dict[str, NDArray[np.float64]] = {}
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            key, sep, payload = line.partition(":")
            if not sep or key.strip() in tables:
                raise ValueError("Invalid or duplicate KITTI calibration entry")
            parts = payload.split()
            if len(parts) != 12:
                raise ValueError(f"KITTI calibration {key} must contain 12 numbers")
            matrix = np.asarray([float(x) for x in parts], dtype=np.float64).reshape(3, 4)
            if not np.isfinite(matrix).all():
                raise ValueError(f"KITTI calibration {key} contains nonfinite values")
            tables[key.strip()] = matrix
    if not all(k in tables for k in ("P0", "Tr")):
        raise ValueError("KITTI Odometry calibration requires P0 and Tr")
    return tables


def kitti_gt_camera_lidar(path: str | Path, camera_index: int = 2
                          ) -> NDArray[np.float64]:
    """Return T_camera_lidar for rectified KITTI camera 0..3.

    Pk = K [I | -Ck] in Camera 0's rectified coordinate axes.
    Relative frame translation for Camera k <- Camera 0 is C0-Ck.
    If the input is NOT KITTI Odometry rectified P matrices, this assumption
    does not hold; use its full 3D camera-to-camera transform instead.
    """
    if type(camera_index) is not int or camera_index not in range(4):
        raise ValueError("camera_index must be an integer 0, 1, 2 or 3")
    cal = parse_kitti_odom_calib(path)
    key = f"P{camera_index}"
    if key not in cal:
        raise ValueError(f"KITTI calibration missing {key}")
    p0, pk = cal["P0"], cal[key]
    cam0 = -np.linalg.solve(p0[:, :3], p0[:, 3])
    camk = -np.linalg.solve(pk[:, :3], pk[:, 3])
    t_k_0 = cam0 - camk
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = cal["Tr"][:, :3]
    transform[:3, 3] = cal["Tr"][:, 3] + t_k_0
    return transform


def evaluate_camera_lidar(
    estimated_camera_lidar: NDArray[np.float64],
    gt_camera_lidar: NDArray[np.float64],
) -> dict[str, Any]:
    """Report Camera-frame rotation-vector and Translation errors.

    Rotation error is Log(R_est @ R_gt.T), a left Camera-frame rotation.
    Translation error is t_est - t_gt in Camera axes.
    """
    est, gt = (np.asarray(x, dtype=np.float64) for x in
               (estimated_camera_lidar, gt_camera_lidar))
    for x, name in ((est, "estimate"), (gt, "ground truth")):
        if x.shape != (4, 4) or not np.isfinite(x).all():
            raise ValueError(f"{name} must be a finite 4x4 matrix")
        if not np.allclose(x[3], [0, 0, 0, 1], atol=1e-8):
            raise ValueError(f"{name} is not a homogeneous transform")
        if not np.allclose(x[:3, :3].T @ x[:3, :3], np.eye(3), atol=1e-4):
            raise ValueError(f"{name} is not orthonormal")
        if np.linalg.det(x[:3, :3]) < 0:
            raise ValueError(f"{name} has an improper rotation")
    rotvec = np.rad2deg(Rotation.from_matrix(est[:3, :3] @ gt[:3, :3].T).as_rotvec())
    trans = est[:3, 3] - gt[:3, 3]
    return {
        "convention": "T_camera_lidar",
        "rotation_error_vector_camera_deg": rotvec.tolist(),
        "rotation_error_total_deg": float(np.linalg.norm(rotvec)),
        "translation_error_vector_camera_m": trans.tolist(),
        "translation_error_total_m": float(np.linalg.norm(trans)),
        "translation_est_camera_m": est[:3, 3].tolist(),
        "translation_gt_camera_m": gt[:3, 3].tolist(),
    }


def evaluate_handeye_json(
    result_path: str | Path, calib_path: str | Path,
    *, camera_index: int = 2,
) -> dict[str, Any]:
    """Evaluate a previously saved Hand-Eye result; GT is never fitted."""
    with Path(result_path).open("r", encoding="utf-8") as inp:
        doc = json.load(inp)
    if "handeye" not in doc or "T_camera_lidar" not in doc["handeye"]:
        raise ValueError("Expected a Hand-Eye report with T_camera_lidar")
    gt = kitti_gt_camera_lidar(calib_path, camera_index=camera_index)
    evaluated = evaluate_camera_lidar(
        np.asarray(doc["handeye"]["T_camera_lidar"], dtype=float), gt)
    evaluated["gt_source"] = "KITTI Odometry calib.txt"
    evaluated["camera_index"] = camera_index
    evaluated["gt_used_for_fitting"] = False
    return evaluated
