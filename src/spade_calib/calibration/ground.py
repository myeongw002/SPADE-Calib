"""Camera/LiDAR plane conventions and sign-stable metric ground constraints.

Plane n.T @ p + d == 0, n unit length, d in metres. Every match describes
ONE physical surface in the LiDAR target frame and the simultaneous camera
frame; it must not mix world-frame or independently-scaled SfM planes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _vec3(x: object, label: str) -> NDArray[np.float64]:
    a = np.asarray(x, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f"{label} must be a finite 3-vector")
    return a


@dataclass(frozen=True)
class Plane:
    normal: NDArray[np.float64]
    offset_m: float

    def __post_init__(self) -> None:
        n = _vec3(self.normal, 'plane normal')
        d = float(self.offset_m)
        length = float(np.linalg.norm(n))
        if not np.isfinite(d) or length < 1e-10:
            raise ValueError('Plane must have a nonzero normal and finite metric offset')
        object.__setattr__(self, 'normal', n / length)
        object.__setattr__(self, 'offset_m', d / length)


@dataclass(frozen=True)
class GroundMatch:
    match_id: str
    lidar: Plane
    camera: Plane
    sigma_normal_deg: float = 3.0
    sigma_offset_m: float = 0.15

    def __post_init__(self) -> None:
        if not isinstance(self.match_id, str) or not self.match_id.strip():
            raise ValueError('Ground match id must be nonempty')
        for label, value in [('sigma_normal_deg', self.sigma_normal_deg),
                             ('sigma_offset_m', self.sigma_offset_m)]:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{label} must be positive and finite')
        if self.sigma_normal_deg >= 90:
            raise ValueError('Ground normal sigma must be smaller than 90 degrees')


def transform_plane_to_camera(plane: Plane, R_camera_lidar: NDArray,
                              t_camera_lidar: NDArray) -> Plane:
    """Transform LiDAR plane to Camera: nC=R*nL, dC=dL-(R*nL).t."""
    R = np.asarray(R_camera_lidar, dtype=np.float64)
    t = _vec3(t_camera_lidar, 'translation')
    if R.shape != (3, 3) or not np.isfinite(R).all():
        raise ValueError('Rotation must be finite 3x3')
    n = R @ plane.normal
    return Plane(n, float(plane.offset_m - n @ t))


def oriented_ground_residual(match: GroundMatch, R: NDArray, t: NDArray,
                             camera_sign: float) -> tuple[NDArray, float]:
    """Unweighted sign-consistent residual (normal cross product, offset metres)."""
    transformed = transform_plane_to_camera(match.lidar, R, t)
    nc = camera_sign * match.camera.normal
    dc = camera_sign * match.camera.offset_m
    return np.cross(transformed.normal, nc), float(transformed.offset_m - dc)


def prepare_matches(matches: list[GroundMatch], initial_rotation: NDArray,
                    *, max_initial_normal_angle_deg: float = 45.0
                    ) -> tuple[NDArray[np.float64], list[dict]]:
    """Resolve plane sign ONCE at initializer for differentiable refinement.

    Opposite normals represent the same plane; initial angle > gate implies
    uncertain correspondence. This is a safety gate, not correspondence GT.
    """
    if not matches:
        raise ValueError('At least one verified ground correspondence is required')
    if len({m.match_id for m in matches}) != len(matches):
        raise ValueError('Duplicate ground match identifiers')
    if not 0 < max_initial_normal_angle_deg < 90:
        raise ValueError('Initial ground-normal angle gate must be in (0, 90) degrees')
    signs, rows = [], []
    for m in matches:
        projected = np.asarray(initial_rotation, dtype=float) @ m.lidar.normal
        dot = float(np.clip(projected @ m.camera.normal, -1.0, 1.0))
        sign = 1.0 if dot >= 0 else -1.0
        angle = float(np.rad2deg(np.arccos(np.clip(abs(dot), 0, 1))))
        if angle > max_initial_normal_angle_deg:
            raise ValueError(f'{m.match_id}: initial ground normals differ by {angle:.2f} degrees; '
                             'verify matching planes, frames and depth scale')
        signs.append(sign)
        rows.append({'id': m.match_id, 'initial_normal_angle_deg': angle,
                     'camera_normal_sign': int(sign)})
    return np.asarray(signs, dtype=float), rows


def ground_residual_vector(matches: list[GroundMatch], signs: NDArray,
                           R: NDArray, t: NDArray, *,
                           normal_enabled: bool = True) -> NDArray[np.float64]:
    """Per-correspondence normalized residual. Averaged over matched planes."""
    if len(matches) != len(signs):
        raise ValueError('Ground matches/signs length mismatch')
    pieces = []
    normalization = np.sqrt(len(matches))
    for match, sign in zip(matches, signs):
        nerr, derr = oriented_ground_residual(match, R, t, float(sign))
        if normal_enabled:
            pieces.append(nerr / np.deg2rad(match.sigma_normal_deg) / normalization)
        pieces.append(np.array([derr / match.sigma_offset_m / normalization]))
    return np.concatenate(pieces)


def ground_metrics(matches: list[GroundMatch], signs: NDArray,
                   R: NDArray, t: NDArray) -> list[dict]:
    rows = []
    for m, sign in zip(matches, signs):
        nerr, off = oriented_ground_residual(m, R, t, float(sign))
        aligned = transform_plane_to_camera(m.lidar, R, t).normal @ (float(sign) * m.camera.normal)
        angle = np.rad2deg(np.arccos(np.clip(aligned, -1.0, 1.0)))
        rows.append({'id': m.match_id, 'normal_angle_deg': float(angle),
                     'offset_error_m': off, 'offset_abs_m': float(abs(off)),
                     'cross_norm': float(np.linalg.norm(nerr))})
    return rows


def load_ground_matches(path: str | Path) -> list[GroundMatch]:
    """Strict JSON interface. GT must never be read from this file."""
    with Path(path).open('r', encoding='utf-8') as src:
        doc = json.load(src)
    if not isinstance(doc, dict) or doc.get('schema_version') != 1:
        raise ValueError('Ground file requires schema_version=1')
    if doc.get('convention') != 'T_camera_lidar':
        raise ValueError('Ground file convention must be T_camera_lidar')
    data = doc.get('matches')
    if not isinstance(data, list) or not data:
        raise ValueError('Ground file requires nonempty matches')
    matches = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f'matches[{i}] is not an object')
        lid, cam = entry.get('lidar'), entry.get('camera')
        if not isinstance(lid, dict) or not isinstance(cam, dict):
            raise ValueError(f'matches[{i}] missing lidar/camera planes')
        if lid.get('frame') != 'lidar_target' or cam.get('frame') != 'camera':
            raise ValueError('Ground planes must use lidar_target and camera frames')
        matches.append(GroundMatch(
            match_id=entry['id'],
            lidar=Plane(lid['normal'], lid['offset_m']),
            camera=Plane(cam['normal'], cam['offset_m']),
            sigma_normal_deg=entry.get('sigma_normal_deg', 3.0),
            sigma_offset_m=entry.get('sigma_offset_m', 0.15),
        ))
    return matches
