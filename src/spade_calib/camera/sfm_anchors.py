"""Build metric Camera-Z SfM anchors from a COLMAP sparse reconstruction.

Reads images.txt/bin and points3D.txt/bin, retains the target image's
actual tracked 2D observations, projects corresponding SfM world points
with the COLMAP world->camera pose, and checks K/distortion reprojection.
The SfM metric scale is supplied externally by saved Hand-Eye (NOT KITTI GT).

COLMAP sparse camera pixels must be in the same *rectified/unrectified image
coordinate system* as the provided CameraGeometry and Depth Anything image.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from spade_calib.camera.planes import CameraGeometry
from spade_calib.io.colmap import load_registered_images, resolve_colmap_images


def _read(stream, amount: int) -> bytes:
    data = stream.read(amount)
    if len(data) != amount:
        raise ValueError('Truncated COLMAP binary reconstruction')
    return data


def _image_observations(images: Path, image_id: int
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Return target-image (N,2) feature xy and 3D IDs, with -1 for untracked."""
    if images.suffix == '.txt':
        with images.open('r', encoding='utf-8') as source:
            while True:
                line = source.readline()
                if not line:
                    break
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                fields = line.split(maxsplit=9)
                if len(fields) != 10:
                    raise ValueError('Malformed images.txt image record')
                observed = source.readline()
                if observed == '':
                    raise ValueError('Missing COLMAP images.txt observations line')
                if int(fields[0]) != image_id:
                    continue
                tokens = observed.split()
                if len(tokens) % 3:
                    raise ValueError('COLMAP images.txt observations are not triples')
                if not tokens:
                    return np.empty((0, 2)), np.empty(0, dtype=np.int64)
                values = np.asarray(tokens, dtype=float).reshape(-1, 3)
                return values[:, :2], values[:, 2].astype(np.int64)
        raise ValueError('Target image not found in images.txt')

    fixed = struct.Struct('<i4d3di')
    with images.open('rb') as source:
        num_images = struct.unpack('<Q', _read(source, 8))[0]
        if num_images > 10_000_000:
            raise ValueError('Implausible COLMAP image count')
        for _ in range(num_images):
            head = fixed.unpack(_read(source, fixed.size))
            while _read(source, 1) != b'\0':
                pass
            n = struct.unpack('<Q', _read(source, 8))[0]
            if n > (images.stat().st_size - source.tell()) // 24:
                raise ValueError('Truncated COLMAP 2D observations')
            if head[0] == image_id:
                raw = np.frombuffer(_read(source, n*24),
                                    dtype=[('x', '<f8'), ('y', '<f8'), ('id', '<i8')])
                return np.column_stack([raw['x'], raw['y']]), raw['id'].copy()
            source.seek(n*24, 1)
    raise ValueError('Target image not found in images.bin')


def _sparse_points(path: Path, selected_ids: set[int], min_track_length: int
                   ) -> dict[int, tuple[np.ndarray, float, int]]:
    """Read only the requested sparse point IDs, without storing all points."""
    result: dict[int, tuple[np.ndarray, float, int]] = {}
    if path.suffix == '.txt':
        with path.open('r', encoding='utf-8') as stream:
            for line in stream:
                if not line.strip() or line.lstrip().startswith('#'):
                    continue
                cols = line.split()
                if len(cols) < 8:
                    raise ValueError('Malformed COLMAP points3D.txt record')
                pid = int(cols[0])
                if pid in selected_ids and (len(cols)-8)//2 >= min_track_length:
                    result[pid] = (np.asarray(cols[1:4], dtype=np.float64),
                                   float(cols[7]), (len(cols)-8)//2)
        return result

    fixed = struct.Struct('<QdddBBBdQ')
    with path.open('rb') as stream:
        total = struct.unpack('<Q', _read(stream, 8))[0]
        if total > 500_000_000:
            raise ValueError('Implausible COLMAP points3D count')
        for _ in range(total):
            record = fixed.unpack(_read(stream, fixed.size))
            pid, track_length = int(record[0]), int(record[-1])
            if track_length > (path.stat().st_size - stream.tell()) // 8:
                raise ValueError('Truncated COLMAP point track')
            if pid in selected_ids and track_length >= min_track_length:
                result[pid] = (np.asarray(record[1:4], dtype=np.float64),
                               float(record[-2]), track_length)
            stream.seek(track_length * 8, 1)
    return result


def _find_points_file(image_file: Path) -> Path:
    root = image_file.parent
    for suffix in ('.bin', '.txt'):
        p = root / ('points3D' + suffix)
        if p.is_file():
            return p
    raise FileNotFoundError(f'COLMAP points3D.bin/txt missing in {root}')


def build_sfm_metric_anchors(model_path: str | Path, image_name: str,
                             geometry: CameraGeometry, sfm_scale_to_metric: float,
                             *, max_reprojection_error_px: float = 3.0,
                             min_track_length: int = 2, max_anchors: int = 50000
                             ) -> tuple[dict, dict]:
    """Project sparse world points to the selected image and return anchor arrays.

    Features with 3D_id=-1 are excluded. A distinct sparse model, different
    camera preprocessing or a misapplied metric scale must be rejected by
    diagnostics rather than silently used for metric calibration.
    """
    if not np.isfinite(sfm_scale_to_metric) or sfm_scale_to_metric <= 0:
        raise ValueError('SfM scale must be positive and finite')
    if not np.isfinite(max_reprojection_error_px) or max_reprojection_error_px <= 0:
        raise ValueError('Reprojection limit must be positive')
    if min_track_length < 2 or max_anchors < 10:
        raise ValueError('Invalid track length or anchor count')
    image_file = resolve_colmap_images(Path(model_path))
    if image_file.name == 'camera_poses.csv':
        raise ValueError('Anchor extraction requires real COLMAP images.bin/txt with 2D tracks')
    images = load_registered_images(image_file)
    key = image_name.replace('\\', '/').removeprefix('./')
    if key not in images:
        candidates = [k for k in images if Path(k).name == Path(key).name]
        if len(candidates) != 1:
            raise ValueError(f'Image name {image_name!r} is missing or ambiguous in COLMAP')
        key = candidates[0]
    record = images[key]
    xy, ids = _image_observations(image_file, record.image_id)
    tracked = ids >= 0
    unique_ids = set(map(int, ids[tracked]))
    if not unique_ids:
        raise ValueError(f'COLMAP target image {key} has no reconstructed 3D tracks')
    points = _sparse_points(_find_points_file(image_file), unique_ids, min_track_length)
    image_ids = np.flatnonzero(tracked & np.isin(ids, list(points.keys())))
    if not len(image_ids):
        raise ValueError('No target-image points survived COLMAP track length filter')
    xyz = np.asarray([points[int(ids[k])][0] for k in image_ids])
    observation_xy = xy[image_ids]
    q = np.asarray(record.qvec_wxyz)
    Rcw = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    camera_xyz = xyz @ Rcw.T + np.asarray(record.tvec)
    z = camera_xyz[:, 2] * sfm_scale_to_metric
    import cv2
    projected, _ = cv2.projectPoints(
        camera_xyz.reshape(-1, 1, 3).astype(np.float64),
        np.zeros(3), np.zeros(3), geometry.K, geometry.distortion)
    reproj = np.linalg.norm(projected[:, 0, :] - observation_xy, axis=1)
    good = (
        np.isfinite(z) & (z > 0) & np.isfinite(observation_xy).all(axis=1) &
        np.isfinite(reproj) & (reproj <= max_reprojection_error_px) &
        (observation_xy[:, 0] >= 0) & (observation_xy[:, 0] < geometry.width) &
        (observation_xy[:, 1] >= 0) & (observation_xy[:, 1] < geometry.height)
    )
    good_ids = np.flatnonzero(good)
    if len(good_ids) > max_anchors:
        # Stable deterministic downsample, without requiring all sparse tracks.
        good_ids = good_ids[np.linspace(0, len(good_ids)-1, max_anchors, dtype=int)]
    if len(good_ids) < 15:
        raise ValueError(
            f'Only {len(good_ids)} anchors passed COLMAP/K reprojection gate. '
            'Check camera rectification, sparse reconstruction and SfM scale.'
        )
    return {
        'xy': observation_xy[good_ids],
        'depth_m': z[good_ids],
        'reprojection_error_px': reproj[good_ids],
        'image_shape': np.asarray([geometry.height, geometry.width], dtype=np.int64),
        'sfm_scale': np.asarray(sfm_scale_to_metric, dtype=np.float64),
    }, {
        'kind': 'colmap_sfm_metric_anchors_not_ground_truth',
        'image_name': key,
        'image_id': record.image_id,
        'camera_id': record.camera_id,
        'source_images': str(image_file),
        'source_points3D': str(_find_points_file(image_file)),
        'sfm_scale_to_metric': float(sfm_scale_to_metric),
        'feature_count': int(len(ids)),
        'tracked_feature_count': int(tracked.sum()),
        'valid_track_count': int(len(image_ids)),
        'accepted_metric_anchors': int(len(good_ids)),
        'median_reprojection_error_px': float(np.median(reproj[good_ids])),
        'max_reprojection_error_gate_px': float(max_reprojection_error_px),
        'image_geometry_matches_colmap_reconstruction_assumed': True,
        'gt_used_for_fitting': False,
    }
