"""Trajectory-only scene segmentation and Hand-Eye motion observability diagnostics.

Important: the information matrix constructed here is a *motion excitation proxy*,
not a statistically calibrated Fisher Information Matrix of the final plane/depth
calibration residuals. SfM translations have an unknown common metric scale.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence
import json
import math
import re

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp

from spade_calib.io.colmap import load_registered_images, RegisteredImage
from spade_calib.lidar.trajectory import GlimTrajectory

_NS = 1_000_000_000
_MAX_NS = (1 << 63) - 1


@dataclass(frozen=True)
class SelectionSettings:
    window_s: float = 30.0
    window_stride_s: float = 15.0
    pair_baseline_s: float = 2.0
    pair_stride_s: float = 1.0
    pair_tolerance_s: float = 0.12
    min_pairs: int = 10
    min_lidar_scans_per_window: int = 10
    max_lidar_gap_s: float = 1.5
    max_scenes: int = 4
    regularizer: float = 1e-3
    max_rotation_residual_deg: float = 15.0
    min_rotation_excitation_deg: float = 3.0
    min_translation_scale_sv: float = 0.025
    # Explicit scale for comparing [translation(m), log(SfM scale)] columns.
    translation_parameter_unit_m: float = 1.0

    def __post_init__(self) -> None:
        for field in ('window_s', 'window_stride_s', 'pair_baseline_s',
                      'pair_stride_s', 'pair_tolerance_s', 'regularizer',
                      'max_rotation_residual_deg', 'min_rotation_excitation_deg',
                      'min_translation_scale_sv', 'translation_parameter_unit_m', 'max_lidar_gap_s'):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{field} must be finite and positive')
        if self.pair_baseline_s >= self.window_s:
            raise ValueError('pair_baseline_s must be smaller than window_s')
        if self.min_pairs < 3 or self.max_scenes < 1 or self.min_lidar_scans_per_window < 2:
            raise ValueError('min_pairs >=3, max_scenes >=1, and min_lidar_scans_per_window >=2 required')


@dataclass(frozen=True)
class MatchedTrajectories:
    timestamps_ns: NDArray[np.int64]
    image_names: tuple[str, ...]
    lidar_R_world: NDArray[np.float64]
    lidar_t_world_m: NDArray[np.float64]
    camera_R_world_sfm: NDArray[np.float64]
    camera_t_world_sfm: NDArray[np.float64]
    total_registered: int
    discarded_outside_glim: int

    def __post_init__(self) -> None:
        n = len(self.timestamps_ns)
        if n < 5 or len(self.image_names) != n or self.timestamps_ns.shape != (n,):
            raise ValueError('Need >=5 synchronized registered images')
        for name, x, shape in (
            ('lidar_R_world', self.lidar_R_world, (n, 3, 3)),
            ('lidar_t_world_m', self.lidar_t_world_m, (n, 3)),
            ('camera_R_world_sfm', self.camera_R_world_sfm, (n, 3, 3)),
            ('camera_t_world_sfm', self.camera_t_world_sfm, (n, 3))):
            if x.shape != shape or not np.isfinite(x).all():
                raise ValueError(f'{name} must be finite {shape}')
        if np.any(np.diff(self.timestamps_ns) <= 0):
            raise ValueError('Registered image timestamps must strictly increase')


@dataclass(frozen=True)
class MotionPairs:
    Ra: NDArray[np.float64]
    ta: NDArray[np.float64]
    Rb: NDArray[np.float64]
    tb: NDArray[np.float64]
    indices: NDArray[np.int64]
    starts_ns: NDArray[np.int64]

    @property
    def count(self) -> int:
        return len(self.indices)


def _timestamp_from_name(name: str) -> int:
    stem = Path(name).stem
    if not re.fullmatch(r'[0-9]{17,20}', stem):
        raise ValueError(f'COLMAP image name must contain integer ns timestamp: {name!r}')
    value = int(stem)
    if value > _MAX_NS:
        raise ValueError('Timestamp exceeds int64')
    return value


def match_trajectories(
    glim: GlimTrajectory,
    sfm_records: Sequence[RegisteredImage],
    *,
    camera_to_lidar_offset_ns: int = 0,
    allowed_image_names: Collection[str] | None = None,
    image_timestamp_map: Mapping[str, int] | None = None,
) -> MatchedTrajectories:
    """Align GLIM positions to registered camera timestamps; do NOT align world frames.

    The supplied clock offset is added to the camera timestamp before querying
    GLIM. A COLMAP image pose is T_camera_sfm_world (wxyz, tvec).
    """
    if type(camera_to_lidar_offset_ns) is not int:
        raise ValueError('camera_to_lidar_offset_ns must be an integer')
    permitted = set(allowed_image_names) if allowed_image_names is not None else None
    registered = list(sfm_records)
    if permitted is not None:
        registered = [p for p in registered if p.name in permitted or Path(p.name).name in permitted]
    if not registered:
        raise ValueError('No registered camera poses within allowed images')
    # Explicit metadata takes precedence over filename parsing. This permits
    # frame-named (KITTI) and arbitrary custom recording images.
    supplied = ({str(k).replace('\\', '/').removeprefix('./'): v
                 for k, v in image_timestamp_map.items()}
                if image_timestamp_map is not None else None)
    basename_counts: dict[str, int] = {}
    for pose in registered:
        basename = Path(pose.name).name
        basename_counts[basename] = basename_counts.get(basename, 0) + 1
    timestamp_map: dict[int, RegisteredImage] = {}
    for pose in registered:
        if supplied is None:
            timestamp = _timestamp_from_name(pose.name)
        else:
            name = pose.name.replace('\\', '/').removeprefix('./')
            if name in supplied:
                timestamp = supplied[name]
            elif basename_counts[Path(name).name] == 1 and Path(name).name in supplied:
                timestamp = supplied[Path(name).name]
            else:
                raise ValueError(f'Missing or ambiguous SfM image timestamp: {pose.name}')
            if type(timestamp) is not int or not (0 <= timestamp <= _MAX_NS):
                raise ValueError(f'Invalid nanosecond timestamp for {pose.name}')
        if timestamp in timestamp_map:
            raise ValueError('Duplicate SfM image timestamp')
        timestamp_map[timestamp] = pose
    ordered = sorted(timestamp_map.items())
    orig_total = len(ordered)
    target_ns = [ts + camera_to_lidar_offset_ns for ts, _ in ordered]
    if any(ts < 0 or ts > _MAX_NS for ts in target_ns):
        raise ValueError('Camera-to-LiDAR offset causes timestamp overflow')
    # Ignore near-boundary samples to avoid float epoch precision extrapolation.
    min_ns = round(float(glim.timestamps[0]) * _NS) + 2_000
    max_ns = round(float(glim.timestamps[-1]) * _NS) - 2_000
    valid = [(raw_ts, pose, corrected_ns) for (raw_ts, pose), corrected_ns in zip(ordered, target_ns)
             if min_ns <= corrected_ns <= max_ns]
    if len(valid) < 5:
        raise ValueError('GLIM and COLMAP have <5 overlapping registered camera poses')
    stamps = np.asarray([v[2] for v in valid], dtype=np.int64)
    offsets = np.asarray([(int(t) - round(float(glim.timestamps[0]) * _NS)) * 1e-9
                          for t in stamps], dtype=np.float64)
    try:
        rL, tL = glim.at_offset(offsets)
    except ValueError as exc:
        raise ValueError('GLIM pose interpolation outside trajectory bounds') from exc
    q = np.array([v[1].qvec_wxyz for v in valid], dtype=np.float64)
    tvec = np.array([v[1].tvec for v in valid], dtype=np.float64)
    Rcw = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    Rwc = np.swapaxes(Rcw, 1, 2)
    twc = -np.einsum('nij,nj->ni', Rwc, tvec)
    return MatchedTrajectories(
        timestamps_ns=stamps,
        image_names=tuple(v[1].name for v in valid),
        lidar_R_world=np.asarray(rL, dtype=np.float64),
        lidar_t_world_m=np.asarray(tL, dtype=np.float64),
        camera_R_world_sfm=Rwc,
        camera_t_world_sfm=twc,
        total_registered=orig_total,
        discarded_outside_glim=orig_total - len(valid),
    )


def load_trajectory_data(glim_path: str | Path, sfm_path: str | Path,
                         *, camera_to_lidar_offset_ns: int = 0,
                         image_timestamps_csv: str | Path | None = None) -> MatchedTrajectories:
    """Trajectory-only fallback for archived data without full raw RGB/PCD.

    For full recordings, prefer match_trajectories with image-name allowlist
    from RecordingIndex, which additionally verifies actual data availability.
    """
    raw = np.loadtxt(glim_path, dtype=np.float64, ndmin=2)
    glim = GlimTrajectory.from_array(raw)
    poses = load_registered_images(sfm_path)
    from spade_calib.io.recording import read_timestamp_csv
    explicit = read_timestamp_csv(Path(image_timestamps_csv)) if image_timestamps_csv else None
    return match_trajectories(
        glim, tuple(poses.values()), camera_to_lidar_offset_ns=camera_to_lidar_offset_ns,
        image_timestamp_map=explicit,
    )


def from_recording_index(index: Any) -> MatchedTrajectories:
    """Use the PR#5 RecordingIndex; only poses backed by available RGB/PCD are considered."""
    allowed = {a.image.relative_path for a in index.associations
               if a.lidar is not None and a.sfm is not None and a.glim_covered}
    poses = [a.sfm for a in index.associations if a.lidar is not None and a.sfm is not None
             and a.glim_covered]
    if not allowed:
        raise ValueError('No camera frames have SfM pose and associated raw LiDAR')
    sfm_timestamps = {a.sfm.name: int(a.image.timestamp_ns)
                      for a in index.associations if a.lidar is not None and a.sfm is not None
                      and a.glim_covered}
    return match_trajectories(index.glim_trajectory, poses,
                              camera_to_lidar_offset_ns=index.config.camera_to_lidar_offset_ns,
                              image_timestamp_map=sfm_timestamps)


def _nearest(sorted_stamps: NDArray[np.int64], target_ns: int) -> int:
    k = int(np.searchsorted(sorted_stamps, target_ns))
    options = [i for i in (k - 1, k) if 0 <= i < len(sorted_stamps)]
    return min(options, key=lambda i: abs(int(sorted_stamps[i]) - target_ns))


def build_motion_pairs(samples: MatchedTrajectories, start_ns: int, stop_ns: int,
                       settings: SelectionSettings) -> MotionPairs:
    """Subsample non-adjacent relative motions on a fixed physical-time grid.

    Pair sets are consistent for all windows; non-independent pairs are not
    interpreted as independent statistical measurements.
    """
    times = samples.timestamps_ns
    baseline_ns = round(settings.pair_baseline_s * _NS)
    stride_ns = round(settings.pair_stride_s * _NS)
    tolerance_ns = round(settings.pair_tolerance_s * _NS)
    candidates = []
    end_start = stop_ns - baseline_ns
    for target in range(int(start_ns), int(end_start) + 1, stride_ns):
        i = _nearest(times, target)
        j = _nearest(times, target + baseline_ns)
        if i >= j or int(times[i]) < start_ns or int(times[j]) > stop_ns:
            continue
        if abs(int(times[i]) - target) > tolerance_ns:
            continue
        if abs(int(times[j]) - (target + baseline_ns)) > tolerance_ns:
            continue
        if candidates and (i, j) == candidates[-1]:
            continue
        candidates.append((i, j))
    if not candidates:
        return MotionPairs(np.empty((0,3,3)), np.empty((0,3)),
                           np.empty((0,3,3)), np.empty((0,3)),
                           np.empty((0,2), dtype=np.int64),np.empty(0,dtype=np.int64))
    ij = np.asarray(candidates, dtype=np.int64)
    i,j = ij[:,0],ij[:,1]
    L0,L1=samples.lidar_R_world[i],samples.lidar_R_world[j]
    C0,C1=samples.camera_R_world_sfm[i],samples.camera_R_world_sfm[j]
    Ra=np.einsum('nji,njk->nik',L0,L1)
    Rb=np.einsum('nji,njk->nik',C0,C1)
    ta=np.einsum('nji,nj->ni',L0,samples.lidar_t_world_m[j]-samples.lidar_t_world_m[i])
    tb=np.einsum('nji,nj->ni',C0,samples.camera_t_world_sfm[j]-samples.camera_t_world_sfm[i])
    return MotionPairs(Ra,ta,Rb,tb,ij,times[i])


def estimate_rotation_and_sfm_scale(pairs: MotionPairs) -> tuple[NDArray[np.float64], float, dict]:
    """Provisional T_camera_lidar Rotation/Scale for diagnostics, not calibration.

    The closed-form fit first estimates inverse T_lidar_camera then inverts it.
    Translation uses one global metric SfM scale.
    """
    if pairs.count < 3:
        raise ValueError('Need >=3 motion pairs for initial rotation/scale')
    a=Rotation.from_matrix(pairs.Ra).as_rotvec()
    b=Rotation.from_matrix(pairs.Rb).as_rotvec()
    u,sv,vh=np.linalg.svd(a.T@b)
    diag=np.diag([1.,1.,np.sign(np.linalg.det(u@vh))])
    R_LC=u@diag@vh
    # estimate tX and scale jointly; rank is explicitly reported
    G=np.column_stack([(pairs.Ra-np.eye(3)).reshape(-1,3),
                       -np.einsum('ij,nj->ni',R_LC,pairs.tb).reshape(-1)])
    sol,res,rank,singular=np.linalg.lstsq(G,-pairs.ta.reshape(-1),rcond=None)
    s=float(sol[-1])
    error=Rotation.from_matrix(pairs.Ra@R_LC@np.swapaxes(pairs.Rb,1,2)@R_LC.T).magnitude()
    R_CL = R_LC.T
    t_CL = -R_CL @ sol[:3]
    report={'rotation_alignment_singular_values':sv.tolist(),
            'rotation_residual_median_deg':float(np.rad2deg(np.median(error))),
            'rotation_residual_p95_deg':float(np.rad2deg(np.percentile(error,95))),
            'translation_scale_system_rank':int(rank),
            'provisional_sfm_scale_to_m':s,
            'scale_is_positive':bool(s>0),
            'provisional_t_camera_lidar_m':t_CL.tolist()}
    if not np.isfinite(s) or s <= 0:
        raise ValueError('Estimated SfM scale is nonpositive; verify pose conventions and time alignment')
    return R_CL,s,report


def _information(pairs: MotionPairs, R_X: NDArray[np.float64], s: float,
                 settings: SelectionSettings) -> NDArray[np.float64]:
    """Normalized block-diagonal motion information *proxy* for selection.

    Camera-coordinate Rotation block: (Rb-I); Camera Translation / log-scale:
    [(Rb-I)*L_param, s*tb]. Cross-block couplings omitted; no noise calibration.
    """
    if pairs.count < 1:
        return np.zeros((7,7),dtype=np.float64)
    J= pairs.Rb-np.eye(3)
    Hrot=np.einsum('nij,nik->jk',J,J)/pairs.count
    scale_col=pairs.tb*s
    G=np.concatenate((J*settings.translation_parameter_unit_m,
                      scale_col[:,:,None]),axis=2)
    Htrans=np.einsum('nij,nik->jk',G,G)/pairs.count
    H=np.zeros((7,7),dtype=np.float64)
    H[:3,:3]=Hrot
    H[3:,3:]=Htrans
    return (H+H.T)*.5


def _information_report(info: NDArray, settings: SelectionSettings) -> dict[str,Any]:
    rot_v, rot_vec=np.linalg.eigh(info[:3,:3]);trans_v,trans_vec=np.linalg.eigh(info[3:,3:])
    # eigenvalues of per-pair J^T J are squared excitation magnitudes
    rot_exc=np.rad2deg(np.sqrt(np.maximum(rot_v,0)))
    trans_exc=np.sqrt(np.maximum(trans_v,0))
    return {
        'rotation_excitation_deg':rot_exc.tolist(),
        'rotation_weak_axis_camera_xyz':rot_vec[:,0].tolist(),
        'rotation_weak_under_threshold':bool(rot_exc[0]<settings.min_rotation_excitation_deg),
        'translation_logscale_singular_values':trans_exc.tolist(),
        'translation_scale_weak_direction_camera_txyz_and_logscale':trans_vec[:,0].tolist(),
        'translation_scale_weak_under_threshold':bool(trans_exc[0]<settings.min_translation_scale_sv),
        'information_rank_at_1e-8':int(np.count_nonzero(np.linalg.eigvalsh(info)>1e-8)),
    }


def _score(info: NDArray, regularizer: float) -> float:
    sign,logdet=np.linalg.slogdet(np.eye(7)+info/regularizer)
    if sign <= 0 or not np.isfinite(logdet):
        raise ValueError('Invalid motion information matrix')
    return float(logdet)


def select_scenes(samples: MatchedTrajectories,
                  settings: SelectionSettings = SelectionSettings(),
                  *, lidar_scan_stamps_ns: Sequence[int] | None = None) -> dict[str,Any]:
    """Generate temporal candidates and greedily choose complementary disjoint windows.

    No scene's geometry is inspected. Candidates with insufficient motion pairs
    are rejected. Selected windows must not overlap in time.
    """
    first,last=map(int,(samples.timestamps_ns[0],samples.timestamps_ns[-1]))
    global_pairs=build_motion_pairs(samples,first,last,settings)
    if global_pairs.count < settings.min_pairs:
        raise ValueError('Insufficient overlapping GLIM/COLMAP motion pairs')
    R_X,s,init=estimate_rotation_and_sfm_scale(global_pairs)
    if init['rotation_residual_median_deg']>settings.max_rotation_residual_deg:
        raise ValueError('High cross-trajectory rotation inconsistency: check clocks and pose conventions')
    full_info=_information(global_pairs,R_X,s,settings)
    duration_ns=round(settings.window_s*_NS)
    stride_ns=round(settings.window_stride_s*_NS)
    candidates=[]
    for k,start in enumerate(range(first,last-duration_ns+1,stride_ns)):
        stop=start+duration_ns
        pair=build_motion_pairs(samples,start,stop,settings)
        if pair.count<settings.min_pairs:
            continue
        info=_information(pair,R_X,s,settings)
        within=(samples.timestamps_ns>=start)&(samples.timestamps_ns<=stop)
        available_lidar=None
        lidar_coverage_valid=None
        lidar_max_gap_s=None
        if lidar_scan_stamps_ns is not None:
            stamps=np.asarray(lidar_scan_stamps_ns,dtype=np.int64)
            window_scans=np.sort(stamps[(stamps>=start)&(stamps<=stop)])
            available_lidar=int(len(window_scans))
            # Include coverage gaps at the scene boundaries; a few scans
            # in a 30s scene are NOT evidence of complete raw data coverage.
            gap_stamps=np.r_[np.int64(start),window_scans,np.int64(stop)]
            lidar_max_gap_s=float(np.max(np.diff(gap_stamps).astype(np.float64))*1e-9)
            lidar_coverage_valid=bool(available_lidar>=settings.min_lidar_scans_per_window
                                      and lidar_max_gap_s<=settings.max_lidar_gap_s)
        candidates.append({
            'id':f'candidate_{k:03d}', 'start_ns':start,'stop_ns':stop,
            'duration_s':settings.window_s,'image_pose_count':int(np.sum(within)),
            'motion_pair_count':pair.count,'available_lidar_scan_count':available_lidar,
            'lidar_max_gap_s':lidar_max_gap_s,'lidar_coverage_valid':lidar_coverage_valid,
            'initial_score':_score(info,settings.regularizer),
            'diagnostics':_information_report(info,settings),
            '_info':info,
        })
    if not candidates:
        raise ValueError('No scene has sufficient motion pairs')
    selected=[];total=np.zeros((7,7),dtype=np.float64)
    while len(selected)<settings.max_scenes:
        before=_score(total,settings.regularizer)
        valid=[x for x in candidates if x['id'] not in {v['id'] for v in selected} and
               all(x['stop_ns']<=v['start_ns'] or x['start_ns']>=v['stop_ns'] for v in selected)]
        # Full RecordingIndex: only accept windows with continuous raw scan coverage.
        if lidar_scan_stamps_ns is not None:
            valid=[v for v in valid if v['lidar_coverage_valid']]
        if not valid:break
        ranked=sorted(valid,key=lambda x:(-_score(total+x['_info'],settings.regularizer),
                                          x['start_ns']))
        best=ranked[0].copy()
        best['marginal_gain']=_score(total+best['_info'],settings.regularizer)-before
        if best['marginal_gain']<=1e-10:break
        selected.append(best)
        total+=best['_info']
    def clean(row):
        return {key:value for key,value in row.items() if key!='_info'}
    return {
        'schema_version':1,
        'kind':'trajectory_motion_selection_not_calibration',
        'recording':{
            'registered_camera_poses_in_input':samples.total_registered,
            'matched_camera_glim_poses':len(samples.timestamps_ns),
            'discarded_outside_glim':samples.discarded_outside_glim,
            'start_ns':first,'stop_ns':last,
            'duration_s':(last-first)*1e-9,
            'global_motion_pairs':global_pairs.count,
        },
        'configuration':asdict(settings),
        'provisional_handeye':init,
        'full_recording_diagnostics':_information_report(full_info,settings),
        'selected_union_diagnostics':_information_report(total,settings),
        'selected_union_mean_per_scene_diagnostics':_information_report(
            total/max(len(selected),1), settings),
        'selected_union_score':_score(total,settings.regularizer),
        'candidates':[clean(c) for c in candidates],
        'selected':[clean(s) for s in selected],
        'selected_count':len(selected),
        'asset_coverage_checked':lidar_scan_stamps_ns is not None,
        'eligible_candidates_with_raw_scan_coverage':sum(c['lidar_coverage_valid'] is True for c in candidates) if lidar_scan_stamps_ns is not None else None,
        'selection_warnings':(['No candidate window has continuous raw LiDAR coverage; no executable scene selected']
                              if lidar_scan_stamps_ns is not None and not selected else [])
          + (['Weak rotational motion direction persists after scene selection']
             if selected and _information_report(total/len(selected), settings)['rotation_weak_under_threshold'] else []),
        'caveats':[
            'Motion information is a scale/weight-dependent proxy, NOT the final calibration Fisher information.',
            'Weak eigen-directions are parameter combinations, NOT independently missing Euler-axis DOFs.',
            'Provisional SfM scale/rotation are estimated using the full recording; this is NOT held-out evaluation.',
            'Trajectory excitation alone cannot confirm camera/LiDAR geometry correspondence or calibration accuracy.',
            'Selected windows are disjoint; correspondence quality and per-point deskew coverage remain unchecked.',
            'When raw LiDAR scan metadata is supplied, window scan-gap coverage is checked, but PCD bytes are not decoded.',
            'The selected union sums per-scene matrices; compare its mean-per-scene diagnostics with full-recording diagnostics to avoid count inflation.',
        ],
    }


def write_scene_selection(report: dict, dest: str | Path, *, index: Any | None = None) -> Path:
    """Write the selection JSON with image/scan paths when raw RecordingIndex exists."""
    dst=Path(dest).expanduser().resolve()
    if index is not None:
        for row in report['selected']:
            start,stop=row['start_ns'],row['stop_ns']
            row['image_files']=[str(x.path) for x in index.images if start<=x.timestamp_ns<=stop]
            row['lidar_scan_files']=[str(x.path) for x in index.lidar_scans if start<=x.timestamp_ns<=stop]
            row['files_verified_by_index']=True
    else:
        for row in report['selected']:
            row['files_verified_by_index']=False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp=dst.with_name('.'+dst.name+'.tmp')
    try:
        with tmp.open('w',encoding='utf-8') as out:
            json.dump(report,out,ensure_ascii=False,indent=2,allow_nan=False)
            out.write('\n')
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst