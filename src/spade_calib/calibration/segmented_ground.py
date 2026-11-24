"""Fixed-support ground + multi-scale Hand-Eye diagnostic (B4).

One shared T_camera_lidar. The camera ground planes are metric at the
reference segment scales. A changed scale multiplies their offsets by
s_new / s_reference; normals are unchanged. No GT is used by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from spade_calib.calibration.ground import Plane
from spade_calib.handeye.residuals import residuals
from spade_calib.handeye.solver import HandEyeSettings
from spade_calib.selection.scale_segmentation import ScalePartition, validate_extrinsic
from spade_calib.selection.trajectory import MotionPairs
from spade_calib.lidar.ground_planes import LidarGroundSettings, extract_lidar_ground


@dataclass(frozen=True)
class SegmentGround:
    segment_id: int
    lidar: Plane
    camera: Plane  # metric at the saved HE scale, positive-up orientation


@dataclass(frozen=True)
class GroundSettings:
    min_depth_m: float = 3.
    max_depth_m: float = 25.
    half_width_m: float = 3.
    min_height_m: float = .35
    max_height_m: float = 3.5
    max_tilt_deg: float = 15.
    max_match_angle_deg: float = 10.
    max_match_offset_m: float = 1.
    threshold_m: float = .12
    max_rmse_m: float = .12
    grid_m: float = 1.5
    min_common_cells: int = 6
    min_sfm_points: int = 25
    min_lidar_points: int = 100
    min_minor_extent_m: float = .30

    def __post_init__(self):
        values = [self.min_depth_m, self.max_depth_m, self.half_width_m,
                  self.min_height_m, self.max_height_m, self.max_tilt_deg,
                  self.max_match_angle_deg, self.max_match_offset_m, self.threshold_m,
                  self.max_rmse_m, self.grid_m, self.min_minor_extent_m]
        if (not np.isfinite(values).all() or min(values) <= 0
                or self.max_depth_m <= self.min_depth_m
                or self.max_height_m <= self.min_height_m
                or not 0 < self.max_tilt_deg < 80
                or not 0 < self.max_match_angle_deg < 90
                or min(self.min_common_cells, self.min_sfm_points, self.min_lidar_points) < 3):
            raise ValueError("Invalid ground settings")


def extract_segment_ground(camera_xyz_m: np.ndarray, lidar_xyz_m: np.ndarray,
                           X0: np.ndarray, segment_id: int,
                           settings: GroundSettings = GroundSettings()):
    """Fit below-sensor near-forward ground independently in each modality.

    HE guides only the spatial ROI/common horizontal grid; it never supplies
    a target plane normal/offset to either fit. Road/sidewalk identity is
    still provisional without independent semantic/physical verification.
    """
    X = validate_extrinsic(X0)
    R, t = X[:3, :3], X[:3, 3]
    c = np.asarray(camera_xyz_m, dtype=float)
    l = np.asarray(lidar_xyz_m, dtype=float)
    if any(p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() for p in (c,l)):
        raise ValueError("Expected finite Nx3 clouds")
    cfg = settings
    def forward(p):
        return ((p[:, 2] >= cfg.min_depth_m) & (p[:, 2] <= cfg.max_depth_m)
                & (abs(p[:, 0]) <= cfg.half_width_m))
    c = c[forward(c) & (c[:,1] >= cfg.min_height_m) & (c[:,1] <= cfg.max_height_m)]
    l = l[forward(l @ R.T + t) & (-l[:,2] >= cfg.min_height_m)
          & (-l[:,2] <= cfg.max_height_m)]
    diag = {"segment_id": segment_id, "camera_roi_points": len(c), "lidar_roi_points": len(l)}
    def fit(points, up, count):
        opts = LidarGroundSettings(voxel_m=.08, min_range_m=0.,
            max_range_m=cfg.max_depth_m + cfg.half_width_m + cfg.max_height_m + 5,
            min_height_m=cfg.min_height_m, max_height_m=cfg.max_height_m,
            normal_max_tilt_deg=cfg.max_tilt_deg, inlier_threshold_m=cfg.threshold_m,
            refit_threshold_m=cfg.threshold_m, min_inliers=count)
        result, _ = extract_lidar_ground(points, opts, lidar_up=np.array(up, dtype=float))
        return result.plane, result.xyz[result.inlier_mask]
    def cells(p):
        return np.floor(p[:,[0,2]] / cfg.grid_m).astype(np.int64)
    def keys(p):
        return set(map(tuple, cells(p)))
    def common_support(pc, pl):
        return keys(pc) & keys(pl @ R.T + t)
    def subset(p, projected, common):
        return p[np.array([tuple(row) in common for row in cells(projected)], dtype=bool)]
    try:
        _, cp = fit(c, [0,-1,0], cfg.min_sfm_points)
        _, lp = fit(l, [0,0,1], cfg.min_lidar_points)
        common = common_support(cp, lp)
        diag["initial_common_cells"] = len(common)
        if len(common) < cfg.min_common_cells:
            raise ValueError("insufficient common horizontal support")
        # Refit using independent raw ROI points in the same horizontal cells,
        # not distances to the opposite sensor's fitted plane.
        cplane, cp = fit(subset(c,c,common), [0,-1,0], cfg.min_sfm_points)
        lplane, lp = fit(subset(l,l @ R.T+t,common), [0,0,1], cfg.min_lidar_points)
        common_final = common_support(cp,lp)
        diag["final_common_cells"] = len(common_final)
        if len(common_final) < cfg.min_common_cells:
            raise ValueError("insufficient final common support")
        def metrics(plane,p):
            sv = np.linalg.svd(p-p.mean(0), compute_uv=False) / np.sqrt(len(p))
            return {"inliers": len(p), "rmse_m": float(np.sqrt(np.mean(
                    (p @ plane.normal + plane.offset_m)**2))),
                    "minor_extent_m": float(sv[1]), "normal": plane.normal.tolist(),
                    "offset_m": plane.offset_m}
        diag["camera"], diag["lidar"] = metrics(cplane,cp), metrics(lplane,lp)
        if any(v["rmse_m"] > cfg.max_rmse_m or v["minor_extent_m"] < cfg.min_minor_extent_m
               for v in (diag["camera"], diag["lidar"])):
            raise ValueError("poor planar support or thickness")
        nl = R @ lplane.normal
        angle = float(np.rad2deg(np.arccos(np.clip(cplane.normal @ nl,-1.,1.))))
        offset = float(cplane.offset_m - lplane.offset_m + nl @ t)
        diag.update(initial_normal_angle_deg=angle, initial_offset_error_m=offset)
        if angle > cfg.max_match_angle_deg or abs(offset) > cfg.max_match_offset_m:
            raise ValueError("initial cross-modal normal/offset gate")
    except ValueError as exc:
        diag.update(accepted=False, reason=str(exc))
        return None, diag, None, None
    diag.update(accepted=True, reason="provisional_geometry_ground_candidate")
    return SegmentGround(segment_id,lplane,cplane), diag, cp, lp


def ground_residuals(matches, X, scales, reference_scales):
    """Signed positive-up normal difference and metric offset, shape (M,4)."""
    rows = []
    R,t = X[:3,:3], X[:3,3]
    for m in matches:
        k = m.segment_id
        n = R @ m.lidar.normal
        d = m.camera.offset_m * scales[k] / reference_scales[k]
        rows.append(np.r_[m.camera.normal - n, d - m.lidar.offset_m + n @ t])
    return np.asarray(rows, dtype=float).reshape(-1,4)


def _robust(r, loss):
    """Signed sqrt(rho(r^2)); apply weights AFTER robustification."""
    if loss == "linear":
        return r
    if loss == "soft_l1":
        return r * np.sqrt(2. / (np.hypot(1.,r)+1.))
    if loss == "huber":
        return np.where(abs(r)<=1.,r,np.sign(r)*np.sqrt(np.maximum(2*abs(r)-1,0)))
    if loss == "cauchy":
        return np.sign(r)*np.sqrt(np.log1p(r*r))
    raise ValueError("Unsupported robust loss")


def fit_ground_handeye(pairs: MotionPairs, partition: ScalePartition, pair_mask: np.ndarray,
                       X_he: np.ndarray, reference_scales: np.ndarray,
                       matches: list[SegmentGround], *,
                       he_settings: HandEyeSettings = HandEyeSettings(),
                       ground_weight: float = 1., sigma_normal_deg: float = 3.,
                       sigma_offset_m: float = .15, refit_scales: bool = False,
                       normal_enabled: bool = True,
                       max_log_scale_delta: float = .15, max_rotation_deg: float = 5.,
                       max_translation_m: float = 1., initial_X: np.ndarray | None = None):
    """Fixed pairs and ground supports. B4-a fixed scales; B4-b positive scales.

    The reference HE defines bounds and camera metric scale even in an
    explicitly labeled GT-initialized diagnostic. No GT matching/re-gating.
    Ground weight 0 returns a HE-only diagnostic with the same parameterization.
    """
    X0 = validate_extrinsic(X_he)
    s0 = np.asarray(reference_scales, dtype=float)
    K = partition.count
    groups = partition.groups(pairs)
    mask = np.asarray(pair_mask)
    if (mask.shape != (pairs.count,) or mask.dtype != np.bool_ or
            np.any(mask & (groups<0)) or s0.shape != (K,)
            or not np.isfinite(s0).all() or np.any(s0<=0)):
        raise ValueError("Invalid saved pair mask / reference scales")
    values = [ground_weight,sigma_normal_deg,sigma_offset_m,max_log_scale_delta,
              max_rotation_deg,max_translation_m]
    if (not np.isfinite(values).all() or ground_weight<0 or min(values[1:])<=0
            or max_rotation_deg>=90 or sigma_normal_deg>=90):
        raise ValueError("Invalid optimization settings")
    if (len({m.segment_id for m in matches}) != len(matches)
            or any(not 0<=m.segment_id<K for m in matches)):
        raise ValueError("Require at most one ground pair per valid segment")
    if ground_weight>0 and not matches:
        return {"success":False,"reason":"no_ground_pairs"}
    sub = []
    for k in range(K):
        m = mask & (groups==k)
        sub.append(MotionPairs(pairs.Ra[m],pairs.ta[m],pairs.Rb[m],pairs.tb[m],
                               pairs.indices[m],pairs.starts_ns[m]))
    N = int(mask.sum())
    if N<7 or any(p.count<3 for p in sub):
        raise ValueError("Insufficient saved within-segment pairs")
    dim = 6 + (K if refit_scales else 0)
    def state(delta):
        X = X0.copy()
        X[:3,:3] = Rotation.from_rotvec(delta[:3]).as_matrix() @ X0[:3,:3]
        X[:3,3] += delta[3:6]
        return X, s0*np.exp(delta[6:]) if refit_scales else s0
    def blocks(delta):
        X,s = state(delta)
        h = np.concatenate([residuals(p,X[:3,:3],X[:3,3],float(s[k]))
                            for k,p in enumerate(sub)])
        g = ground_residuals(matches,X,s,s0)
        h[:,:3] /= np.deg2rad(he_settings.sigma_relative_rotation_deg)
        h[:,3:] /= he_settings.sigma_relative_translation_m
        g[:,:3] /= np.deg2rad(sigma_normal_deg)
        g[:,3] /= sigma_offset_m
        return h,g
    def objective(delta,robust=True,include_ground=True):
        h,g = blocks(delta)
        if not normal_enabled:
            g = g[:,3:4]
        if robust:
            h = _robust(h,he_settings.robust_loss)
            g = _robust(g,"soft_l1")
        result = h.ravel()/np.sqrt(N)
        if include_ground and ground_weight>0 and len(g):
            result = np.r_[result,g.ravel()*np.sqrt(ground_weight/len(g))]
        return result
    def diag(delta):
        X,s = state(delta)
        h,g = blocks(delta)
        raw = ground_residuals(matches,X,s,s0)
        rows = []
        for m,r in zip(matches,raw):
            angle = np.rad2deg(np.arccos(np.clip(m.camera.normal @ (X[:3,:3]@m.lidar.normal),-1,1)))
            rows.append({"segment_id":m.segment_id,"normal_angle_deg":float(angle),
                         "offset_error_m":float(r[3])})
        gr = _robust(g,"soft_l1")
        return {"he_cost":float(.5*np.sum(_robust(h,he_settings.robust_loss)**2)/N),
                "ground_normal_cost":float(.5*np.sum(gr[:,:3]**2)/len(g)) if len(g) else None,
                "ground_offset_cost":float(.5*np.sum(gr[:,3:]**2)/len(g)) if len(g) else None,
                "ground_cost":float(.5*np.sum((gr if normal_enabled else gr[:,3:4])**2)/len(g)) if len(g) else None,
                "total_cost":float(.5*np.sum(objective(delta)**2)),
                "ground":rows,"scales":s.tolist(),"T_camera_lidar":X.tolist(),
                "motion_translation_rms_m":float(np.sqrt(np.mean(np.sum(h[:,3:]**2,axis=1)))
                                                 *he_settings.sigma_relative_translation_m)}
    def sensitivity(delta, include_ground):
        units = np.r_[np.full(3,np.deg2rad(he_settings.reference_rotation_deg)),
                      np.full(3,he_settings.reference_translation_m),
                      np.full(dim-6,he_settings.reference_log_scale)]
        steps = np.eye(dim)*units*1e-5
        J = np.column_stack([(objective(delta+d,False,include_ground)
                              -objective(delta-d,False,include_ground))/2e-5 for d in steps])
        sv = np.linalg.svd(J,compute_uv=False)
        rank = int(np.sum(sv > max(sv[0],1e-15)*he_settings.rank_sv_ratio))
        nuisance = J[:,np.r_[0:3,6:dim]]
        U,S,_ = np.linalg.svd(nuisance,full_matrices=False)
        U = U[:,S>max(S[0],1e-15)*he_settings.rank_sv_ratio]
        T = J[:,3:6]-U@(U.T@J[:,3:6])
        _,ts,vh = np.linalg.svd(T,full_matrices=False)
        return {"rank":rank,"full_rank":rank==dim,"singular_values":sv.tolist(),
                "translation_marginal_singular_values":ts.tolist(),
                "weakest_translation_camera_xyz":vh[-1].tolist()}
    bound = np.r_[np.full(3,np.deg2rad(max_rotation_deg)),np.full(3,max_translation_m),
                  np.full(dim-6,max_log_scale_delta)]
    zero = np.zeros(dim)
    x0 = zero.copy()
    if initial_X is not None:
        Xi = validate_extrinsic(initial_X)
        x0[:3] = Rotation.from_matrix(Xi[:3,:3]@X0[:3,:3].T).as_rotvec()
        x0[3:6] = Xi[:3,3]-X0[:3,3]
        if np.any(abs(x0)>=bound):
            return {"success":False,"reason":"diagnostic_initializer_outside_HE_centered_bounds"}
    pre = sensitivity(x0,True)
    if not pre["full_rank"]:
        return {"success":False,"reason":"rank_deficient","initial":diag(x0),
                "initial_observability":pre}
    sol = least_squares(objective,x0,bounds=(-bound,bound),loss="linear",x_scale="jac",
                        max_nfev=he_settings.max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    post = sensitivity(sol.x,True)
    return {"success":bool(sol.success and post["full_rank"]),"solver_converged":bool(sol.success),
            "message":str(sol.message),"nfev":int(sol.nfev),"ground_weight":ground_weight,
            "refit_scales":refit_scales,"normal_enabled":normal_enabled,
            "parameter_count":dim,"motion_pair_count":N,
            "ground_pair_count":len(matches),"bound_hit":bool(np.any(sol.active_mask)),
            "baseline":diag(zero),"initial":diag(x0),"final":diag(sol.x),
            "initial_observability":pre,"final_observability":post,
            "HE_only_observability_at_baseline":sensitivity(zero,False),
            "warning":"Fixed provisional ground supports; rank is not calibrated covariance"}
