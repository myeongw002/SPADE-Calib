"""Rectified KITTI projection diagnostics, never used for fitting.

T_camera_lidar maps directly into the chosen physical rectified camera.
Use K=P_k[:,:3], NOT full P_k: the camera baseline is already in X.
Images must be original-size rectified images for that camera.
"""
from __future__ import annotations

import logging
from pathlib import Path
import cv2
import numpy as np
from spade_calib.selection.scale_segmentation import validate_extrinsic

LOG = logging.getLogger(__name__)


def project_xyz(xyz, X, K, width, height, *, min_depth_m=.5, max_depth_m=65.):
    """Return source indices, continuous pixel coordinates and optical Z.

    Filter behind-camera, nonfinite, out-of-range and out-of-image points.
    Pixel visibility is handled separately so comparisons can use source IDs.
    """
    X = validate_extrinsic(X)
    K = np.asarray(K,dtype=float)
    p = np.asarray(xyz,dtype=float)
    if (p.ndim!=2 or p.shape[1]!=3 or K.shape!=(3,3)
            or not np.isfinite(K).all() or not np.allclose(K[2],[0,0,1])
            or min(K[0,0],K[1,1])<=0 or min(width,height)<=0
            or not np.isfinite([min_depth_m,max_depth_m]).all()
            or not 0<min_depth_m<max_depth_m):
        raise ValueError("Invalid rectified projection inputs")
    ids = np.flatnonzero(np.isfinite(p).all(axis=1))
    pc = p[ids] @ X[:3,:3].T + X[:3,3]
    keep = (pc[:,2]>=min_depth_m)&(pc[:,2]<=max_depth_m)
    ids,pc = ids[keep],pc[keep]
    uv = (pc@K.T)[:,:2]/pc[:,2:3]
    keep = (uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
    return ids[keep],uv[keep],pc[keep,2]


def visible_pixels(uv,z,width):
    """Nearest optical Z per floored pixel; not complete scene occlusion."""
    pix = np.floor(uv).astype(np.int64)
    flat = pix[:,1]*width+pix[:,0]
    order = np.argsort(z,kind="stable")
    _,first = np.unique(flat[order],return_index=True)
    return order[first]


def render_overlay(image, uv, z, max_depth_m):
    """Fixed depth color scale across all transforms; radius 1 px."""
    out = image.copy()
    v = visible_pixels(uv,z,image.shape[1])
    if not len(v):
        return out, 0
    pix = np.floor(uv[v]).astype(int)
    colors = cv2.applyColorMap(np.clip((1-z[v]/max_depth_m)*255,0,255).astype(np.uint8).reshape(-1,1),
                               cv2.COLORMAP_TURBO).reshape(-1,3)
    # Draw far-to-near so nearby points also win adjacent marker overlap.
    for i in np.argsort(z[v])[::-1]:
        cv2.circle(out,tuple(pix[i]),1,tuple(int(c) for c in colors[i]),-1,lineType=cv2.LINE_8)
    return out,len(v)


def _save(path,image):
    if not cv2.imwrite(str(path),image):
        raise OSError(f"Could not write {path}")


def _numbered_file(folder,frame,suffixes):
    for suffix in suffixes:
        candidate = Path(folder)/f"{frame:06d}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing frame {frame:06d} in {folder}")


def _read_scan(path):
    if path.stat().st_size%16:
        raise ValueError(f"Expected KITTI float32 XYZI scan: {path}")
    return np.fromfile(path,dtype="<f4").reshape(-1,4)[:,:3]


def export_projections(*,image_dir,scan_dir,run_dir,out_dir,anchors,extra_frames,arm,
                       cloud_mode,K,transforms,read_xyz_ply,max_depth_m=65.):
    """Same cloud/image/depth mapping for HE, each successful fit and GT.

    Single-scan anchor and requested extra frames are raw synchronous scans.
    Accumulated cloud projections are diagnostic only (motion/visibility
    artifacts), never treated as pixel correspondence ground truth.
    """
    if cloud_mode not in ("single","accumulated","both"):
        raise ValueError("Unknown projection cloud mode")
    if not np.isfinite(max_depth_m) or max_depth_m<=.5:
        raise ValueError("Projection max depth must exceed .5m")
    if not Path(image_dir).is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if cloud_mode in ("single","both") and not Path(scan_dir).is_dir():
        raise FileNotFoundError(f"Missing single-scan directory: {scan_dir}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    report = {"image_dir":str(image_dir),"single_scan_dir":str(scan_dir),"cloud_mode":cloud_mode,
              "max_depth_m":max_depth_m,"depth_color":"TURBO, same optical Z scale across panels",
              "intrinsics_K":np.asarray(K).tolist(),"transforms":list(transforms),"frames":[]}
    for frame in sorted(set(anchors)|set(extra_frames)):
        image_path = _numbered_file(image_dir,frame,(".png",".jpg",".jpeg"))
        img = cv2.imread(str(image_path),cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Unreadable image: {image_path}")
        clouds = {}
        if cloud_mode in ("single","both"):
            clouds["single"] = _read_scan(_numbered_file(scan_dir,frame,(".bin",)))
        if cloud_mode in ("accumulated","both") and frame in anchors:
            clouds["accumulated"] = read_xyz_ply(Path(run_dir)/f"segment_{anchors[frame]:02d}"/arm/"cloud_voxel.ply")
        for label,xyz in clouds.items():
            folder = out_dir/f"frame_{frame:06d}"/label
            folder.mkdir(parents=True,exist_ok=True)
            h,w = img.shape[:2]
            projections = {name:project_xyz(xyz,X,K,w,h,max_depth_m=max_depth_m)
                           for name,X in transforms.items()}
            record = {"frame":frame,"segment_id":anchors.get(frame),"cloud":label,
                      "input_points":len(xyz),"image":str(image_path),"image_shape":[h,w],
                      "variants":{}}
            panels=[]
            for name,(ids,uv,z) in projections.items():
                overlay,visible = render_overlay(img,uv,z,max_depth_m)
                _save(folder/f"{name}.png",overlay)
                info={"in_image_points":len(ids),"occupied_pixels":visible,
                      "T_camera_lidar":np.asarray(transforms[name]).tolist()}
                if "GT_reference" in projections:
                    gt_ids,gt_uv,_ = projections["GT_reference"]
                    _,i,j = np.intersect1d(ids,gt_ids,assume_unique=True,return_indices=True)
                    diff = np.linalg.norm(uv[i]-gt_uv[j],axis=1)
                    info["GT_common_projected_source_points"] = len(diff)
                    info["GT_pixel_displacement_median"] = float(np.median(diff)) if len(diff) else None
                    info["GT_pixel_displacement_p90"] = float(np.percentile(diff,90)) if len(diff) else None
                record["variants"][name]=info
                panel = cv2.copyMakeBorder(overlay,28,0,0,0,cv2.BORDER_CONSTANT,value=(24,24,24))
                cv2.putText(panel,f"{name} | frame {frame:06d} | {label} | {visible} px",
                            (8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
                panels.append(panel)
            if len(panels)%2:
                panels.append(np.zeros_like(panels[0]))
            grid = np.vstack([np.hstack(panels[i:i+2]) for i in range(0,len(panels),2)])
            _save(folder/"comparison.png",grid)
            report["frames"].append(record)
            LOG.info("Projection frame %06d %s: %d variants",frame,label,len(panels))
    report["warning"]="GT pixel displacement uses identical source-point IDs, not verified image features; accumulated cloud can contain stale/occluded points."
    return report
