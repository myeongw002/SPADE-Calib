#!/usr/bin/env python3
"""D0–D3 oracle: use saved pairs/partition, no clouds or images needed."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from spade_calib.io.kitti import load_kitti_trajectory_data,frame_id_from_name
from spade_calib.evaluation.kitti import parse_kitti_odom_calib,evaluate_camera_lidar
from spade_calib.selection.scale_segmentation import ScalePartition,build_frame_lag_pairs,ScaleSegmentationSettings
from spade_calib.handeye.solver import HandEyeSettings
from spade_calib.handeye.multiscale import fit_segmented_motion_pairs
from spade_calib.handeye.trajectory_oracle import kitti_oracle_pairs,substitute,subset,rows,whiten,profile_scales,bias,components
from spade_calib.calibration.segmented_ground import _robust


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("run","out","gt-poses","gt-calib"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("sequence","glim","sfm"):
        p.add_argument("--"+name,type=Path)
    args=p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    cfg=json.loads((args.run/"config.json").read_text())
    saved=json.loads((args.run/"segmented_handeye.json").read_text())["fit"]
    segments=saved["segments"]
    partition=ScalePartition(tuple([0]+[int(r["stop_index_exclusive"]) for r in segments]))
    scales=np.array([r["sfm_scale_to_metric"] for r in segments])
    X=np.array(saved["T_camera_lidar"])
    h=HandEyeSettings(**saved["settings"])
    seg=ScaleSegmentationSettings(**saved["segmentation_settings"])
    samples=load_kitti_trajectory_data(args.glim or cfg["glim"],args.sfm or cfg["sfm"],
                    Path(args.sequence or cfg["sequence"])/"times.txt",glim_first_frame=int(cfg["glim_first_frame"]))
    if (list(samples.image_names)!=[n for r in segments for n in r["image_names"]]
            or samples.timestamps_ns.tolist()!=[n for r in segments for n in r["timestamps_ns"]]):
        raise ValueError("Archived frame/timestamp mismatch")
    actual=build_frame_lag_pairs(samples,int(cfg["lag"]))
    selected=np.asarray(saved["selected_pair_rows"])
    if selected.ndim!=1 or selected.dtype.kind not in "iu" or len(np.unique(selected))!=len(selected) or np.any(selected<0) or np.any(selected>=actual.count):
        raise ValueError("Invalid archived pair rows")
    mask=np.zeros(actual.count,dtype=bool);mask[selected]=True
    groups=partition.groups(actual)[mask]
    if np.any(groups<0) or any((groups==k).sum()!=r["motion_pair_count"] for k,r in enumerate(segments)):
        raise ValueError("Archived pair/segment mismatch")
    a=subset(actual,mask)
    cost=.5*np.sum(_robust(whiten(rows(a,groups,scales,X),h),h.robust_loss)**2)
    archived=saved["optimization"]["cost_whitened_robust"]
    if not np.isclose(cost,archived,rtol=1e-5,atol=1e-7):
        raise ValueError(f"Archived HE cost mismatch {cost} vs {archived}")
    frames=np.array([frame_id_from_name(n) for n in samples.image_names])
    gt,Xgt=kitti_oracle_pairs(np.loadtxt(args.gt_poses,ndmin=2),parse_kitti_odom_calib(args.gt_calib),
                             int(cfg["camera_index"]),frames,actual)
    g=subset(gt,mask)
    closure=float(np.max(abs(rows(g,groups,np.ones(partition.count),Xgt))))
    if closure>1e-7: raise ValueError("GT relative-motion closure failed")
    args.out.mkdir(parents=True,exist_ok=True)
    report={"diagnostic_only":True,"warning":"GT trajectories derived from known X; not independent validation.",
            "archived_cost":archived,"recomputed_cost":float(cost),"pair_count":int(mask.sum()),
            "GT_closure_max_abs":closure,"GT_X":Xgt.tolist(),"run":str(args.run),
            "archived_evaluation":evaluate_camera_lidar(X,Xgt),"experiments":{}}
    np.savez_compressed(args.out/"motion_pairs.npz",frame_ids=frames[actual.indices[mask]],
                        segment_ids=groups,Ra=a.Ra,ta=a.ta,Rb=a.Rb,tb=a.tb,
                        Ra_gt=g.Ra,ta_gt=g.ta,Rb_gt=g.Rb,tb_gt=g.tb,
                        reference_scales=scales,X_he=X,X_gt=Xgt)
    for name,lg,cg in [("D0",False,False),("D1",True,False),("D2",False,True),("D3",True,True)]:
        motion=substitute(actual,gt,lg,cg)
        s0=np.ones(partition.count) if cg else scales
        fixed=subset(motion,mask)
        sgt=s0 if cg else profile_scales(fixed,groups,Xgt,s0,h)
        opts=dict(pair_mask=mask,initial_scales=s0,freeze_pair_mask=True,fix_scales=cg)
        fit=fit_segmented_motion_pairs(motion,partition,X,h,seg,**opts)
        Xr=X.copy();Xr[:3,:3]=Xgt[:3,:3]
        rf=fit_segmented_motion_pairs(motion,partition,Xr,h,seg,fix_rotation=True,**opts)
        row={"lidar_GT":lg,"camera_GT":cg,"scale_fixed":cg,"fit":fit.to_dict(),
             "GT_evaluation":evaluate_camera_lidar(fit.T_camera_lidar,Xgt),
             "GT_R_fixed":{"fit":rf.to_dict(),"GT_evaluation":evaluate_camera_lidar(rf.T_camera_lidar,Xgt)},
             "GT_bias":bias(fixed,groups,Xgt,sgt,h,cg,components(a,g,groups,sgt,Xgt) if name=="D0" else None)}
        report["experiments"][name]=row
        if name=="D0":
            report["D0_cost_reproduced"]=bool(np.isclose(fit.diagnostics["cost_whitened_robust"],archived,rtol=1e-5,atol=1e-7))
        (args.out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        e=row["GT_evaluation"]
        print(f"{name}: R={e['rotation_error_total_deg']:.6f} deg t={e['translation_error_total_m']:.6f} m",flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
