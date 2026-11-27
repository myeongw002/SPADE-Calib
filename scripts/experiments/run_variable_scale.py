#!/usr/bin/env python3
"""P1 matched constant vs log-linear scale Hand-Eye experiment."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from spade_calib.io.kitti import load_kitti_trajectory_data,frame_id_from_name
from spade_calib.selection.scale_segmentation import ScalePartition,build_frame_lag_pairs,validate_extrinsic
from spade_calib.selection.trajectory import estimate_rotation_and_sfm_scale
from spade_calib.handeye.solver import HandEyeSettings
from spade_calib.handeye.trajectory_oracle import subset,kitti_oracle_pairs
from spade_calib.handeye.variable_scale import ScaleMotion,blocked_masks,fit,evaluate
from spade_calib.calibration.segmented_ground import _robust
from spade_calib.evaluation.kitti import parse_kitti_odom_calib,evaluate_camera_lidar


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--glim",type=Path);p.add_argument("--sfm",type=Path);p.add_argument("--sequence",type=Path)
    p.add_argument("--gt-calib",type=Path);p.add_argument("--gt-poses",type=Path)
    p.add_argument("--slope-bound",type=float,default=.25)
    args=p.parse_args(argv)
    if bool(args.gt_calib)!=bool(args.gt_poses):raise ValueError("Supply both evaluation GT files or neither")
    if args.out.exists() and any(args.out.iterdir()):raise FileExistsError("Use fresh output")
    read=lambda p:json.loads(p.read_text())
    cfg=read(args.run/"config.json");saved=read(args.run/"segmented_handeye.json")["fit"]
    rows=saved["segments"];X=validate_extrinsic(np.asarray(saved["T_camera_lidar"],float))
    s=np.asarray([r["sfm_scale_to_metric"] for r in rows]);he=HandEyeSettings(**saved["settings"])
    partition=ScalePartition(tuple([0]+[int(r["stop_index_exclusive"]) for r in rows]))
    for k,(a,b) in enumerate(partition.intervals):
        if rows[k]["segment_id"]!=k or rows[k]["start_index"]!=a:raise ValueError("Saved partition mismatch")
    samples=load_kitti_trajectory_data(args.glim or cfg["glim"],args.sfm or cfg["sfm"],
                   Path(args.sequence or cfg["sequence"])/"times.txt",glim_first_frame=int(cfg["glim_first_frame"]))
    if list(samples.image_names)!=[n for r in rows for n in r["image_names"]] or samples.timestamps_ns.tolist()!=[n for r in rows for n in r["timestamps_ns"]]:
        raise ValueError("Archived matched images/timestamps changed")
    pairs=build_frame_lag_pairs(samples,int(cfg["lag"]))
    selected=np.asarray(saved["selected_pair_rows"])
    if selected.ndim!=1 or selected.dtype.kind not in "iu" or len(np.unique(selected))!=len(selected) or np.any(selected<0) or np.any(selected>=pairs.count):
        raise ValueError("Invalid saved pair rows")
    mask=np.zeros(pairs.count,bool);mask[selected]=True
    groups=partition.groups(pairs)
    if np.any(mask&(groups<0)):raise ValueError("Saved pairs cross boundaries")
    pairs=subset(pairs,mask);model=ScaleMotion(samples,pairs,partition)
    if any(np.count_nonzero(model.groups==k)!=r["motion_pair_count"] for k,r in enumerate(rows)):
        raise ValueError("Saved pair counts mismatch")
    full=np.ones(pairs.count,bool);zero=np.zeros(partition.count)
    sigma=np.r_[np.full(3,np.deg2rad(he.sigma_relative_rotation_deg)),np.full(3,he.sigma_relative_translation_m)]
    cost=.5*float(np.sum(_robust(model.raw(X,np.log(s),zero)/sigma,he.robust_loss)**2))
    if not np.isclose(cost,saved["optimization"]["cost_whitened_robust"],rtol=1e-5,atol=1e-7):
        raise ValueError("Original HE cost mismatch; check source poses")
    train,test=blocked_masks(model)
    summary=dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                 archived_cost=cost,archive_cost_verified=True,pair_count=pairs.count,
                 segment_counts=[dict(segment=k,full=int(np.count_nonzero(model.groups==k)),
                                      train=int(np.count_nonzero(train&(model.groups==k))),
                                      test=int(np.count_nonzero(test&(model.groups==k)))) for k in range(partition.count)],
                 discarded_boundary_pairs=int(np.count_nonzero(~(train|test))),
                 source_sha256={k:hashlib.sha256((args.run/k).read_bytes()).hexdigest() for k in ("config.json","segmented_handeye.json")},
                 experiments={},warning="GT evaluation only, no GT used to optimize. Same archived pair mask (original data-dependent selection), blocked fits are conditional diagnostics, not an independently selected validation set.")
    args.out.mkdir(parents=True,exist_ok=True)
    def write(): (args.out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    fitted={}
    baseline=dict(T_camera_lidar=X.tolist(),scales_mid=s.tolist(),log_slopes=zero.tolist(),cost=cost,reference_only=True)
    fitted["archived_constant"]=(baseline,full)
    for label,lin in (("full_constant",False),("full_linear",True)):
        print("Fitting",label,flush=True)
        fitted[label]=(fit(model,X,s,he,full,linear=lin,slope_bound=args.slope_bound),full)
    if all(np.count_nonzero(train&(model.groups==k))>=3 and np.count_nonzero(test&(model.groups==k))>=1 for k in range(partition.count)):
        try:
            R,sg,d=estimate_rotation_and_sfm_scale(subset(pairs,train))
            init=np.eye(4);init[:3,:3]=R;init[:3,3]=d["provisional_t_camera_lidar_m"]
            scales=np.full(partition.count,sg)
            for k in range(partition.count):
                m=train&(model.groups==k);tb=pairs.tb[m]
                c=pairs.ta[m]@R.T-(pairs.Rb[m]-np.eye(3))@init[:3,3]
                sk=float(np.sum(tb*c)/np.sum(tb*tb))
                if np.isfinite(sk) and sk>0:scales[k]=sk
            summary["blocked_initializer"]="training-only closed-form HE and segment OLS"
            for label,lin in (("blocked_constant",False),("blocked_linear",True)):
                print("Fitting",label,flush=True)
                fitted[label]=(fit(model,init,scales,he,train,linear=lin,slope_bound=args.slope_bound),train)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as e:
            summary["blocked_failure"]=str(e)
    else:summary["blocked_failure"]="Insufficient disjoint train/test support in at least one segment"
    # Load GT only after every optimization is complete.
    gt=None;Xgt=None
    frames=np.array([frame_id_from_name(n) for n in samples.image_names])
    if args.gt_calib:
        gt,Xgt=kitti_oracle_pairs(np.loadtxt(args.gt_poses),parse_kitti_odom_calib(args.gt_calib),int(cfg["camera_index"]),frames,pairs)
    for label,(result,m) in fitted.items():
        result["fit_metrics"]=evaluate(model,result,m,gt)
        result["all_selected_metrics"]=evaluate(model,result,full,gt)
        result["outer_holdout_metrics"]=evaluate(model,result,test,gt)
        if Xgt is not None:result["gt_evaluation"]=evaluate_camera_lidar(np.asarray(result["T_camera_lidar"]),Xgt)
        summary["experiments"][label]=result
        write()
        print(label,"cost",result["cost"],"success",result.get("success"),"GT",result.get("gt_evaluation",{}).get("translation_error_total_m"),flush=True)
    # Export every selected pair's integrated camera displacement and split.
    arr=dict(frame_ids=frames[pairs.indices],matched_indices=pairs.indices,segment_ids=model.groups,train=train,test=test,
             Ra=pairs.Ra,ta=pairs.ta,Rb=pairs.Rb,tb_original=pairs.tb)
    if gt is not None:arr["tb_gt"]=gt.tb
    for label,(result,_) in fitted.items():
        arr["tb_"+label]=model.metric_tb(np.log(result["scales_mid"]),result["log_slopes"])
    np.savez_compressed(args.out/"motion_pairs.npz",**arr)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
