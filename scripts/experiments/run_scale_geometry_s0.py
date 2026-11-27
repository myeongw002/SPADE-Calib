#!/usr/bin/env python3
"""Run S0 synthetic observability; no KITTI data needed."""
import argparse
import json
from pathlib import Path
import numpy as np
from spade_calib.calibration.scale_geometry_synthetic import CASES,ARMS,UNITS,run_trial


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--noisy-trials",type=int,default=10)
    a=p.parse_args(argv)
    if a.seed<0 or a.noisy_trials<0: p.error("Nonnegative seed/trial count required")
    if a.out.exists() and any(a.out.iterdir()): raise FileExistsError("Use fresh output")
    a.out.mkdir(parents=True,exist_ok=True)
    result=dict(units_rad_m_logscale=UNITS.tolist(),seed=a.seed,noisy_trials=a.noisy_trials,
        noise=dict(image_endpoint_px=.5,camera_center_nonmetric=.01,camera_rotation_deg=.05,lidar="exact"),
        warning="Synthetic local rank at truth, fixed correspondences, no ground/BA/pose uncertainty, not calibrated covariance. Rank-deficient fit may converge without recovering truth.",
        cases={})
    for case in CASES:
        clean=run_trial(case,a.seed)
        noisy=[run_trial(case,a.seed+i,True) for i in range(a.noisy_trials)]
        summary={}
        for arm in ARMS:
            rows=[r["arms"][arm] for r in noisy]
            summary[arm]=dict(converged=sum(r["converged"] for r in rows),
                bound_hit=sum(r["bound_hit"] for r in rows),
                median_translation_error_m=float(np.median([r["translation_error_m"] for r in rows])) if rows else None,
                median_rotation_error_deg=float(np.median([r["rotation_error_deg"] for r in rows])) if rows else None,
                median_absolute_relative_scale_error=float(np.median([abs(r["relative_scale_error"]) for r in rows])) if rows else None)
        result["cases"][case]=dict(noiseless=clean,noisy=noisy,noisy_summary=summary)
        print(case, {arm:clean["arms"][arm]["observability_at_truth"]["rank"] for arm in ARMS},flush=True)
        (a.out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
