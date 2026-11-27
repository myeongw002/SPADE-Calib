#!/usr/bin/env python3
"""S0-R paired multi-segment synthetic controls."""
import argparse
import json
from pathlib import Path
import numpy as np
from spade_calib.calibration.scale_geometry_synthetic_r import CASES,REGIMES,MODELS,ARMS,run_trial


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=17)
    p.add_argument("--noisy-trials",type=int,default=5)
    a=p.parse_args(argv)
    if a.seed<0 or a.noisy_trials<0: p.error("Nonnegative seed/trials required")
    if a.out.exists() and any(a.out.iterdir()): raise FileExistsError("Use fresh output")
    a.out.mkdir(parents=True,exist_ok=True)
    doc=dict(seed=a.seed,noisy_trials=a.noisy_trials,regimes=REGIMES,
             noise_order=["image_px","camera_translation_m","camera_rotation_deg","lidar_translation_m","lidar_rotation_deg","ground_offset_m"],
             warning="Synthetic independent anchor-relative perturbations, not real GLIM/SfM covariance. Paired starts/noise across arms. Ground normal exact, offset noisy. No BA, no accumulation, known segment boundaries.",
             conditions={})
    for case in CASES:
        for drift in (False,True):
            for regime in REGIMES:
                n=1 if regime=="clean" else a.noisy_trials
                rows=[run_trial(case,regime,a.seed+i,drift) for i in range(n)]
                summary={}
                for model in MODELS:
                    summary[model]={}
                    for arm in ARMS:
                        r=[v["arms"][model][arm] for v in rows]
                        median=lambda key:float(np.median([v[key] for v in r])) if r else None
                        summary[model][arm]=dict(converged=sum(v["converged"] for v in r),
                            bounds=sum(v["bound_hit"] for v in r),
                            median_translation_error_m=median("translation_error_m"),
                            median_rotation_error_deg=median("rotation_error_deg"),
                            median_scale_relative_rmse=median("scale_relative_rmse"),
                            ranks=[v["information_at_solution"]["rank"] for v in r])
                key=f"{case}_{'drift' if drift else 'constant'}_{regime}"
                doc["conditions"][key]=dict(trials=rows,summary=summary)
                (a.out/"summary.json").write_text(json.dumps(doc,indent=2,allow_nan=False)+"\n")
                print(key,flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
