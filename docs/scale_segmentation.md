# Piecewise scale and shared-extrinsic Hand-Eye

Research API port of the KITTI09/10 adaptive segmentation and fixed-8
ablation scripts. No private data, GT, /mnt/data paths, or implicit clock
alignment in the reusable modules. The original global Hand-Eye API is
unchanged. Requires the research extra (SciPy).

## Usage

```python
import json
import numpy as np
from spade_calib.io.kitti import load_kitti_trajectory_data
from spade_calib.selection.scale_segmentation import (
    ScaleSegmentationSettings, ScalePartition, build_frame_lag_pairs,
    adaptive_partition, fixed_partition, estimate_pair_scales, shared_pair_mask,
)
from spade_calib.handeye.multiscale import fit_segmented_motion_pairs

samples = load_kitti_trajectory_data(
    "odom_lidar.txt", "camera_poses.csv", "times.txt", glim_first_frame=0
)
X0 = np.array(json.load(open("handeye.json"))["handeye"]["T_camera_lidar"])
cfg = ScaleSegmentationSettings()
pairs = build_frame_lag_pairs(samples, lag=10)
n = len(samples.timestamps_ns)
adaptive, candidates = adaptive_partition(pairs, n, X0, cfg)
fixed = fixed_partition(n, adaptive.count)
global_partition = ScalePartition((0, n))
evidence = estimate_pair_scales(pairs, X0, cfg)
common = shared_pair_mask(pairs, evidence, adaptive, fixed)
fits = {
    name: fit_segmented_motion_pairs(pairs, partition, X0,
                                     segmentation_settings=cfg, pair_mask=common)
    for name, partition in (
        ("global", global_partition), ("fixed", fixed), ("adaptive", adaptive)
    )
}
with open("segmented_handeye.json", "w") as f:
    json.dump({"candidates": candidates,
               "fits": {name: fit.to_dict(samples) for name, fit in fits.items()}},
              f, indent=2)
```

Use the same source reconstruction and clock/frame association for all stages.
For timestamp-based sampling, use the existing selection.trajectory.build_motion_pairs
and pass its MotionPairs directly. Frame lag counts synchronized registered
poses (not necessarily consecutive original frames). For standalone deployment
rather than ablation, omit pair_mask to use all eligible pairs within the chosen
partition. A partition with insufficient pairs after intersection raises instead
of silently introducing an unobservable scale.

## Formulation and differences from the archived experiments

For B(s) X = X A, c = R_X t_A - (R_B-I)t_X and
s_i = (t_B dot c)/||t_B||^2. Default gates: baseline > 0.08 SfM units,
cosine > 0.95, 1 < s_i < 200. These are inherited heuristics, not universal
thresholds; a different SfM gauge can require different settings.

Adaptive DP minimizes start-bin weighted least-squares SSE C-V^2/W
on the common gated pairs, with 20 matched-pose bins and minimum segment
length 100 matched poses. Candidate count <= 8; pseudo-BIC uses 3N scalar
translation components. Overlapping pairs and estimated change points violate
classical BIC assumptions. The score selects a proposal, not a statistically
identified number of physical scale changes. DP scoring assigns pairs by start
bin, like the original; actual fitting excludes boundary-crossing pairs.
Unlike the archive, the final boundary covers all synchronized poses, including
the final lag tail; minimum segment length is enforced on actual indices, and
candidates must have enough entirely internal pairs. Exact old numeric output
is therefore not guaranteed.

The nonlinear fit reuses the existing 6D motion residual, one shared X and
K positive scales, soft-L1 by default, with 0.5 deg / 0.05 m assumed residual
scales. Both pooled and segment OLS initializations are tried; failed
optimizations are not selected. Rotation/translation fixing is available for
diagnostics. Global/fixed/adaptive must use a common pair mask for comparison.
The returned Jacobian rank is a normalized local numerical diagnostic only,
not physical uncertainty or proof of accuracy. No GT is needed by this API.

## Segment reconstruction handoff

Each segment includes half-open matched-pose indices, metric scale, pair
count, and (when samples are supplied) image names and timestamps. All
segments share T_camera_lidar. Convert the SfM local pose/landmark geometry
to metric with its own segment scale around an explicit reference camera;
do not scale arbitrary world origins independently or assume segment maps
share an aligned world origin. Timestamp intervals are not evidence of raw
scan availability. Check recording coverage before accumulation.

Local LiDAR pose refinement, local SfM BA, plane extraction and plane
association are subsequent stages, not implemented by this PR.
