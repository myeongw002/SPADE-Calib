# EP0 — independent edge points (no lines)

Read existing E1 raw DA2 arrays and W21 cloud manifest. No ELSED arrays,
features.csv, annotations, line extraction, feature budget, matching or fit.
E1 is only a cache for original image and raw relative inverse depth.
Use B4-c as projection initializer. Optional GT is a separate extraction
arm, not a correction or label.

    git pull --ff-only
    KITTI=/media/myungw00/256GB_SSD/kitti
    python scripts/experiments/run_edge_points.py \
        --gt-calib "$KITTI/data_odometry_calib/sequences/07/calib.txt" \
        --out outputs/edgepoints07_EP0

Default inputs: outputs/segment07_trial1, outputs/ground07_B4c,
outputs/edge07_E1_depth, outputs/lines07_M0_ghost. Override --run,
--ground, --camera, --cloud-data; --frames selects common archived frames.
No ZIP required. If missing W21, prepare only (see P2L0 documentation).
Without --gt-calib only B4c is extracted. Default includes ground returns;
--non-ground optionally uses archived segment ground to restrict foreground
and same-depth support, not the z-buffer.

Camera: Canny reference 50/150; DA2 P95-P5 normalized inverse depth,
Gaussian sigma 1.2px; gradient / absolute Hessian eigen-direction NMS.
Verify bilateral affine profiles from original normalized prediction using
3..9px offsets. Jump is signed intercept difference in near->far direction.
Crease is magnitude of opposite outward slope sum times profile span.
Thresholds .02 normalized units, adjustable --camera-jump/--camera-crease.
These are provisional proxies, not physical type labels. Jump can cause
curvature responses; smooth but nonlinear depth can also cause crease.
No threshold tuning on GT, no use of metric depth. Normal maps show
near->far polarity for jump; crease orientation is axial (no polarity).

LiDAR: 1px z-buffer stores metric optical depth, original cloud index,
valid mask. NaN for unobserved depth; no densification. Search exact
Euclidean radius (default 12px) among observed projected points, nearest
per 8 angular sectors. Accept near->far depth jump >= max(1m, .1*near Z)
and opposite foreground support within max(.4m,.05*near Z).
All original observed returns; no reconstructed intersection or midpoint
used as measured edge. Save near, far and support IDs, pixel locations,
near/far XYZ, scan frames, signed near->far direction, inverse-depth jump,
local interpretation plane normal and near-far separation in pixels.
Multiple directions for one foreground ID retained; report unique IDs
separately. The near-far separation is a sampling proxy, not guaranteed
true edge bracket or calibrated confidence. Nearest-sector selection can
miss edges on sparse scans; radius and thresholds are pilot choices.
Occlusion, dynamic objects and accumulated pose error remain.

Outputs per frame: original.png, rgb_canny.png, da2_depth.png,
camera_jump/crease.npz + maps/direction/overlays, B4c_sparse_depth.npz
(depth_m,index_map,valid), B4c_edges.npz, B4c_* maps/overlays, optional
GT_oracle versions, comparison.png, summary.json. Green camera jump,
orange LiDAR near points, magenta camera crease. Overlap is visual only;
no associations or precision estimates. Direction images may overwrite
multiple hypotheses at same pixel; NPZ retains every hypothesis.
Normal-map hue represents direction, not depth or confidence.
Unit tests cover z-buffer ID, observed vs unknown edge, jump polarity,
smooth profile, crease, interpretation normal and empty cloud. Real
KITTI extraction and CLI integration need verification.
