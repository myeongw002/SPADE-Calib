# OPT1 — Fixed-edge extrinsic refinement pilot

Input MV8 review/full (source_features.npz and arm source IDs), EP0 camera_jump
and B4c_sparse_depth. No regeneration of features. Run:
    python scripts/experiments/run_fixed_edge_refinement.py \
      --mv8 outputs/edgepoints07_MV8 --ep0 outputs/edgepoints07_EP0 \
      --out outputs/edgeopt07_OPT1

Default arms MV1_all, MV1_A, MV1_C_HV, Surface_HV, MV1_A_plus_MV7_V.
Same loss and solver across arms. Different initial point counts by design;
not equal-budget ablation. Default train anchors 69/189/389/509/659/995,
held-out 289/819 (same sequence, already seen in feature experiments; not
independent blind validation). Use --holdout with no values to fit all.
Fixed support chosen once using B4c, original EP0 camera-side z-buffer,
nearest candidate per initial pixel and 5px image margin; no GT selection.
Never change IDs, denominator, or visibility during optimization. This avoids
support dropping but may retain newly occluded points or miss newly visible
ones. Fixed support saved per arm/frame. DA2 edge indices integer, EDT bilinear
at continuous projected UV; differs slightly from MV8 floor+0.5 metric.
Outside image/depth gets continuous overflow penalty (heuristic); invalid
counts and fixed-denominator within5 reported. No explicit correspondence or
orientation gating. No HE/ground residual (translation y may drift), fixed
GLIM/SfM/scale, no regularization besides B4c-centered bounds.
Rotation chart R=Exp(rotvec_deg*pi/180) R0, t=t0+delta_t. Modes xz, t, se3
(2/3/6 parameters), bounds +/-2deg and +/-1m per active coordinate. SciPy TRF
soft_l1 scale5px, absolute central numerical Jacobian, max_nfev150.
Train objective pooled per-point mean rho(r^2), no frame balancing.
Primary starts zero delta B4c, identical config all arms. Explicit oracle
starts constrained GT coordinates relative B4c (fixed inactive DOF);
diagnostic only, not used to choose/tune primary result. Exact GT and
constrained GT costs reported on the same B4c support. GT may not minimize
DA2 edge objective. No claim of global convergence. XZ grid fixed B4c R/ty,
21x21 over bounds, train/holdout costs; grid min diagnostic not used as restart.
Compare R/t errors, signed xyz, loss, valid count, held-out improvement and
oracle drift. Summary JSON has all arm/mode/start metrics and landscape, each
T text and per-frame fixed NPZ and before/after images. --no-overlays reduces
output; --grid-n 0 skips landscape. Report failures/boundary hits, do not
select arm by GT on same sequence as final evidence. Numerical Jacobian SV
diagnostic not full observability proof. No real 16ch generalization claimed.
