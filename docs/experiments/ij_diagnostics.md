# I/J: local observability and virtual moment controls

Reuse H graph, W1/W5 P2P factors, fixed GLIM, fixed intrinsics and all H
initial parameters. `bash scripts/experiments/run_ij_all.sh` runs 07/09/10.
IJ_MODE=I or J runs one family. GT never used for selection or fitting.
Default H reference NPZ must match exactly. Existing outputs are not overwritten.

I1/I2 HE componentwise no-ground/ground, I3/I4 scene W5 no-ground/ground.
At common original state and individual final state compute raw weighted
linear and transformed robust residual Jacobians (central FD). The robust
GN is that of the actual transformed residual used by H, not frozen IRLS
or the exact robust Hessian. Eliminate nuisance (8 scales and all local
pose/point variables) by orthogonal projection with sparse LSMR on
column-normalized nuisance J, no full normal matrix. 1deg/.1m X units,
left rotation tangent at evaluation state. Store spectrum, matrix, weak
direction, axis diagonal and per-axis solve quality; no statistical
covariance. Rank threshold relative singular value 1e-8 is operational,
not proof of observability. If projection unreliable, rank is null.
Even reliable iterative projection can obscure extremely weak modes.
Ground-on solution also evaluated with just ground rows removed (same
state); active nuisance fixed variant conditional, not feasible cone.
Extrinsic active bounds reported separately; do not interpret as free
unconstrained accuracy. Initial and solution diagnostics do not establish
global rank. INFO_MAXITER controls projection budget (default 10000).

J0 HE block robust; J1 W1; J2 W5 (alias I4 if both).
J3 zero centroid isotropic covariance with pooled uncentered second
moment trace; J4 pooled full moment; J5 within-segment permutation
(3 reproducible seeds, preserves exact segment distribution); J6 real
centroid + mean within-center covariance; J7 pooled centroid + real
covariance. J7 intentionally changes second-moment trace, report it;
do not interpret as perfectly matched strength. J3/J4/J5/J6 keep the
equal-segment/equal-center average uncentered trace, all preserve the
translation block. No GT selection. J0b isotropic chordal residual with
sigma_scene=sigma_t and covariance alpha I, alpha=(sigma_t/sigma_R)^2/2;
small-angle control for HE whitening and block robustification, not exact
at finite angles. All geometry controls use original W5 support and
same initial state, robust block, visual and ground terms. Report moments,
permutations and cross-evaluation on actual W1/W5; J0b deliberately uses
different sigma to match HE. No claim that trace matching equalizes all
directional information or robust effective weights. Compare rotation,
translation, trajectory metrics and bound/convergence flags. No held-out
evaluation or GLIM refinement here.

Outputs: outputs/local_baXX_IJ_TIMESTAMP/summary.json and per-arm NPZ.
Both mode 4 I + 11 J entries with 3 shuffle repeats. Count: J0,J1,J2,
J3,J4,J5x3,J6,J7,J0b = 11, I4/J2 identical saved alias, 14 actual fits.
Information solves can take longer than joint BA; J-only can run first.
