# S0 / S1: falsify scale–extrinsic coupling hypothesis first

S0: python scripts/experiments/run_scale_geometry_s0.py --out outputs/S0
Five synthetic camera motions and line layouts; exact LiDAR relative poses.
Compare line-only single/multi, HE, HE+single/multi and HE+GLIM-projected
multi-view. Last arm has identical image observations but no direct scale
dependence in the line term. One shared positive scale, no ground or BA.
Noiseless and 10 noisy paired trials by default; perturb image endpoints
0.5px, camera centers 0.01 nonmetric units and rotations 0.05deg.
Truth used only for generating observations, initializing a prescribed
perturbation, and evaluation/rank at truth. Full 7-column local Jacobian
(1deg, .1m, .1 log-scale), complete null space, conditional scale info,
scale-fixed/free X info, numerical rank threshold. Analytical scale
derivative independently checked against central finite differences.
Rank-deficient optimizer convergence does not imply truth recovery.
Not independent image samples, not statistical covariance.

S1: reuse original H archive, no refit.
```bash
for SEQ in 07 09 10; do
  python scripts/experiments/run_local_visual_ba.py --s1 \
    --run outputs/segment${SEQ}_trial1 \
    --ground-run outputs/ground${SEQ}_B4c \
    --scene-factors outputs/scene${SEQ}_H_factors \
    --reference-ba outputs/local_ba${SEQ}_H \
    --window 5 --out outputs/scale_coupling${SEQ}_S1
done
```
Requires original COLMAP model and same original H inputs/config/support.
No GT required. Reconstruct saved H1/H2 parameter vectors, validate
camera/landmark/observations/scale and initial/final objective cost.
Use identical sparse Jacobian at common initializer and archived solution;
raw weighted and transformed robust modes. Marginalize local poses and
landmarks with scales fixed vs marginalize also K log-scales. Report
H_fixed, H_free, PSD difference, common-threshold ranks, eigenvalues,
directional/trace loss fractions, generalized loss fractions on supported
subspace, weak direction, solver reliability and near-bound indices.
No re-optimization, no new labels. Existing H/I/J/M1 unchanged by default.
These are local GN diagnostics with shared fixed GLIM, not covariance
or causal attribution of existing calibration error. Scale loss can be
small; if so, do not develop S2/S3 just to force a coupling hypothesis.
Bounds ignored in information calculation. No adaptive accumulation yet.
