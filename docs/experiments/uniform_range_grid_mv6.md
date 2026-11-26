# MV6 — uniform angular range image, single anchor

MV5 multi-pose virtual extraction is intentionally not used. Fixed W21 raw
cloud and anchor free-space mask as MV4. Same raw data, GLIM, EP0 and M0.
Run:
    python scripts/experiments/run_uniform_range_grid.py \
        --ring-calib /actual/path/to/velodynecalib_S2_factory_flatness_intensity.xml \
        --out outputs/edgepoints07_MV6

Defaults: azimuth .2deg, elevation .1/.2/.4deg, geometric elevation [-26,4)
chosen as a fixed approximate band around HDL-64 nominal elevation limits;
not equal to the physical-beam acceptance region. Report effective steps
and eligible out-of-FOV counts. No beam residual rejection in uniform grid.
Nearest radial range per cell; no interpolation or invented points. Unknown
cells not depth zero. All candidates from original returns. Horizontal jump
uses exact same ring_edges function on uniform elevation rows; vertical
jump uses adjacent occupied rows in each azimuth column, no vertical wrap.
Both use actual angle gaps <=.8deg and identical radial jump/support
thresholds. Horizontal and vertical sampled neighbors may differ in the
orthogonal angle within one cell. No range smoothing or PCA.
Controls: exact MV1 and MV4 Nearest64, per-resolution target-only HV.
Uniform H, V, HV and HV new source IDs vs MV1. Candidate counts differ;
no equal-budget claim. Uniform H vs physical H includes changed beam model
and FOV acceptance, not pure resolution change. H vs HV isolates direction.
Projection uses common EP0 same-pixel visibility, B4c and optional GT;
DA2 jump proximity not physical correspondence precision or calibration.
Outputs summary with pooled counts, angular occupancy, gap distributions,
direction counts/overlap and mixed source hypotheses; grid and feature NPZ,
overlays. No calibration optimization. Synthetic tests only until real run.
