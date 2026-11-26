# E0 — Observability-guided line selection pilot

Actual archived run: KITTI07, eight C0 anchors, 991 eligible ELSED lines (>=40 px), 2148 archived global L0 line candidates. B4-c w0.1, fixed scales, reproduced robust cost 1.4291495212066987. C0 archive originated with H2; only raw 3D/2D candidates reused, not H2 matches. No GT selection, no calibration fit. All 35 unique proposals require physical review. No DA2 checkpoint in execution environment; synthetic depth-cue checks only.

## Run

Use the four existing ZIPs (or corresponding extracted directory roots). Install the existing research dependencies (NumPy, SciPy, OpenCV). Run from repository root.

```bash
python scripts/experiments/run_edge_observability.py \
  --m0 line_support.zip --c0 lines07_C0.zip \
  --ground ground_b4c.zip --motion trajectory_oracle.zip \
  --out outputs/edge07_E0
```

For DA2 use the repository's existing spade_calib.camera.depth_inference.infer_da2 (official relative checkpoint, not metric). Install the repository/depth dependencies and set the actual DA2 path and checkpoint. Original images must match archived C0 pixels exactly. No automatic checkpoint downloads.

```bash
KITTI=/media/myungw00/256GB_SSD/kitti
python scripts/experiments/run_edge_observability.py \
  --m0 line_support.zip --c0 lines07_C0.zip \
  --ground ground_b4c.zip --motion trajectory_oracle.zip \
  --image-dir "$KITTI/data_odometry_color/sequences/07/image_2" \
  --da2-repo /path/to/Depth-Anything-V2 \
  --checkpoint /path/to/depth_anything_v2_vitl.pth \
  --encoder vitl --device cuda --out outputs/edge07_E0_depth
```

Alternatively `--depth-dir` points to original-resolution raw float .npy predictions named `000069.npy`, etc. External prediction provenance is not verified; only use same-image arrays. No depth-only detector implemented: depth arm ranks existing ELSED candidates by depth support, fusion arm weights the normal proxy using support. Depth cue is a local bilateral inverse-depth affine-profile fit, no metric normal or RGB line displacement. All thresholds/weights are pilot heuristics. DA2 real-inference integration not executed in the original pilot. Review rows are blank, red projected 3D/green image candidate; 2D overlays alone may not suffice for physical identity, use original L0 support when reviewing. Information is fixed-state raw GN with 1deg/.1m scaling, assumed 2px residual scale, equal matched-line mass, provisional weights .01/.1/1; not covariance. Postmatch compares up to two proposals per frame from same pool. Random is one deterministic draw, not a distribution. No performance conclusion on calibration or match precision.

## Archived results (no depth)

| Pre-match selection | selected | mean proxy min eigenvalue | unique proposals |
|---|---:|---:|---:|
| length | 80 | 0.0626 | 4 |
| random | 80 | 0.0507 | 4 |
| normal | 80 | 0.3053 | 1 |

Postmatch (16 selected; lambda=.1): baseline min eigenvalue .1334; length .1451; random .2023; normal .1539; full Jacobian .2431. These are conditional on unverified correspondence. Manual reference perpendicular distances at B4-c: frames 189/389/659 = 39.23/65.55/26.85 px. Existing distance gate 20px: capture range remains unresolved.

## Outputs

summary.json, features.csv, review.csv, frame overlays and review images. Original archive: edge_observability_E0.zip (not committed). Original synthetic checks: central/side vertical normals, affine-invariant synthetic linear/jump/crease profiles, analytic pixel Jacobian vs central finite differences.
