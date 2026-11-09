# D1 matching + plane-based calibration: numerical reproduction

**Status: reproduced from archived source; not yet ported.** This phase runs the original private Python scripts without editing research algorithms or their D1 parameters.

## Scope

Original experiment bundle: COMPA_D1_RefinedPCD_CameraGuided_Calibration_20261107.zip.
It includes Joint and PGO refined point clouds, Hand–Eye initialization, SfM / DA2 observations, camera-plane masks, saved correspondences and saved four-fold 6/8-DoF results.

- run_camera_guided_refined.py: Original Hand–Eye projection, guided RANSAC, geometry filters, duplicate suppression. Normal Diversity is ranking only; all 24 candidates are searched.
- run_calibration_refined.py: historical **plane-only** optimization; 8-DoF additionally optimizes inverse-depth scale and shift against SfM anchors.
- **Scene-wide depth is evaluated, but not included in the optimizer** in this baseline.
- The newer scene + plane + SfM Joint Optimization source is still unavailable.

## Reproduce privately

The original ZIP is intentionally **not published**. Only run trusted research archives; the following tool executes Python files included inside the ZIP.

~~~bash
python -m pip install -e '.[research,dev]'
python scripts/reproduce_d1.py \
  --archive /private/COMPA_D1_RefinedPCD_CameraGuided_Calibration_20261107.zip \
  --output outputs/d1_reproduction_report.json
~~~

The runner extracts files to an isolated temporary directory, moves the archived result directories to reference_matches_* and reference_calibration_*, reruns both refiners, checks the results, and removes the temporary raw data.

For debugging, --keep-workdir /private/new_directory preserves the private inputs and logs. This directory must be new and must not be committed.

Execution sets OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1 and PYTHONHASHSEED=0, with REFINER=Joint or PGO.

## Regression checks

1. Matching summary counts and exact ordered accepted plane IDs.
2. All matching CSV fields and rows.
3. Fitted LiDAR plane parameters.
4. **Exact** LiDAR support point-ID arrays.
5. All Calibration CSV rows, fold/method identities, status and numerical metrics.
6. All 4x4 Extrinsic matrices, across folds and ablations.
7. Status completed-fold/row counts; elapsed time is ignored.

Numerical default tolerances: atol=1e-6, rtol=1e-6. Cross-version BLAS/SciPy differences require investigation and explicitly documented tolerance changes.

## Verified local reproduction, 2026-11-09

| Artifact | Joint | PGO |
| --- | ---: | ---: |
| LiDAR points | 215,504 | 215,504 |
| Camera plane candidates | 24 | 24 |
| Accepted camera plane IDs | 0,6,1,5,3,19 | 0,6,1,5,3 |
| Accepted matches | 6 | 5 |
| Matching CSV rows | 24 | 24 |
| Point-support arrays | identical | identical |
| Plane parameter arrays | identical | identical |
| Calibration folds | 4 | 4 |
| Calibration CSV rows | 44 | 40 |
| Extrinsic matrices | 44 | 40 |
| Maximum absolute Extrinsic error | **0.0** | **0.0** |
| Maximum Calibration CSV numeric difference | **0.0** | **0.0** |

Verified with Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0 and OpenCV 4.13.0, Linux single-thread BLAS.

Original archive SHA-256: e7395f91e0d27c88bef1ce105168bc958dd81b162c8e007ddbe31343b4d215ab

This hash identifies the exact archive; it does not grant distribution rights.

## Limitations

- Reproducing computations does **not** establish ground-truth Extrinsic accuracy.
- Matched Plane geometric gates do not establish physical correspondence truth.
- The Camera Plane masks were constructed from a globally SfM-aligned depth map: the 4-fold SfM holdout is not fully prospective.
- Historical common_ids=[0,5,6,1,3] are hard-coded in the old optimizer.
- Original Joint/PGO Refined PCD creation is not repeated; both XYZ clouds are provided as inputs in the archive.
- Baseline reference outputs are in the same ZIP as the scripts; the archive SHA freezes the baseline package, not physical accuracy.

## CI and private opt-in

Synthetic comparator tests run on public GitHub Actions. Private numerical D1 regression must be opted into separately:

~~~bash
SPADE_D1_REPRO_ARCHIVE=/private/original.zip \
  pytest -q tests/regression/test_private_d1_optional.py
~~~

The generated JSON contains counts, array differences, versions and the input SHA, not raw image or point records.

## Next

Port the existing camera-guided matching and plane-only optimizer to reusable Python functions, preserving this baseline before adding the newer scene-wide loss.
