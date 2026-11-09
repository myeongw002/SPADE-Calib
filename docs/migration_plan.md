# Migration milestones and acceptance gates

## Milestone 0 — bootstrap (this PR)

- Package skeleton, basic typed/validated geometry models and synthetic tests.
- Explicit coordinate conventions and provisional interchange schema.
- Legacy dependency inventory and public-data guardrails.
- No promise of a runnable calibration pipeline.

**Gate:** `pip install -e ".[dev]"` and `pytest` pass in clean environments.

## Milestone 1 — freeze reproducible D1 baseline

- Keep source ZIPs and raw data **outside** the public repository.
- Record package/tool versions and archive SHA-256 hashes.
- Re-execute existing D1 plane-only matching/calibration, capture matrices, match IDs, 4-fold metrics and optimizer status.
- Confirm consistent initial Hand–Eye transform and coordinate frames.

**Gate:** a script compares reproduced results to saved baselines using documented tolerances.

## Milestone 2 — adapters, then algorithms

- Implement IO adapters for the exact legacy NPY/NPZ/CSV/YAML formats.
- Extract pure functions for camera backprojection and plane extraction.
- Extract LiDAR deskew/refinement functions (do not silently conflate B/C1 and Joint/PGO).
- Extract Normal Diversity and guided matching with explicit budget/seed.
- Migrate plane-only 6/8-DoF optimization *unchanged* before revising any residual.

**Gate:** same D1 results for the same input data and seeds.

## Milestone 3 — unified runner

- Configuration-driven stage runner/CLI, deterministic output directories.
- Input/config hash provenance, caching, validation and standard logs.
- End-to-end D1 pipeline plus non-data-dependent integration tests.

## Milestone 4 — research-method restoration (separate from refactor)

- Reimplement scene-wide depth residual with visibility/occlusion rules.
- Compare scene-only 6-DoF, plane-only 6-DoF, scene+plane 6-DoF, scene+SfM 8-DoF, scene+plane+SfM 8-DoF.
- Reintroduce Normal Diversity with **equal search budgets** and fixed per-plane RANSAC seeds.
- Restore D2/D3 fixed-common-support evaluation.
- Study correspondence reliability, plane-position uncertainty and extrinsic/depth coupling.

**Gate:** no performance claim beyond the metrics actually measured; independent extrinsic truth required for accuracy claims.

## Publication safeguards

The GitHub repository is public. Do not upload raw camera imagery, scan archives, SfM reconstructions, positional metadata, credentials, or published/unpublished partner data without checking rights and privacy. Code can also contain personally identifying absolute paths; review before migrating.

Do not select a software license on the user's behalf.
