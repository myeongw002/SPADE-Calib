# Migration milestones and acceptance gates

## Milestone 0 — bootstrap (merged PR #1)

- Installable Python package skeleton with validated camera/transform/cloud/plane models.
- Synthetic tests, coordinate conventions and original experiment inventory.

## Milestone 1A — legacy D1 input adapters (this PR)

- Read exact Joint/PGO XYZ, SfM anchors, aligned depth, plane CSV/masks and Hand–Eye files.
- Validate shapes, pixel counts, transforms and coordinate-frame labels.
- Preserve invalid depth pixels instead of silently repairing them.
- Support config-relative datasets and read-only input manifest with SHA-256 hashes.
- Add synthetic tests and an optional original D1 artifact-count regression.

**Gate:** Python package install, synthetic CI, and opt-in original D1 artifact
checks all pass. This does not constitute calibration result reproduction.

## Milestone 1B — freeze numerical D1 baseline (next)

- Preserve source ZIPs and private data outside the public repository.
- Record package/tool versions and archive SHA-256 hashes.
- Re-run exact legacy camera-guided / plane-only calibration.
- Store extrinsic matrices, correspondence IDs, optimizer status and held-out
  4-fold metrics in private golden references.
- Confirm initial Hand–Eye transform and PCD coordinate frames.

**Gate:** deterministic numeric regression for the original D1 pipeline.

## Milestone 2 — extract pure algorithm modules

- Refactor Camera ROI-free plane extraction and supporting geometry.
- Extract LiDAR deskew/refinement without conflating B/C1 and Joint/PGO.
- Extract Normal Diversity and guided plane matching with explicit budget/seed.
- Migrate original plane-only 6/8-DoF solver without changing loss semantics.
- Add dataset-independent tests with synthetic calibration scenes.

**Gate:** regression metrics match the frozen baseline at documented tolerances.

## Milestone 3 — unified runner

- Config-driven, deterministic intermediate artifacts; stage-level CLI.
- Data and config checksums, caching and reproducible experiment metadata.
- End-to-end D1 plane-only pipeline (reduced demo where permitted).

## Milestone 4 — new research method (separate changes)

- Reimplement scene-wide depth residual with explicit visibility/occlusion.
- Compare scene-only 6D, plane-only 6D, scene+plane 6D,
  scene+SfM 8D and scene+plane+SfM 8D.
- Reinstate Normal Diversity under equal search budgets and fixed RANSAC seeds.
- Restore D2/D3 common-support transfer evaluation.
- Study joint observability and physical plane-match reliability.

Do not equate internal depth consistency with true extrinsic accuracy.
