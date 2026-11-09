# Legacy experiment code

Original scripts are deliberately **not yet published**. They are inventoried in [docs/legacy_inventory.md](../docs/legacy_inventory.md).

Migration policy:
1. Keep original archive SHA-256 and exact inputs in a private experiment store.
2. Verify publication permission and remove personal/local paths and undistributable data before publishing code.
3. Preserve original files unmodified as baseline references.
4. Move tested functionality into `src/spade_calib`; do not wire the legacy scripts directly into the public CLI.
