# Dataset configuration

The dataset inspection CLI is **implemented**. The complete calibration
pipeline and calibration-mode configuration are **not implemented** yet.

1. Extract the private D1 experiment archive outside this repository.
2. Copy examples/d1.yaml to a local config file.
3. Set dataset.root to the extracted inputs directory.
4. Verify lidar_reference and camera match the actual physical frames.
5. Run:

    spade-calib inspect --config path/to/local-d1.yaml --output outputs/d1_manifest.json

Relative dataset.root paths are resolved relative to the YAML file itself.
Relative paths inside inputs are resolved relative to dataset.root.
SHA-256 hashes record the exact file bytes; no raw points or images are
embedded in the JSON manifest.
