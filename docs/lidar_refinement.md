# D1 LiDAR Refinement modules (phase 4)

## Source provenance and what is NOT being claimed

- **A:** per-return GLIM trajectory deskew of the 10 original VLP-16 scans (target/anchor scan 5). This source and input data are included in the private 2026-11-06 COMPA_D1_B_C1_Refinement_ReDeskew archive.
- **B:** new 2026-11-06 robust, regularized scan-pair joint 6-DoF point-to-plane estimator applied on A. Scan 5 is fixed. Training deliberately excludes the historical wall ROI.
- **C1:** time interpolation of B's scan corrections applied to identical A returns. This is **not** an independent solver or continuous-time IMU inference.
- **Joint/PGO:** the separate historical Joint and PGO refined clouds used in D1 Calibration are **not** reproduced here. Their original complete pose correction sources have not yet been identified. Do not rename B/C1 as Joint/PGO.

## Implemented

- `lidar.scan_io`: strict D1 binary PCD `FIELDS x y z intensity ring time`, with 22-byte records and original return IDs; exported ZIP + manifest + embedded GLIM trajectory. Keeps source return identity and scan ID.
- `lidar.trajectory`: GLIM poses as `T_world_lidar`. Quaternion slerp and linear position interpolation; no extrapolation. Uses relative timestamps to avoid precision loss with Unix nanoseconds.
- `lidar.deskew`: `T_target_source = inv(T_world_lidar(target)) @ T_world_lidar(source_return_time)`; rigid baseline option; float32 output, stable original record order.
- `lidar.pose_refinement`: reference D1 B (robust point-to-plane) and C1 correction application. Exact historical D1 algorithm thresholds/seed, not generalized to other scenes yet.
- `lidar.ghost_filter`: **optional** target-scan free-space consistency filter. Only non-target points in front of target returns (same angle-bin) are removed. This does not guarantee removing all ghost points. It is not enabled in A/B/C1 reproduction by default.

## Usage

~~~bash
pip install -e '.[research,dev]'
python scripts/prepare_d1_refinement.py \
  --archive /private/rosbag2_2026_10_08-15_05_54_1791439637016960512_distance.zip \
  --output outputs/d1_A --variant A

python scripts/prepare_d1_refinement.py \
  --archive /private/rosbag2_2026_10_08-15_05_54_1791439637016960512_distance.zip \
  --output outputs/d1_B --variant B

# Explicitly change input point population; NOT the historical baseline.
python scripts/prepare_d1_refinement.py \
  --archive /private/rosbag2_2026_10_08-15_05_54_1791439637016960512_distance.zip \
  --output outputs/d1_A_freespace --variant A --ghost-filter
~~~

Output is compressed NPZ with `xyz`, `scan_id`, `ring`, `time_offset_s`, `source_return_index`, and `reference_timestamp_ns`, plus JSON provenance. The full raw input ZIP and raw output NPZ should remain local and out of the repository.

## Reproduction results / baseline gates (2026-11-09)

- Original D1 archive contains 10 scans and **215,504** valid VLP-16 returns.
- Reconstructed A XYZ matches `D1_original_aligned_input.npz` with **maximum Euclidean distance 0.0 m** in the present test environment, same point ordering.
- B 5-step optimizer uses 22 scan pairs and 16,500 match samples each iteration. Compared to the original B corrections CSV, max translation-component difference is **0.000081 m**, max rotation-component difference is approximately **0.0000082 rad**. Small cross-run drift can arise in neighbor/least-squares order and numerical libraries; unlike A this is not bitwise reproduction.
- Point-level B/C1 comparisons to the archived scan-colored PLY using unchanged source ordering: B max XYZ deviation **0.0007134 m** (P95 **0.0001296 m**), C1 max XYZ deviation **0.0006944 m** (P95 **0.0001199 m**). Both preserve all 215,504 returns. These are approximate numerical reproductions; neither is bitwise-identical to the archived PLY.
- Optional A target-frame free-space filter with 0.2° azimuth, 2.0° elevation, 0.5 m margin rejected **33,384 / 215,504** points, leaving **182,120**. This is **not** a validated ghost detection precision or calibration accuracy improvement. It only counts geometric contradictions to scan 5.
- Synthetic tests cover time interpolation, projection frames, per-return vs rigid motion, source identity, filtering conditions. An optional local D1 test compares against the original A PCD; publicly shared CI does not receive raw data.

## Known limitations and engineering risks

1. B/C1 rigid corrections shift some points across range-map bins; target-based free space is a heuristic and depends on the chosen reference scan. It must not be misrepresented as a scene-wide ground-truth ghost mask.
2. A/B/C1 hold all source returns fixed. Applying the ghost filter **changes sample count and support**; any Calibration comparison must use matched point IDs/common support or explicitly treat support changes as an ablation variable.
3. The time-relative convention of the archived PCD is **negative offsets (~ -0.1 s to 0)**; treating `time` as an absolute Unix timestamp is incorrect.
4. A reproduction and B/C1 evaluation require different precision/tolerance gates. 215,504 points are represented as float32 in original A; B corrections involve cKDTree and sparse iterative LSMR.
5. No external pose or extrinsic ground truth. Better internal plane fit does not prove improved extrinsic accuracy.
6. Historical Joint/PGO production remains a **separate unfinished provenance task**; do not feed the new B result into the archived Joint/PGO D1 Regression and claim numerical equivalence.
