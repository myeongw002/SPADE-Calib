# SPADE-Calib KITTI 07 Accumulation Pilot

## Scope / 의도

KITTI Odometry Sequence 07에서 **Fixed ±N Frames**와 **Fixed Euclidean Distance D**를 비교하는 1차 실험입니다.

파이프라인:

```
KITTI frame selection (target fixed)
  -> GLIM T_world_lidar row-index association
  -> SPADE-Calib existing accumulate_glim(mode='rigid')
  -> SPADE-Calib existing solve_joint(d1_roi=False)
  -> accumulated PLY / plane RANSAC / quality diagnostics
```

* 기존 사용자의 D1 Joint scan-level 6-DoF refinement를 포팅한 `spade_calib.lidar.joint_solver.solve_joint`를 **그대로 사용**합니다. 이번 패키지에서 새로운 pose optimizer를 구현하지 않았습니다.
* KITTI `.bin`은 `x y z reflectance` 4×float32 형식이며 **per-return timestamp/ring이 없습니다**. 따라서 Scan-level rigid accumulation만 수행하며 pointwise deskew는 하지 않습니다.
* 정식 KITTI Odometry LiDAR는 **HDL-64E(64-beam) 데이터**입니다. 이 실험은 기본적으로 원본 빔을 사용합니다. VLP-16 성능/결과라고 해석하면 안 됩니다. Low-resolution 모의 실험에는 **별도 재현 가능한 beam-subsetting 정책**이 필요합니다.
* 검출된 Ground/Wall은 단지 RANSAC **후보**입니다. 정적·물리적 평면 검증이나 Camera–LiDAR Plane Correspondence를 수행하지 않습니다.
* 최종 Extrinsic GT 정확도는 이 패키지에서 계산하지 않습니다. 목적은 Accumulation 및 Plane Quality의 비교입니다.

## Requirements

- Python >= 3.10
- `SPADE-Calib` 저장소를 clone한 뒤 저장소 루트에서 실행합니다. `main`의 함수 시그니처에 맞춰 작성했습니다.
- Python packages: numpy, scipy, matplotlib(프리뷰에만), pytest(테스트에만)

```bash
git clone https://github.com/myeongw002/SPADE-Calib.git
cd SPADE-Calib
python -m pip install -e '.[research,dev]'
python scripts/experiments/run_accumulation_experiment.py --help
```

`SPADE-Calib`의 Python 모듈이 import되지 않으면 `pip install -e '.[research]'`을 먼저 수행하세요.

## 입력 구조

```text
/path/to/KITTI/dataset/sequences/07/
  times.txt
  velodyne/
    000000.bin
    ...
    000384.bin
    ...
    000898.bin

/path/to/GLIM/output/odom_lidar.txt
```

**KITTI Frame index = 이미지 이름 숫자 = LiDAR `.bin` 파일 이름 숫자**라는 Odometry sequence 기본 대응을 가정합니다.

### 필수 확인: GLIM Pose indexing

현재 제공된 KITTI 07 샘플은 `times.txt` 1,101행, GLIM `odom_lidar.txt` 1,091행입니다. GLIM 첫 행의 **기록된 timestamp는 1.0초**이지만, KITTI GT와 GLIM의 상대 이동 속도 패턴을 비교했을 때 **GLIM row 0이 KITTI frame 0에 대응**하는 것이 가장 일치합니다(속도 크기 correlation 약 0.999). 그래서 **timestamp nearest-neighbor로 맞추면 틀릴 수 있습니다**.

기본 매핑은

```
GLIM row j -> KITTI frame j
```

이며, 다른 데이터셋에서 첫 Pose가 누락되었으면 `--pose-index-offset K`로 `GLIM row j -> KITTI frame j+K`를 명시적으로 변경합니다. Pose가 없는 끝 프레임에는 외삽하지 않습니다. 출력 `run_manifest.json`에 원래 GLIM timestamp와 실제 매핑을 기록합니다.

이는 **주어진 KITTI07 파일의 실험적 확인**이지, 모든 GLIM 로그에 적용되는 일반 규칙이 아닙니다.

## 0. 파일 없이 Scan Selection부터 확인

```bash
python scripts/experiments/run_accumulation_experiment.py \
  --times /path/to/KITTI/dataset/sequences/07/times.txt \
  --odom /path/to/GLIM/odom_lidar.txt \
  --window 005 --target-frame 384 \
  --half-frames 5 10 20 \
  --radii-m 2 5 10 \
  --match-scan-budget \
  --out results/selection_005 \
  --dry-run
```

`selected_scans.csv`와 `summary.csv`에 후보 Scan Index 및 실제 경로 이동 거리 등을 기록합니다. **원본 `.bin`이 없어도 가능**합니다.

## 1. 최초 실제 실험: Window 005 / Frame 384

처음에는 두 조건만 실행하세요. Pose Refinement가 계산 비용이 가장 큽니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/experiments/run_accumulation_experiment.py \
  --velodyne /path/to/KITTI/dataset/sequences/07/velodyne \
  --times /path/to/KITTI/dataset/sequences/07/times.txt \
  --odom /path/to/GLIM/odom_lidar.txt \
  --window 005 \
  --target-frame 384 \
  --half-frames 5 \
  --radii-m 5 \
  --refiner joint \
  --refine-iterations 4 \
  --out results/window_005
```

> **주의:** Frame 384 근방은 ±5 Frame과 5m 반경이 **정확히 동일한 Scan 11개**를 선택했습니다(현재 GLIM 궤적 기준). 따라서 이 경우 두 조건은 Algorithm Consistency/재현성 확인용으로 생각해야 하며, 선택 기준의 성능 차이를 입증하는 실험은 아닙니다.

## 2. 비교에 더 적합한 구간: Window 012 / Frame 898

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/experiments/run_accumulation_experiment.py \
  --velodyne /path/to/KITTI/dataset/sequences/07/velodyne \
  --times /path/to/KITTI/dataset/sequences/07/times.txt \
  --odom /path/to/GLIM/odom_lidar.txt \
  --window 012 \
  --target-frame 898 \
  --half-frames 5 10 \
  --radii-m 2 5 \
  --match-scan-budget \
  --refiner joint \
  --out results/window_012
```

Frame 898 근방에서는 ±5 Frame = 11 scans / 약 5.13m 경로, 5m 반경 = 20 scans / 약 9.84m 경로가 선택되었습니다. **같은 데이터량 비교**를 위해 `--match-scan-budget`을 사용하면 Fixed Frame과 동일한 N개를 공간적으로 가장 가까운 Scan에서 고르는 `distance_budget` 변형이 추가됩니다. 이는 순수 Fixed Radius와 구분해야 합니다.

### 빠른 누적 / RANSAC 검증 (Refinement 제외)

```bash
# 실제 저장소 설치 및 KITTI scans 경로는 필요
python scripts/experiments/run_accumulation_experiment.py \
  --velodyne /path/to/07/velodyne \
  --times /path/to/07/times.txt \
  --odom /path/to/odom_lidar.txt \
  --window 005 --target-frame 384 \
  --half-frames 5 --radii-m 5 \
  --refiner none \
  --out results/smoke_no_refinement
```

### 후보 Plane ROI 제약 (필요할 때만)

```bash
--roi 3 30 -20 20 -3 7
```

ROI는 **Target LiDAR Frame의 x/y/z 미터**이며 **Plane Detection에만 적용**됩니다. 기존 Joint Pose Refinement에는 이 ROI를 적용하지 않습니다. 값을 실제 PLY를 본 후 결정하세요. Camera SfM Plane의 Normal을 타깃 LiDAR Frame으로 변환해 이미 알고 있다면 `--wall-normal NX NY NZ --wall-normal-tol-deg 35`를 사용할 수 있습니다. 카메라 좌표의 normal을 그대로 넣으면 안 됩니다.

## 출력

```text
results/window_012/
  summary.csv
  selected_scans.csv
  plane_quality.csv
  run_manifest.json
  fixed_frame_n5/
    initial_scan_colors.ply
    initial_preview.png
    initial_sample.npz
    initial_planes.json
    refined_scan_colors.ply
    refined_preview.png
    refined_sample.npz
    refined_planes.json
    refined_corrections.npy
    refined_transforms.npy
    refinement.json
  fixed_distance_5m/
    ...
  distance_budget_n5/  # --match-scan-budget가 있을 때만
    ...
```

- `summary.csv`: 실행 방식, Scan 수, 누적 이동 거리, 시간 및 선택된 평면 후보 개수
- `selected_scans.csv`: 로컬 Scan ID ↔ 원래 KITTI frame ↔ target으로부터 거리
- `plane_quality.csv`: Plane 후보별 inliers, scan support, normal/offset, RMSE(m), tangent coverage eigenvalues(m²), split-half normal(°)/offset(m)
- `initial_*.ply`, `refined_*.ply`: **Voxel + Point cap으로 sampling된 시각화용 PLY**; 전체 raw 누적 Point를 저장한 것이 아닙니다. 원본 PCD는 필요 시 선정 frame 목록으로 재생성합니다.
- `*_sample.npz`: quality 계산에 사용된 downsampled xyz와 local scan ID. `source_frame_indices`로 KITTI frame 역추적 가능합니다.
- `refinement.json`: 실제 저장소 `solve_joint()` 진단과 local anchor 인덱스
- `refined_transforms.npy`: Refiner가 추정한 **GLIM에서 이미 target으로 누적된 Point에 대한 추가 correction**이며, raw Scan→Target LiDAR absolute 변환 자체는 아닙니다.

**Plane 측정값 해석 제한**

1. Inlier RMSE는 검출한 Plane으로 선택된 점의 **training residual**입니다. Ground truth 측정 정확도는 아닙니다.
2. Tangent coverage `coverage_eig_minor_m2`/`major_m2`는 편차의 분산(단위 m²)이며 실제 관측 면적이 아닙니다.
3. `normal_split_deg`, `offset_split_m`: 검출된 한 Plane의 Inlier를 Scan local ID 짝수/홀수로 나눠 독립적으로 PCA fitting한 **진단용 split-half 편차**입니다. 같은 GLIM/Refiner 추정치를 공유하므로 독립된 실제 Ground Truth가 아닙니다.
4. `non_ground`은 normal이 중력축과 다르다는 뜻만 보장합니다. 동일한 물리적 건물 외벽이라는 증거가 아닙니다. 나무/트럭 오검출 여부를 `*_preview.png`와 PLY로 확인해야 합니다.
5. Fixed Distance는 **target으로부터 Euclidean 거리 ≤ D**이고, Scene Window [start, stop]에도 제한됩니다. 시간 순서로 불연속인 Scan도 포함될 수 있습니다. `selected_scans.csv`를 확인하세요.
6. `joint_solver`는 D1에서 튜닝된 설정을 그대로 포팅한 구현입니다. KITTI07의 HDL-64E와 주행 속도에서 correspondence graph가 단절되지 않았는지 보장되지 않습니다. 첫 iteration의 매칭 Edge 수가 `scan_count-1`보다 작으면 경고를 출력합니다. 다만 충분한 Edge 수만으로 Graph Connectivity가 증명되지는 않습니다(`graph_connectivity_verified=false`). Refiner 결과를 Ground Truth처럼 해석하면 안 됩니다.

## Testing

```bash
python -m pip install pytest numpy scipy matplotlib
python -m pytest tests/test_accumulation_experiment.py -q
```

저장소의 `tests/test_accumulation_experiment.py`는 **실제 SPADE-Calib 모듈**을 사용해 Scan Selection, Rigid Accumulation, RANSAC/품질 진단의 합성 데이터 스모크 테스트를 수행합니다. KITTI 07 원본 데이터를 이용한 Joint 6-DoF Refinement 정확도는 별도 검증이 필요합니다.

이 스크립트 제작에 사용된 자료에는 KITTI07 `times.txt`/`odom_lidar.txt`는 있으나 Window 005/012의 전체 Velodyne `.bin` 파일은 없습니다. 따라서 제공된 궤적으로 **선택 Dry-run**을 했고, 실제 KITTI LiDAR를 사용한 Refinement 결과는 아직 없습니다.
