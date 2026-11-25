#!/usr/bin/env bash
# Run from project root. Reuse existing W1/W5 factors; never overwrite.
set -uo pipefail
KITTI="${KITTI:-/media/myungw00/256GB_SSD/kitti}"
GT_POSES_DIR="${GT_POSES_DIR:-$KITTI/data_odometry_poses/poses}"
if (( $# )); then SEQS=("$@"); else SEQS=(07 09 10); fi
mkdir -p logs outputs || exit 1
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
failed=0
for SEQ in "${SEQS[@]}"; do
    (
        echo "=== H ablation $SEQ $(date) ==="
        FACTORS="outputs/scene${SEQ}_H_factors"
        REF="outputs/local_ba${SEQ}_H"
        OUT="outputs/local_ba${SEQ}_H_ablation_$STAMP"
        if [[ ! -f "$FACTORS/factors_w1/segment_07.npz" || ! -f "$FACTORS/factors_w5/segment_07.npz" ]]; then
            FACTORS="outputs/scene${SEQ}_H_factors_w1w5_$STAMP"
            python -u scripts/experiments/run_scene_consistency.py \
                --run "outputs/segment${SEQ}_trial1" --out "$FACTORS" \
                --windows 1 5 --center-stride 5 --support-mode voxel --local-voxel .18 --prepare-only || exit 1
        fi
        extra=()
        if [[ -f "$REF/Hfixed_HE/segment_00.npz" ]]; then extra+=(--reference-ba "$REF"); fi
        if [[ -f "$GT_POSES_DIR/$SEQ.txt" ]]; then extra+=(--gt-poses "$GT_POSES_DIR/$SEQ.txt"); else
            echo "GT pose file not found: $GT_POSES_DIR/$SEQ.txt; skipping trajectory GT metrics"
        fi
        if [[ "${EXTENDED:-1}" == 1 ]]; then extra+=(--extended-ablation); fi
        if [[ "${PROJECTION:-0}" == 1 ]]; then extra+=(--image-dir "$KITTI/data_odometry_color/sequences/${SEQ}/image_2"); fi
        python -u scripts/experiments/run_local_visual_ba.py \
            --run "outputs/segment${SEQ}_trial1" --ground-run "outputs/ground${SEQ}_B4c" \
            --scene-factors "$FACTORS" --out "$OUT" --window 5 --ablation-windows 1 5 \
            --scene-kind point --max-nfev 80 --visual-max-nfev 300 \
            --gt-calib "$KITTI/data_odometry_calib/sequences/${SEQ}/calib.txt" "${extra[@]}" || exit 1
        echo "Output: $OUT"
        python - "$OUT/summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
bad=[k for k,v in d["experiments"].items() if not v["success"]]
print("Nonconverged/invalid arms:",bad)
sys.exit(bool(bad))
PY
    ) 2>&1 | tee "logs/H_ablation_${SEQ}_$STAMP.log"
    status=${PIPESTATUS[0]}
    if (( status )); then failed=1; fi
    echo "=== $SEQ exit $status $(date) ==="
done
exit "$failed"
