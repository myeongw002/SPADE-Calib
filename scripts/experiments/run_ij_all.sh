#!/usr/bin/env bash
# Reuse H factors and H visual reference; from repository root, active venv.
set -uo pipefail
KITTI="${KITTI:-/media/myungw00/256GB_SSD/kitti}"
GT_POSES_DIR="${GT_POSES_DIR:-$KITTI/data_odometry_poses/poses}"
MODE="${IJ_MODE:-both}"
if (( $# )); then SEQS=("$@"); else SEQS=(07 09 10); fi
mkdir -p logs outputs || exit 1
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
failed=0
for SEQ in "${SEQS[@]}"; do
    (
        echo "=== IJ $SEQ $MODE $(date) ==="
        FACTORS="outputs/scene${SEQ}_H_factors"
        REF="outputs/local_ba${SEQ}_H"
        OUT="outputs/local_ba${SEQ}_IJ_$STAMP"
        if [[ ! -f "$FACTORS/factors_w1/segment_07.npz" || ! -f "$FACTORS/factors_w5/segment_07.npz" ]]; then
            FACTORS="outputs/scene${SEQ}_IJ_factors_$STAMP"
            python -u scripts/experiments/run_scene_consistency.py \
                --run "outputs/segment${SEQ}_trial1" --out "$FACTORS" \
                --windows 1 5 --center-stride 5 --support-mode voxel --local-voxel .18 --prepare-only || exit 1
        fi
        extra=()
        if [[ -f "$REF/Hfixed_HE/segment_00.npz" ]]; then extra+=(--reference-ba "$REF"); else
            echo "Prior H reference missing: $REF; cannot verify exact H support"
            exit 1
        fi
        if [[ -f "$GT_POSES_DIR/$SEQ.txt" ]]; then extra+=(--gt-poses "$GT_POSES_DIR/$SEQ.txt"); else
            echo "GT pose file missing; trajectory evaluation skipped"
        fi
        python -u scripts/experiments/run_local_visual_ba.py \
            --run "outputs/segment${SEQ}_trial1" --ground-run "outputs/ground${SEQ}_B4c" \
            --scene-factors "$FACTORS" --out "$OUT" --window 5 --ij --ij-mode "$MODE" \
            --shuffle-repeats "${SHUFFLE_REPEATS:-3}" --information-maxiter "${INFO_MAXITER:-10000}" \
            --gt-calib "$KITTI/data_odometry_calib/sequences/${SEQ}/calib.txt" "${extra[@]}" || exit 1
        echo "Output: $OUT"
        python - "$OUT/summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));bad=[];unreliable=[]
for k,v in d["experiments"].items():
    if not v["success"]: bad.append(k)
    info=v.get("information")
    if info and not v.get("alias_of"):
        for state in ("common_initial","solution"):
            for mode,entries in info[state].items():
                for label,r in entries.items():
                    if not r["reliable"]: unreliable.append("/".join((k,state,mode,label)))
print("Nonconverged arms:",bad)
print("Unreliable information projections:",unreliable)
sys.exit(bool(bad or unreliable))
PY
    ) 2>&1 | tee "logs/IJ_${SEQ}_$STAMP.log"
    status=${PIPESTATUS[0]}
    if (( status )); then failed=1; fi
    echo "=== $SEQ exit $status $(date) ==="
done
exit "$failed"
