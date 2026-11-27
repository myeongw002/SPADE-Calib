#!/usr/bin/env bash
# Run from SPADE-Calib root, active Python environment. Never delete outputs.
set -uo pipefail

KITTI="${KITTI:-/media/myungw00/256GB_SSD/kitti}"
CENTER_STRIDE="${CENTER_STRIDE:-5}"
VOXEL="${VOXEL:-0.18}"
WINDOW=5
if (( $# )); then SEQUENCES=("$@"); else SEQUENCES=(07 09 10); fi
mkdir -p outputs logs || exit 1
STAMP="$(date +%Y%m%d-%H%M%S)-$$"
failed=0

valid_factors() {
    python - "$1" "$2" "$CENTER_STRIDE" "$VOXEL" "$WINDOW" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np
folder, run = map(Path, sys.argv[1:3])
stride, voxel, window = int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
try:
    d=json.loads((folder/"summary.json").read_text())
    fit=json.loads((run/"segmented_handeye.json").read_text())["fit"]
    assert Path(d["run"]).name==run.name
    assert d["settings"]["prepare_only"] and d["settings"]["center_stride"]==stride
    assert d["support_mode"]=="voxel" and np.isclose(d["local_voxel_m"],voxel)
    assert window in d["settings"]["windows"]
    assert np.allclose(d["baseline"]["T_camera_lidar"],fit["T_camera_lidar"],atol=1e-9)
    assert np.isclose(d["recomputed_cost"],fit["optimization"]["cost_whitened_robust"],rtol=1e-9)
    assert len(d["segments"])==len(fit["segments"])
    for k in range(len(fit["segments"])):
        with np.load(folder/f"factors_w{window}"/f"segment_{k:02d}.npz",allow_pickle=False) as z:
            n=len(z["frames"])
            assert n>=3 and np.array_equal(z["frames"],d["segments"][k]["selected_centers"])
            for key,shape in (("Ra",(n,3,3)),("Rb",(n,3,3)),("ta",(n,3)),("tb",(n,3)),
                              ("point_factors",(n,12,12)),("plane_factors",(n,12,12)),("point_counts",(n,))):
                assert z[key].shape==shape and np.isfinite(z[key]).all()
except (OSError,ValueError,KeyError,AssertionError,IndexError,TypeError) as e:
    print(f"Cannot reuse {folder}: {e}",file=sys.stderr)
    sys.exit(1)
PY
}

for SEQ in "${SEQUENCES[@]}"; do
    (
        echo "=== KITTI ${SEQ} START: $(date) ==="
        RUN="outputs/segment${SEQ}_trial1"
        FACTORS="outputs/scene${SEQ}_H_factors"
        OUT="outputs/local_ba${SEQ}_H"
        if valid_factors "$FACTORS" "$RUN"; then
            echo "Reusing complete factor archive: $FACTORS"
        else
            if [[ -e "$FACTORS" ]]; then
                FACTORS="$(mktemp -d "outputs/scene${SEQ}_H_factors_retry.XXXXXX")" || exit 1
            fi
            echo "Preparing factors: $FACTORS"
            python -u scripts/experiments/run_scene_consistency.py \
                --run "$RUN" --out "$FACTORS" --windows "$WINDOW" \
                --center-stride "$CENTER_STRIDE" --support-mode voxel \
                --local-voxel "$VOXEL" --prepare-only || exit 1
            valid_factors "$FACTORS" "$RUN" || exit 1
        fi
        if [[ -e "$OUT" ]]; then
            OUT="$(mktemp -d "outputs/local_ba${SEQ}_H_retry.XXXXXX")" || exit 1
        fi
        echo "BA output: $OUT"
        python -u scripts/experiments/run_local_visual_ba.py \
            --run "$RUN" --ground-run "outputs/ground${SEQ}_B4c" \
            --scene-factors "$FACTORS" --out "$OUT" --window "$WINDOW" --scene-kind point \
            --gt-calib "$KITTI/data_odometry_calib/sequences/${SEQ}/calib.txt" \
            --image-dir "$KITTI/data_odometry_color/sequences/${SEQ}/image_2" || exit 1
        # A completed program can still report nonconvergence in a fit.
        python - "$OUT/summary.json" <<'PY'
import json,sys
with open(sys.argv[1]) as f:
    d=json.load(f)
expected={"Hfixed_HE","Hfixed_scene","H0_visual","H0_then_HE","H0_then_scene","H1_joint_HE","H2_joint_scene"}
bad=sorted(k for k in expected if not d["experiments"].get(k,{}).get("success",False))
if bad:
    print("Incomplete/nonconverged BA arms:",", ".join(bad))
    sys.exit(2)
PY
    ) 2>&1 | tee "logs/H_${SEQ}_${STAMP}.log"
    status=${PIPESTATUS[0]}
    if (( status == 0 )); then
        echo "KITTI $SEQ: SUCCESS"
    else
        echo "KITTI $SEQ: FAILED or NONCONVERGED (exit $status)"
        failed=1
    fi
    echo "=== KITTI ${SEQ} END: $(date) ==="
done
echo "ALL SEQUENCES FINISHED: $(date), failure=$failed"
exit "$failed"
