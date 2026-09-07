#!/usr/bin/env bash
# run_ablation_local.sh — leave-one-out constraint ablation over the 10-case
# reference (GT) set, LOCALLY.
# Runs sam2_propagate_ablation.py once per reference case, writing four volumes
# (full / no_continuity / no_depth / no_clip) from one shared SAM2 pass to
# <BASE>/ablation/<case>_<variant>.nii.gz (+ logs). Done cases skipped (FORCE=1).
#
# Usage:  ./run_ablation_local.sh [BASE=~/Downloads/output] [MODEL=large]
#   Env: SAM2_DIR (../sam2)  CLIP_SCALE (1.0)  MIN_COMP_PX (20)  VARIANTS
#        MAX_PROP_DEPTH (0=whole video)  FORCE (0)
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC="$ROOT/src"

BASE="${1:-$HOME/Downloads/output}"
MODEL="${2:-large}"
SAM2_DIR="${SAM2_DIR:-$ROOT/sam2}"
CLIP_SCALE="${CLIP_SCALE:-1.0}"
MIN_COMP_PX="${MIN_COMP_PX:-20}"
VARIANTS="${VARIANTS:-full,no_continuity,no_depth,no_clip}"
MAX_PROP_DEPTH="${MAX_PROP_DEPTH:-0}"
FORCE="${FORCE:-0}"

OUT="$BASE/ablation"; LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"

# The 10 reference cases (GT_MAP values in analysis/gt_compare.py).
CASES=(
    brac51681a_01
    brac52351a_01
    brac52351d_00
    brac54131f_00
    brac62451c_03
    brac62721b_00
    brac62731a_01
    dabj171d_02
    dabj201c_01
    dabj91c_02
)

echo "BASE=$BASE  MODEL=$MODEL"
echo "variants=$VARIANTS  clip_scale=$CLIP_SCALE  min_comp_px=$MIN_COMP_PX  max_prop_depth=$MAX_PROP_DEPTH"
echo "checkpoint: $SAM2_DIR/checkpoints/sam2.1_hiera_${MODEL}.pt"
echo

ok=0; fail=0; skip=0; i=0
for case in "${CASES[@]}"; do
    i=$((i+1))
    inp="$BASE/sam2_input/$case"
    lbl="$BASE/labels/${case}.nii.gz"
    stem="$OUT/${case}.nrrd"
    done_marker="$OUT/${case}_full.nii.gz"
    printf '[%d/%d] %-16s ... ' "$i" "${#CASES[@]}" "$case"

    if [ ! -d "$inp" ] || [ ! -f "$lbl" ]; then
        echo "SKIP (missing sam2_input/ or label)"; skip=$((skip+1)); continue
    fi
    if [ "$FORCE" != "1" ] && [ -f "$done_marker" ]; then
        echo "skip (already done)"; skip=$((skip+1)); continue
    fi

    t0=$(date +%s)
    if python "$SRC/sam2_propagate_ablation.py" \
            --sam2-dir        "$SAM2_DIR" \
            --input-dir       "$inp" \
            --label           "$lbl" \
            --output          "$stem" \
            --model           "$MODEL" \
            --clip-scale      "$CLIP_SCALE" \
            --min-comp-px     "$MIN_COMP_PX" \
            --variants        "$VARIANTS" \
            --max-prop-depth  "$MAX_PROP_DEPTH" \
            > "$LOGS/${case}.log" 2>&1; then
        echo "OK ($(( $(date +%s) - t0 ))s)"; ok=$((ok+1))
    else
        echo "FAIL -> $LOGS/${case}.log"; fail=$((fail+1))
    fi
done

echo
echo "Done.  ok=$ok  fail=$fail  skip=$skip"
echo "  volumes : $OUT/<case>_<variant>.nii.gz"
echo "  logs    : $LOGS/"
