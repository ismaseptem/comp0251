#!/usr/bin/env bash
# run_clip_local.sh — run SAM2 clip propagation on every prepared case, LOCALLY.
# ---------------------------------------------------------------------------
# Loops the extraction output (build_dataset.sh: cases.txt + sam2_input/ +
# labels/) and runs sam2_propagate_clip.py on each case, sequentially, with the
# chosen model (default: small — fits a laptop GPU/MPS for an overnight run).
# Each case writes  <BASE>/out/<case>_clip.nrrd  + per-slice QC PNGs, and a full
# log under <BASE>/logs/. Cases already done are skipped so you can resume after
# an interrupt (set FORCE=1 to redo them).
#
# Usage:
#   ./run_clip_local.sh [BASE] [MODEL]
#     BASE  = extraction output dir (default: ~/Downloads/output)
#     MODEL = tiny|small|base_plus|large   (default: small)
#   Env overrides: SAM2_DIR (./sam2)  CLIP_SCALE (1.0)  MIN_COMP_PX (20)  FORCE (0)
#
# Examples:
#   ./run_clip_local.sh                                  # small, ~/Downloads/output
#   ./run_clip_local.sh ~/Downloads/output tiny          # faster, rougher
#   FORCE=1 ./run_clip_local.sh                          # recompute everything
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${1:-$HOME/Downloads/output}"
MODEL="${2:-small}"
SAM2_DIR="${SAM2_DIR:-$HERE/sam2}"
CLIP_SCALE="${CLIP_SCALE:-1.0}"
MIN_COMP_PX="${MIN_COMP_PX:-20}"
FORCE="${FORCE:-0}"

OUT="$BASE/out"; SLICES="$BASE/sam2_slices"; LOGS="$BASE/logs"
mkdir -p "$OUT" "$SLICES" "$LOGS"

# ── Build the case list: prefer cases.txt, else every sam2_input/<case> dir ──
# (plain while-read + array, so it works on macOS's bash 3.2 — no mapfile.)
CASES=()
if [ -f "$BASE/cases.txt" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        [ -n "$line" ] && CASES+=("$line")
    done < "$BASE/cases.txt"
else
    for d in "$BASE"/sam2_input/*/; do
        [ -d "$d" ] && CASES+=("$(basename "$d")")
    done
fi
[ "${#CASES[@]}" -gt 0 ] || { echo "No cases found under $BASE"; exit 1; }

echo "BASE=$BASE"
echo "MODEL=$MODEL  clip_scale=$CLIP_SCALE  min_comp_px=$MIN_COMP_PX  cases=${#CASES[@]}"
echo "checkpoint: $SAM2_DIR/checkpoints/sam2.1_hiera_${MODEL}.pt"
echo

ok=0; fail=0; skip=0; i=0
for case in "${CASES[@]}"; do
    i=$((i+1))
    inp="$BASE/sam2_input/$case"
    lbl="$BASE/labels/${case}.nii.gz"
    outfile="$OUT/${case}_clip.nrrd"
    printf '[%d/%d] %-20s ... ' "$i" "${#CASES[@]}" "$case"

    if [ ! -d "$inp" ] || [ ! -f "$lbl" ]; then
        echo "SKIP (missing sam2_input/ or label)"; skip=$((skip+1)); continue
    fi
    if [ "$FORCE" != "1" ] && [ -f "$outfile" ]; then
        echo "skip (already done)"; skip=$((skip+1)); continue
    fi

    t0=$(date +%s)
    if python "$HERE/sam2_propagate_clip.py" \
            --sam2-dir      "$SAM2_DIR" \
            --input-dir     "$inp" \
            --label         "$lbl" \
            --output        "$outfile" \
            --model         "$MODEL" \
            --clip-scale    "$CLIP_SCALE" \
            --min-comp-px   "$MIN_COMP_PX" \
            --slice-results "$SLICES/$case" \
            > "$LOGS/${case}.log" 2>&1; then
        echo "OK ($(( $(date +%s) - t0 ))s)"; ok=$((ok+1))
    else
        echo "FAIL -> $LOGS/${case}.log"; fail=$((fail+1))
    fi
done

echo
echo "Done.  ok=$ok  fail=$fail  skip=$skip"
echo "  labels : $OUT/"
echo "  QC     : $SLICES/"
echo "  logs   : $LOGS/"
