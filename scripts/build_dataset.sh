#!/usr/bin/env bash
# build_dataset.sh — local CPU stages for every case in a dataset dir.
# Per case (<ROOT>/<case>_raw/ + <ROOT>/<case>_roi.rois_series): preflight.py
# (gate) -> roi_series_to_volume[_split].py (label + QC) -> prepare_sam2.py
# (frames/ + seed masks/), appending passing cases to <OUT>/cases.txt. Stage 2
# defaults to the SPLIT variant (SPLIT=0 for plain OR). No `set -e`: a bad case
# is counted and skipped, not fatal.
#
# Usage:  ./build_dataset.sh [ROOT=samples] [OUT=pipeline_out]
#   Env: SPLIT (1)  GAP_PX ()  FORCE (0)
set -uo pipefail

ROOT="${1:-samples}"
OUT="${2:-pipeline_out}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SPLIT="${SPLIT:-1}"
GAP_PX="${GAP_PX:-}"
FORCE="${FORCE:-0}"

if [ "$SPLIT" = "1" ]; then
    VOL_SCRIPT="$HERE/roi_series_to_volume_split.py"
else
    VOL_SCRIPT="$HERE/roi_series_to_volume.py"
fi
[ -f "$VOL_SCRIPT" ] || { echo "Missing volume script: $VOL_SCRIPT"; exit 1; }

GAP_ARGS=()
[ -n "$GAP_PX" ] && GAP_ARGS=(--gap-px "$GAP_PX")

mkdir -p "$OUT/labels" "$OUT/sam2_input" "$OUT/qc" "$OUT/logs"

echo "ROOT=$ROOT  OUT=$OUT"
echo "stage 2: $(basename "$VOL_SCRIPT")${GAP_PX:+  --gap-px $GAP_PX}   FORCE=$FORCE"
echo

shopt -s nullglob
DONE=()
n_ok=0 n_skip=0 n_fail=0 i=0
ROIS=("$ROOT"/*_roi.rois_series)
[ "${#ROIS[@]}" -gt 0 ] || { echo "No *_roi.rois_series found under $ROOT"; exit 1; }

for roi in "${ROIS[@]}"; do
    i=$((i+1))
    case="$(basename "$roi" _roi.rois_series)"
    raw="$ROOT/${case}_raw"
    lbl="$OUT/labels/${case}.nii.gz"
    inp="$OUT/sam2_input/${case}"
    log="$OUT/logs/${case}.log"
    printf '[%d/%d] %-20s ... ' "$i" "${#ROIS[@]}" "$case"

    if [ ! -d "$raw" ] || [ -z "$(ls -A "$raw" 2>/dev/null)" ]; then
        echo "SKIP (missing/empty $raw)"; n_skip=$((n_skip+1)); continue
    fi

    # Resume: a case counts as done only if BOTH the label and the seed masks
    # exist, so a run interrupted between stage 2 and 3 is redone, not trusted.
    if [ "$FORCE" != "1" ] && [ -f "$lbl" ] && [ -d "$inp/masks" ] \
       && [ -n "$(ls -A "$inp/masks" 2>/dev/null)" ]; then
        echo "skip (already done)"; DONE+=("$case"); n_skip=$((n_skip+1)); continue
    fi

    : > "$log"
    if ! python "$HERE/preflight.py" --raw-dir "$raw" --rois "$roi" >>"$log" 2>&1; then
        echo "SKIP (preflight FAILED -> $log)"; n_skip=$((n_skip+1)); continue
    fi

    if ! python "$VOL_SCRIPT" \
            --rois "$roi" --raw-dir "$raw" \
            --output "$OUT/labels/${case}.nrrd" \
            ${GAP_ARGS[@]+"${GAP_ARGS[@]}"} \
            --slice-results "$OUT/qc/${case}" >>"$log" 2>&1; then
        echo "FAIL (volume -> $log)"; n_fail=$((n_fail+1)); continue
    fi

    if ! python "$HERE/prepare_sam2.py" \
            --raw-dir "$raw" \
            --label "$lbl" \
            --output "$inp" >>"$log" 2>&1; then
        echo "FAIL (prepare_sam2 -> $log)"; n_fail=$((n_fail+1)); continue
    fi

    echo "OK"
    DONE+=("$case")
    n_ok=$((n_ok+1))
done

# Manifest is rewritten from the cases that actually have artefacts on disk —
# including ones skipped as already-done — so an interrupted or partial run
# never leaves a silently truncated cases.txt behind.
: > "$OUT/cases.txt"
[ "${#DONE[@]}" -gt 0 ] && printf '%s\n' "${DONE[@]}" > "$OUT/cases.txt"

echo
echo "Done.  ok=$n_ok  fail=$n_fail  skip=$n_skip   manifest: $OUT/cases.txt ($(grep -c . "$OUT/cases.txt" 2>/dev/null || echo 0) cases)"
echo "Review $OUT/qc/<case>/ overlays, then rsync $OUT/{sam2_input,labels,cases.txt} to the cluster."
[ "$n_fail" -eq 0 ] || echo "WARNING: $n_fail case(s) failed — see $OUT/logs/"
