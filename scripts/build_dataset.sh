#!/usr/bin/env bash
# build_dataset.sh — local CPU stages for every case in a dataset dir.
# For each  <ROOT>/<case>_raw/  +  <ROOT>/<case>_roi.rois_series  it runs:
#   1) preflight.py            (gate — a case that FAILs is skipped)
#   2) roi_series_to_volume.py (label volume + QC overlay PNGs)
#   3) prepare_sam2.py         (SAM2 frames/ + seed masks/)
# and appends the case to <OUT>/cases.txt (the array-job manifest).
#
# Then eyeball <OUT>/qc/<case>/ before shipping <OUT>/{sam2_input,labels} to the
# cluster. All three stages are seconds of CPU — keep them local.
#
# Usage:  ./build_dataset.sh [ROOT=samples] [OUT=pipeline_out]
set -euo pipefail

ROOT="${1:-samples}"
OUT="${2:-pipeline_out}"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUT/labels" "$OUT/sam2_input" "$OUT/qc"
: > "$OUT/cases.txt"

shopt -s nullglob
n_ok=0 n_skip=0
for roi in "$ROOT"/*_roi.rois_series; do
    case="$(basename "$roi" _roi.rois_series)"
    raw="$ROOT/${case}_raw"

    if [ ! -d "$raw" ] || [ -z "$(ls -A "$raw" 2>/dev/null)" ]; then
        echo "[SKIP] $case: missing/empty $raw"; n_skip=$((n_skip+1)); continue
    fi

    echo "===================== $case ====================="
    if ! python "$HERE/preflight.py" --raw-dir "$raw" --rois "$roi"; then
        echo "[SKIP] $case: preflight FAILED — fix the export and re-run"
        n_skip=$((n_skip+1)); continue
    fi

    python "$HERE/roi_series_to_volume.py" \
        --rois "$roi" --raw-dir "$raw" \
        --output "$OUT/labels/${case}.nrrd" \
        --slice-results "$OUT/qc/${case}"

    python "$HERE/prepare_sam2.py" \
        --raw-dir "$raw" \
        --label "$OUT/labels/${case}.nii.gz" \
        --output "$OUT/sam2_input/${case}"

    echo "$case" >> "$OUT/cases.txt"
    n_ok=$((n_ok+1))
done

echo
echo "Prepared $n_ok case(s), skipped $n_skip.  Manifest: $OUT/cases.txt"
echo "Review $OUT/qc/<case>/ overlays, then rsync $OUT/{sam2_input,labels,cases.txt} to the cluster."
