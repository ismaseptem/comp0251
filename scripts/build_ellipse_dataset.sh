#!/usr/bin/env bash
# build_ellipse_dataset.sh — assemble an nnU-Net dataset whose labels are the
# MODEL-FREE ellipse baseline (ellipse_baseline.py) instead of the SAM2 clip.
# Two local CPU stages: (1) ellipse_baseline.py per case -> <OUT>/ellipse/, using
# the seed masks + grid-reference label already built by build_dataset.sh;
# (2) prepare_nnunet.py -> nnUNet_raw/Dataset<ID>_<NAME>/. No `set -e`.
#
# Usage:  ./build_ellipse_dataset.sh [RAW_ROOT] [OUT] [DATASET_ID] [DATASET_NAME]
#   defaults: RAW_ROOT=~/Downloads/samples  OUT=~/Downloads/output
#             DATASET_ID=503  DATASET_NAME=TumourEllipse
#   Env: CASES (<OUT>/cases.txt)  FORCE (0)  NNUNET_RAW (nnUNet_raw)
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
RAW_ROOT="${1:-$HOME/Downloads/samples}"
OUT="${2:-$HOME/Downloads/output}"
DATASET_ID="${3:-503}"
DATASET_NAME="${4:-TumourEllipse}"
CASES="${CASES:-$OUT/cases.txt}"
FORCE="${FORCE:-0}"
NNUNET_RAW="${NNUNET_RAW:-$HERE/nnUNet_raw}"

ELL_DIR="$OUT/ellipse"
mkdir -p "$ELL_DIR"

[ -f "$CASES" ] || { echo "Missing manifest: $CASES"; exit 1; }
CASE_LIST=()
while IFS= read -r line; do
    [ -n "$line" ] && CASE_LIST+=("$line")
done < "$CASES"
[ "${#CASE_LIST[@]}" -gt 0 ] || { echo "No cases in $CASES"; exit 1; }

echo "RAW_ROOT=$RAW_ROOT"
echo "OUT=$OUT   ellipse labels -> $ELL_DIR"
echo "Dataset  Dataset${DATASET_ID}_${DATASET_NAME}  -> $NNUNET_RAW"
echo "Cases    ${#CASE_LIST[@]}   FORCE=$FORCE"
echo

# ── Stage 1: ellipse baseline label for every case ────────────────────────
n_ok=0 n_skip=0 n_fail=0 i=0
for case in "${CASE_LIST[@]}"; do
    i=$((i+1))
    seed="$OUT/sam2_input/$case"
    ref="$OUT/labels/$case.nii.gz"
    lbl="$ELL_DIR/${case}_ellipse.nii.gz"
    printf '[%d/%d] %-20s ... ' "$i" "${#CASE_LIST[@]}" "$case"

    if [ ! -d "$seed/masks" ] || [ -z "$(ls -A "$seed/masks" 2>/dev/null)" ]; then
        echo "SKIP (no seed masks in $seed/masks)"; n_skip=$((n_skip+1)); continue
    fi
    if [ ! -f "$ref" ]; then
        echo "SKIP (no grid-ref label $ref)"; n_skip=$((n_skip+1)); continue
    fi
    if [ "$FORCE" != "1" ] && [ -f "$lbl" ]; then
        echo "skip (already built)"; n_ok=$((n_ok+1)); continue
    fi

    if python "$HERE/ellipse_baseline.py" \
            --input-dir "$seed" \
            --label "$ref" \
            --output "$ELL_DIR/${case}_ellipse.nrrd" >/dev/null 2>&1; then
        echo "OK"; n_ok=$((n_ok+1))
    else
        echo "FAIL (ellipse_baseline)"; n_fail=$((n_fail+1))
    fi
done

echo
echo "Ellipse labels: ok=$n_ok  fail=$n_fail  skip=$n_skip   -> $ELL_DIR"
[ "$n_ok" -gt 0 ] || { echo "Nothing built — aborting before prepare_nnunet."; exit 1; }
echo

# ── Stage 2: assemble the nnU-Net raw dataset ─────────────────────────────
python "$HERE/prepare_nnunet.py" \
    --raw-root     "$RAW_ROOT" \
    --labels-dir   "$ELL_DIR" \
    --label-suffix _ellipse \
    --cases        "$CASES" \
    --dataset-id   "$DATASET_ID" \
    --dataset-name "$DATASET_NAME" \
    --output       "$NNUNET_RAW"
