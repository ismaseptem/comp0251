#!/bin/bash -euo pipefail
#$ -S /bin/bash
#$ -N sam2_prepare
#$ -l h_rt=4:00:00
#$ -l mem=16G
#$ -l gpu=1
#$ -l tmpfs=10G
#$ -pe smp 4
#$ -j y
#$ -o logs/sam2_prepare_$JOB_ID.log
#$ -cwd

# ── User-defined paths (edit these) ──────────────────────────────────────────
CONDA_ENV="comp0251"
DCM_DIR="/home/ucabia4/comp0251/brac46551b_5316"  # annotated DICOM folder
SAM2_DIR="/home/ucabia4/sam2"                      # root of the SAM2 repo clone
OUTPUT_DIR="/home/ucabia4/comp0251/sam2_input"     # written here; pass same path to run_sam2_propagate.sh
MODEL="large"                                      # tiny | small | base_plus | large

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load beta-modules
module load gcc-libs/10.2.0
module load cuda/12.2.2/gnu-10.2.0

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── Sanity checks ─────────────────────────────────────────────────────────────
echo "==== Job info ===="
echo "  Job ID   : $JOB_ID"
echo "  Host     : $(hostname)"
echo "  Date     : $(date)"
echo "  Python   : $(which python)"
echo "  CUDA dev : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "=================="

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# ── Generate SAM2-predicted seed masks from annotated DICOMs ─────────────────
python prepare_sam2.py \
    --dcm-dir   "$DCM_DIR" \
    --sam2-dir  "$SAM2_DIR" \
    --output    "$OUTPUT_DIR" \
    --model     "$MODEL"

echo "==== Done: $(date) ===="
echo "Inspect masks in $OUTPUT_DIR/masks/ before running run_sam2_propagate.sh"
