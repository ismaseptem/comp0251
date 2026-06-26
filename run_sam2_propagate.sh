#!/bin/bash -l
#$ -S /bin/bash
#$ -N sam2_propagate
#$ -l h_rt=12:00:00
#$ -l mem=24G
#$ -l gpu=1
#$ -l tmpfs=20G
#$ -pe smp 4
#$ -j y
#$ -o logs/sam2_propagate_$JOB_ID.log
#$ -cwd

# ── User-defined paths (edit these) ──────────────────────────────────────────
CONDA_ENV="comp0251"           # name of your conda environment
SAM2_DIR="/home/ucabia4/sam2"            # root of the SAM2 repo clone
INPUT_DIR="/home/ucabia4/comp0251/sam2_input_test01"     # directory with frames/ and masks/ subdirs
LABEL="/home/ucabia4/comp0251/test01.nii.gz"       # original label file for spatial metadata
OUTPUT="/home/ucabia4/comp0251/propagated_label.nrrd"
MODEL="large"                       # tiny | small | base_plus | large
BATCH_SIZE=16                       # increase on A100/V100; reduce if OOM
SLICE_RESULTS="/home/ucabia4/comp0251/slice_01/"  # set to "" to skip per-slice PNGs

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load beta-modules
module load gcc-libs/10.2.0
module load cuda/12.2.2/gnu-10.2.0   # adjust to the CUDA version on Myriad

# Activate conda from user-installed miniforge
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

mkdir -p "$(dirname "$OUTPUT")"
mkdir -p logs

# ── Run ───────────────────────────────────────────────────────────────────────
ARGS=(
    --sam2-dir   "$SAM2_DIR"
    --input-dir  "$INPUT_DIR"
    --label      "$LABEL"
    --output     "$OUTPUT"
    --model      "$MODEL"
    --batch-size "$BATCH_SIZE"
)

if [ -n "$SLICE_RESULTS" ]; then
    ARGS+=(--slice-results "$SLICE_RESULTS")
fi

python sam2_propagate.py "${ARGS[@]}"

echo "==== Done: $(date) ===="
