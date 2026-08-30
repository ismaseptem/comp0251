#!/bin/bash -l
# run_sam2_array.sh — SGE array job: one SAM2 propagation task per case.
# Reads case names from $BASE/cases.txt (line N = task N), so submit with the
# array size matched to the manifest:
#
#     qsub -t 1-$(wc -l < $HOME/comp0251/pipeline_out/cases.txt) run_sam2_array.sh
#
# Each task consumes the locally-prepared inputs (frames/ + seed masks/) and
# writes  $BASE/out/<case>_propagated.nrrd  plus per-slice QC PNGs.
#$ -S /bin/bash
#$ -N sam2_prop
#$ -l h_rt=2:00:00
#$ -l mem=24G
#$ -l gpu=1
#$ -l tmpfs=20G
#$ -pe smp 4
#$ -j y
#$ -o logs/sam2_prop_$JOB_ID.$TASK_ID.log
#$ -cwd

# ── User-defined paths (edit these) ──────────────────────────────────────────
CONDA_ENV="comp0251"
SAM2_DIR="$HOME/sam2"
BASE="$HOME/comp0251/pipeline_out"     # holds cases.txt, sam2_input/, labels/
MODEL="large"                           # cluster GPU is cheap → use large
CLIP_SCALE=1.0                          # shrinking-ellipse size multiplier
MIN_COMP_PX=20                          # noise cull; matches seed threshold (keeps small lesions)

# ── Pick this task's case from the manifest ──────────────────────────────────
# Guard an unset/blank SGE_TASK_ID first: without it `sed -n "p"` prints EVERY
# line and the -z check below would pass the multi-line blob straight through
# (the same footgun that ran nnU-Net's job as "fold -1"). Submit with the array
# flag BEFORE the script:  qsub -t 1-N run_sam2_array.sh  (not ... run_... -t 1-N).
if ! [[ "$SGE_TASK_ID" =~ ^[0-9]+$ ]]; then
    echo "SGE_TASK_ID is '$SGE_TASK_ID', not a task number. Submit as: qsub -t 1-N $0"; exit 1
fi
case="$(sed -n "${SGE_TASK_ID}p" "$BASE/cases.txt")"
if [ -z "$case" ]; then
    echo "No case on line $SGE_TASK_ID of $BASE/cases.txt"; exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load beta-modules
module load gcc-libs/10.2.0
module load cuda/12.2.2/gnu-10.2.0       # adjust to the cluster's CUDA
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p "$BASE/out" "$BASE/sam2_slices/$case" logs
echo "==== Task $SGE_TASK_ID → $case on $(hostname) at $(date) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"

python sam2_propagate_clip.py \
    --sam2-dir        "$SAM2_DIR" \
    --input-dir       "$BASE/sam2_input/$case" \
    --label           "$BASE/labels/${case}.nii.gz" \
    --output          "$BASE/out/${case}_clip.nrrd" \
    --model           "$MODEL" \
    --clip-scale      "$CLIP_SCALE" \
    --min-comp-px     "$MIN_COMP_PX" \
    --slice-results   "$BASE/sam2_slices/$case"

echo "==== Done $case: $(date) ===="
