#!/bin/bash -l
# run_nnunet_train.sh — SGE array job: train nnU-Net v2, one task per (config, fold).
# ---------------------------------------------------------------------------
# The array index selects a configuration + CV fold from CONFIGS × {0..FOLDS-1}:
#     task 1..5   -> 3d_fullres  folds 0..4
#     task 6..10  -> 2d          folds 0..4      (only if CONFIGS has both)
# Submit with the array size = (#CONFIGS × FOLDS):
#
#     qsub -t 1-10 run_nnunet_train.sh          # both configs, 5 folds each
#     qsub -t 1-5  run_nnunet_train.sh          # just the first config
#
# ONE-TIME prep before the FIRST submit (CPU, run on a login/interactive node):
#     export nnUNet_raw=... nnUNet_preprocessed=... nnUNet_results=...
#     nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
#     # inject the patient-grouped split (nnU-Net wrote its own random one):
#     cp "$nnUNet_raw/Dataset501_Tumour/splits_final.json" \
#        "$nnUNet_preprocessed/Dataset501_Tumour/splits_final.json"
#
#$ -S /bin/bash
#$ -N nnunet_tr
#$ -l h_rt=48:00:00
#$ -l mem=8G
#$ -l gpu=1
#$ -l tmpfs=40G
#$ -pe smp 8
#$ -j y
#$ -o logs/nnunet_tr_$JOB_ID.$TASK_ID.log
#$ -cwd

# ── User-defined config (edit these) ─────────────────────────────────────────
CONDA_ENV="comp0251-gpu"
DATASET_ID=501
DATASET_NAME="Tumour"
FOLDS=5
CONFIGS=(3d_fullres 2d)          # drop "2d" to train only 3d_fullres
TRAINER="nnUNetTrainer"          # or nnUNetTrainer_250epochs for a fast first look

# nnU-Net data roots (must match what plan_and_preprocess used)
export nnUNet_raw="$HOME/comp0251/nnUNet_raw"
export nnUNet_preprocessed="$HOME/comp0251/nnUNet_preprocessed"
export nnUNet_results="$HOME/comp0251/nnUNet_results"
export nnUNet_n_proc_DA=4         # parallel augmentation workers, to keep the GPU fed. (Earlier "deadlocks"
                                  # were just buffered stdout, not real hangs — see PYTHONUNBUFFERED below.)
export nnUNet_compile=f           # disable torch.compile — it hung indefinitely before Epoch 0 on Myriad GPU nodes
export OMP_NUM_THREADS=1          # cap BLAS/OpenMP threads (harmless with DA=0; kept for safety)
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1         # flush stdout immediately — batch stdout is block-buffered, which
                                  # made healthy training look "stuck" in the SGE .log (it wasn't).
                                  # nnU-Net's own fold_*/training_log_*.txt is the authoritative progress log.

# ── Map SGE_TASK_ID → (config, fold) ─────────────────────────────────────────
idx=$((SGE_TASK_ID - 1))
cfg_i=$((idx / FOLDS))
fold=$((idx % FOLDS))
if [ "$cfg_i" -ge "${#CONFIGS[@]}" ]; then
    echo "Task $SGE_TASK_ID out of range for ${#CONFIGS[@]} config(s) × $FOLDS folds"; exit 1
fi
CONFIG="${CONFIGS[$cfg_i]}"

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load beta-modules
module load gcc-libs/10.2.0
module load cuda/12.2.2/gnu-10.2.0       # adjust to the cluster's CUDA
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p logs
echo "==== Task $SGE_TASK_ID → $CONFIG fold $fold on $(hostname) at $(date) ===="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"

# ── Sanity: preprocessing done + grouped split injected ──────────────────────
DS_PRE="$nnUNet_preprocessed/Dataset$(printf '%03d' "$DATASET_ID")_${DATASET_NAME}"
if [ ! -f "$DS_PRE/dataset_fingerprint.json" ]; then
    echo "ERROR: $DS_PRE not preprocessed. Run nnUNetv2_plan_and_preprocess -d $DATASET_ID first."; exit 1
fi
if [ ! -f "$DS_PRE/splits_final.json" ]; then
    echo "ERROR: no splits_final.json in $DS_PRE — copy the patient-grouped split in before training."; exit 1
fi

# ── Resume-aware training ────────────────────────────────────────────────────
# nnU-Net checkpoints per fold. If a fold already finished, skip it; if it was
# cut off by walltime, continue from the latest checkpoint (--c).
FOLD_DIR="$nnUNet_results/Dataset$(printf '%03d' "$DATASET_ID")_${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${fold}"
RESUME=""
if [ -f "$FOLD_DIR/checkpoint_final.pth" ]; then
    echo "fold already complete ($FOLD_DIR/checkpoint_final.pth) — nothing to do."; exit 0
elif [ -f "$FOLD_DIR/checkpoint_latest.pth" ]; then
    echo "resuming from checkpoint_latest.pth"; RESUME="--c"
fi

# diagnostic: prove the DA/compile fix is actually in effect + show shared-mem size
echo "DIAG compile=$nnUNet_compile  DA=$nnUNet_n_proc_DA  shm=$(df -h /dev/shm | tail -1)"

nnUNetv2_train "$DATASET_ID" "$CONFIG" "$fold" -tr "$TRAINER" $RESUME

echo "==== Done $CONFIG fold $fold: $(date) ===="
