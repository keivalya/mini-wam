#!/bin/bash
#SBATCH --job-name=pwam-supervision
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --array=0-4%2
#SBATCH --output=logs/supervision-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
PY="${PICO_WAM_PY:-python}"
SEED=${SLURM_ARRAY_TASK_ID}
RUN="runs/job-${SLURM_ARRAY_JOB_ID}/seed-${SEED}"
mkdir -p "$RUN"
$PY -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(), flush=True)'
$PY -m scripts.collect_data --env toy --episodes 200 --max-steps 80 --seed "$((42000 + 1000 * SEED))" --random-action-prob 0.3 --output-path "$RUN/data.npz"
$PY -m scripts.supervision_ablation --dataset "$RUN/data.npz" --seed "$SEED" --epochs 15 --device cuda --output "$RUN/ablation"
