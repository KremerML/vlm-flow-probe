#!/bin/bash
# Generic one-GPU job for any vfp-* command. Submit through scripts/snellius_submit.sh.
#SBATCH --job-name=vfp
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "${VFP_ROOT:-$HOME/vlm-flow-probe}"
# shellcheck disable=SC1091
source scripts/snellius_env.sh
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
echo "== $(date) host=$(hostname) job=$SLURM_JOB_ID gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
VFP_CMD=${VFP_CMD:-$(cat "$VFP_CMD_FILE")}
echo "== $VFP_CMD"
eval "$VFP_CMD"
echo "== done $(date)"
