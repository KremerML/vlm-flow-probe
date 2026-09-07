#!/usr/bin/env bash
# Submit one vfp-* command as a one-GPU Slurm job. Run ON the cluster:
#
#     scripts/snellius_submit.sh <name> "<command>" [time] [partition] [extra sbatch args]
#
# e.g. scripts/snellius_submit.sh knockout "vfp-knockout --config configs/experiments/gemma3_4b/knockout_clevr_lite.yaml" 2-00:00:00
# Logs land in $HOME/logs/<name>-<jobid>.out. The command travels in a file
# rather than through --export, which splits its value at every comma (a
# "--layers 0,1,2" argument would be silently truncated to "--layers 0").
set -euo pipefail
NAME=${1:?name}; CMD=${2:?command}; TIME=${3:-24:00:00}; PART=${4:-gpu_a100}; EXTRA=${5:-}
mkdir -p "$HOME/logs"
cd "$(dirname "$0")/.."
CMD_FILE=$(mktemp "$HOME/logs/$NAME-XXXXXX.cmd")
printf '%s\n' "$CMD" > "$CMD_FILE"
# shellcheck disable=SC2086
sbatch --job-name="$NAME" --time="$TIME" --partition="$PART" $EXTRA \
    --output="$HOME/logs/%x-%j.out" \
    --export=ALL,VFP_CMD_FILE="$CMD_FILE" scripts/slurm/vfp.sbatch
