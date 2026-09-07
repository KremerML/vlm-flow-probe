#!/usr/bin/env bash
# One-time Snellius setup, safe to re-run. From the workstation, push the tree
# first (the repo is private, the cluster cannot clone it), then run this on a
# login node:
#
#     scripts/snellius_sync.sh                                  # local
#     ssh snellius 'cd vlm-flow-probe && scripts/snellius_setup.sh'
#
# Login nodes have outbound internet; compute nodes are not assumed to, so this
# also pre-fetches nothing but pip wheels — model weights are pulled here too,
# by scripts/snellius_fetch_weights.sh, before any GPU job runs.
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source scripts/snellius_env.sh

echo "== python: $(python --version 2>&1) ($(command -v python))"

if [ ! -d .venv ]; then
    echo "== creating .venv"
    python -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

echo "== installing (this pulls torch, several GB, first run only)"
python -m pip install -q -U pip
python -m pip install -q -e '.[dev]'

echo "== scratch: $VFP_OUTPUT_ROOT"
mkdir -p "$VFP_OUTPUT_ROOT"

echo "== CPU test suite"
pytest -q

cat <<MSG

Setup complete. For every later session:

    cd ${VFP_ROOT:-$HOME/vlm-flow-probe} && source scripts/snellius_env.sh

Weights:   scripts/snellius_fetch_weights.sh <hf-model-id>   (login node)
GPU check: srun -p gpu_mig -n1 --gpus 1 -t 10:00 --pty bash
MSG
