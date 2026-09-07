#!/usr/bin/env bash
# Snellius (SURF) environment. Source it, do not execute it:
#
#     source scripts/snellius_env.sh
#
# Deliberately small: the module stack, the venv, and the two paths that must
# not be allowed to default into $HOME. scripts/snellius_setup.sh builds what
# this file assumes already exists.

VFP_ROOT=${VFP_ROOT:-$HOME/vlm-flow-probe}
VFP_SCRATCH=${VFP_SCRATCH:-/scratch-shared/$USER/vlm-flow-probe}

# 2024 stack: Python 3.12 (the repo needs >=3.10; the login node's system
# python3 is 3.9 and cannot install this package).
if command -v module >/dev/null 2>&1; then
    module load 2024
    module load Python/3.12.3-GCCcore-13.3.0
fi

# Weights stay in $HOME. /scratch-shared is purged on a 14-day rolling window,
# and re-downloading every model is a worse trade than the home quota.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

# Bulk regenerable artifacts go to scratch: the 200 GiB home quota will not
# hold activation caches or SAE checkpoints.
export VFP_OUTPUT_ROOT=${VFP_OUTPUT_ROOT:-$VFP_SCRATCH/output}

export PY=${PY:-$VFP_ROOT/.venv/bin/python}
if [ -f "$VFP_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "$VFP_ROOT/.venv/bin/activate"
fi
