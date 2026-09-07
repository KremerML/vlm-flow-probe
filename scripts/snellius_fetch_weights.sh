#!/usr/bin/env bash
# Pre-fetch model weights on a LOGIN node, before submitting a GPU job.
#
#     scripts/snellius_fetch_weights.sh llava-hf/llava-1.5-7b-hf
#
# Compute nodes are not assumed to reach huggingface.co, so a job that expects
# to download its own weights can die minutes in. Everything lands in HF_HOME
# (see scripts/snellius_env.sh), which is on the non-purged home filesystem.
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source scripts/snellius_env.sh

MODEL=${1:?usage: scripts/snellius_fetch_weights.sh <hf-model-id>}
echo "== fetching $MODEL into $HF_HOME"
python - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))
PY
