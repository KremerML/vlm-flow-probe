#!/usr/bin/env bash
# Full equivalence gate against the archive repo (~4-7 GPU-hours worst case,
# ~1h in practice). See gate/README.md for the criteria.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export PYTHONUNBUFFERED=1
ARCHIVE=${ARCHIVE:-$HOME/Documents/Github/cross-modal-information-flow-in-MLLM}
$PY -m vlmflowprobe.cli.verify_equivalence --all --archive-root "$ARCHIVE" "$@"
