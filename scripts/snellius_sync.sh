#!/usr/bin/env bash
# Push this working tree to Snellius. Run LOCALLY, on the workstation:
#
#     scripts/snellius_sync.sh
#
# The repo is private, so the cluster cannot clone it directly. rsync also
# carries uncommitted work, which is usually what you want to test on a GPU.
# .git is included so provenance.json still records a real commit SHA.
set -euo pipefail
cd "$(dirname "$0")/.."

REMOTE=${REMOTE:-snellius}
DEST=${DEST:-vlm-flow-probe}

rsync -az --delete \
    --exclude '.venv/' --exclude 'output/' --exclude 'datasets/' \
    --exclude '__pycache__/' --exclude '*.pyc' \
    --exclude '.pytest_cache/' --exclude '.ruff_cache/' \
    ./ "$REMOTE:$DEST/"

echo "synced -> $REMOTE:$DEST  ($(git rev-parse --short HEAD)$(git diff --quiet || echo '+dirty'))"
