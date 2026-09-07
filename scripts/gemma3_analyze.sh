#!/usr/bin/env bash
# Post-run analysis for the Gemma 3 replication: redundancy analysis of the
# multi-layer matrix, distilled summaries of every large result file, and the
# paper figures. CPU only; run on a login node or locally on synced outputs.
#
#   scripts/gemma3_analyze.sh [multilayer experiment dir]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
ML=${1:-output/experiments/gemma3_4b_multilayer_clevr_lite_l11-17_attn_z_question}
SPAN=${SPAN:-"11 12 13 14 15 16 17"}
CONC=${CONC:-17}

if [ -d "$ML/conditions" ]; then
  $PY -m vlmflowprobe.cli.analyze_multilayer --experiment_dir "$ML"
fi
$PY -m vlmflowprobe.cli.distill_results --root output/experiments
# shellcheck disable=SC2086
$PY scripts/gemma3_figures.py --root output/experiments --span $SPAN --concentrated "$CONC" \
    --multilayer_dir "$ML" --figdir output/paper_figures/gemma3
echo "analysis done"
