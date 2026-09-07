#!/usr/bin/env bash
# Gemma 3: import the Gemma Scope 2 dictionary, identify causal features, and run the
# single-layer ablation for a set of layers, resumably (each step is skipped when its
# artifact exists). Needs the val-split activations from vfp-collect first:
#
#   vfp-collect --config configs/experiments/gemma3_4b/sae_layer0_attn_z_question.yaml \
#       --layers $(seq -s, 0 33) --output_dir output/activations/gemma3_4b_clevr_lite_val_attn_z \
#       --override dataset.split=val
#   scripts/run_gemma3_layers.sh 10 11 12
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
ACTS=${ACTS:-output/activations/gemma3_4b_clevr_lite_val_attn_z}
TOP_K=${TOP_K:-200}
ABLATE_N=${ABLATE_N:-256}
[ $# -gt 0 ] || { echo "usage: $0 <layer> [layer...]"; exit 2; }
export PYTHONUNBUFFERED=1

for L in "$@"; do
  CFG="configs/experiments/gemma3_4b/sae_layer${L}_attn_z_question.yaml"
  DIR="output/experiments/gemma3_4b_sae_clevr_lite_layer${L}_attn_z_question"
  if [ ! -f "$DIR/sae_checkpoint.pt" ]; then
    echo "== layer $L: importing Gemma Scope 2 dictionary =="
    $PY -m vlmflowprobe.cli.import_sae --config "$CFG" --activations_path "$ACTS"
  fi
  if [ ! -f "${DIR}_causal/causal_feature_catalog.json" ]; then
    echo "== layer $L: causal feature identification =="
    $PY -m vlmflowprobe.cli.identify_features --config "$CFG" --target margin --top_k "$TOP_K" --no_progress
  fi
  if [ ! -f "${DIR}_causal/results/ablation_v2_results.json" ]; then
    echo "== layer $L: single-layer ablation vs matched controls =="
    $PY -m vlmflowprobe.cli.run_ablation --config "$CFG" --max_samples "$ABLATE_N" --no_progress
  fi
done
echo "done: layers $*"
