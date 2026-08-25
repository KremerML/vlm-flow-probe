#!/usr/bin/env bash
# Train + causal feature ID + ablation for a set of layers, resumably:
# each step is skipped when its output artifact already exists.
# Usage: scripts/run_full_pipeline.sh [layers...]   (default: 0 10 11 12 13 14)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
LAYERS=("${@:-0 10 11 12 13 14}")
[ $# -eq 0 ] && LAYERS=(0 10 11 12 13 14)
ACTS=${ACTS:-output/activations/clevr_lite_question}

export PYTHONUNBUFFERED=1

if [ ! -d "$ACTS" ]; then
  echo "== collecting activations for layers ${LAYERS[*]} =="
  $PY -m vlmflowprobe.cli.collect_activations \
    --config configs/experiments/sae_layer11_attn_out_question.yaml \
    --layers "$(IFS=,; echo "${LAYERS[*]}")" --output_dir "$ACTS"
fi

for L in "${LAYERS[@]}"; do
  CFG="configs/experiments/sae_layer${L}_attn_out_question.yaml"
  DIR="output/experiments/sae_clevr_lite_layer${L}_attn_out_question"
  if [ ! -f "$DIR/sae_checkpoint.pt" ]; then
    echo "== layer $L: training SAE =="
    $PY -m vlmflowprobe.cli.train_sae --config "$CFG" --activations_path "$ACTS"
  fi
  if [ ! -f "${DIR}_causal/causal_feature_catalog.json" ]; then
    echo "== layer $L: causal feature identification =="
    $PY -m vlmflowprobe.cli.identify_features --config "$CFG" --target margin --top_k 200
  fi
  if [ ! -f "${DIR}_causal/results/ablation_v2_results.json" ]; then
    echo "== layer $L: ablation =="
    $PY -m vlmflowprobe.cli.run_ablation --config "$CFG" --skip_passthrough --max_samples 256
  fi
done
echo "done."
