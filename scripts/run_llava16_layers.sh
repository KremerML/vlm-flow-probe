#!/usr/bin/env bash
# LLaVA-1.6: train the SAE, identify causal features, and run the single-layer
# ablation for a set of layers, resumably (each step is skipped when its artifact
# exists). Unlike Gemma, this model trains its own dictionaries, so step one is
# vfp-train-sae rather than vfp-import-sae. Needs the activations from vfp-collect:
#
#   vfp-collect --config configs/experiments/llava16/sae_layer0_attn_out_question.yaml \
#       --layers $(seq -s, 0 31) --output_dir output/activations/llava16_clevr_lite_question
#   scripts/run_llava16_layers.sh 10 11 12
#
# One layer per Slurm job on the cluster:
#   scripts/snellius_submit.sh sae-l11 "scripts/run_llava16_layers.sh 11" 08:00:00 gpu_h100
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
ACTS=${ACTS:-output/activations/llava16_clevr_lite_question}
TOP_K=${TOP_K:-200}
ABLATE_N=${ABLATE_N:-256}
[ $# -gt 0 ] || { echo "usage: $0 <layer> [layer...]"; exit 2; }
export PYTHONUNBUFFERED=1

for L in "$@"; do
  CFG="configs/experiments/llava16/sae_layer${L}_attn_out_question.yaml"
  DIR="output/experiments/llava16_sae_clevr_lite_layer${L}_attn_out_question"
  if [ ! -f "$DIR/sae_checkpoint.pt" ]; then
    echo "== layer $L: training SAE =="
    $PY -m vlmflowprobe.cli.train_sae --config "$CFG" --activations_path "$ACTS" --show_progress 0
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
