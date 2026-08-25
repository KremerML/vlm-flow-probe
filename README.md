# vlm-flow-probe

Mechanistic-interpretability toolkit for studying **cross-modal information flow in
vision-language models**, pairing two causal interventions at a matched locus:

1. **Attention knockout** — block an attention flow (e.g. `Image->Question`) at chosen layers and
   measure the drop in the forced-choice margin `log P(true) − log P(false)`. This identifies
   *which layers* carry causal image→text information, independent of any learned dictionary.
2. **SAE feature ablation** — train a sparse autoencoder on activations at those layers, select
   causally important features by gradient×activation attribution, zero them, and measure the same
   margin drop. This tests whether the flow is mediated by *sparse, interpretable features* — and
   the knockout effect provides the model-centric ceiling to calibrate the ablation against.

The package is **model-agnostic behind a `ModelAdapter` interface** (`vlmflowprobe.adapters`).
Everything that touches a specific model — prompt construction, image preprocessing, token
geometry, attention-mask editing, module resolution — lives in one adapter class per model family.
Shipped adapter: HF-native LLaVA-1.5 (`llava-hf/llava-1.5-7b-hf`).

## Status

Ported from the research repo
[`cross-modal-information-flow-in-MLLM`](https://github.com/KremerML/cross-modal-information-flow-in-MLLM)
(now the frozen archive of record for published outputs and the paper). The port is
**behavior-preserving**, enforced by an equivalence gate (`vfp-gate`) that reproduces archived
numbers — token geometry exactly, baseline margins at r ≥ 0.999, knockout layer profile, and the
layer-11 SAE-ablation regression — see `gate/README.md`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest -q          # CPU-only test suite, no model download, seconds
```

## Pipeline

Each stage is a console script driven by the same YAML config (`configs/experiments/*.yaml`,
composed from `configs/fragments/` via `include:`; `configs/frozen/` holds fully-resolved configs
that reproduce published runs).

```bash
vfp-knockout  --config configs/experiments/knockout_clevr_lite.yaml     # 1. layer sweep
vfp-collect   --config <cfg> --layers 0,10,11,12,13,14 --output_dir <d> # 2. cache activations
vfp-train-sae --config <cfg> --activations_path <d>                     # 3. train SAE per layer
vfp-identify  --config <cfg> --target margin --top_k 200                # 4. causal feature ID
vfp-ablate    --config <cfg> --max_samples 256                          # 5. ablation + controls
vfp-analyze   --config <cfg> --results <path>                           # 6. stats report
vfp-multilayer --config <cfg> --phases all                              # multi-layer condition matrix
vfp-analyze-multilayer --experiment_dir <d>                             # redundancy index R = A/K
vfp-distill   --root output                                             # distill raw results to summaries
vfp-gate      --all --archive-root <archive>                            # equivalence gate
```

Every stage writes `provenance.json` (git SHA, package versions, GPU, resolved config, seed, argv)
into its experiment directory.

## Layout

- `src/vlmflowprobe/adapters/` — `ModelAdapter` ABC, registry, stub (CPU tests), `hf_llava`
- `src/vlmflowprobe/core/` — SAE module, config system
- `src/vlmflowprobe/knockout/` — flow grammar, teacher-forced scoring, sweep runner
- `src/vlmflowprobe/ablation/` — feature ablator, three-condition experiment, multi-layer matrix, statistics
- `src/vlmflowprobe/features/` — gradient×activation causal feature identification
- `src/vlmflowprobe/positions.py` — the single position resolver (policy-pinned)
- `src/vlmflowprobe/contracts.py` — output-artifact contract and condition-id grammar
- `gate/` — vendored reference artifacts + gate criteria

## Provenance

Published results in the paper were produced in the archive repo; `gate/references/SOURCES.md`
records the archive commit and artifact paths each gate reference was vendored from.
