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
Shipped adapters: HF-native LLaVA-1.5 (`llava-hf/llava-1.5-7b-hf`) and Gemma 3 4B
(`google/gemma-3-4b-it`, run on the pre-trained Gemma Scope 2 attention dictionaries).

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

## Compute

Development is local; GPU runs go to Snellius, SURF's
[national supercomputer](https://www.surf.nl/en/services/snellius-the-national-supercomputer).
Useful partitions are `gpu_h100` and `gpu_a100` (5-day walltime cap)
with `gpu_mig` for debug-sized jobs. Home directories are quota'd at 200 GiB, so activation caches
and SAE checkpoints belong on scratch (8 TiB), not under `$HOME` or in the repo tree.

Access is per-user and not configured by this repo: SURF whitelists your workstation's public IP and
you register an ed25519 public key in the SURF portal. Once `ssh snellius` works:

```bash
scripts/snellius_sync.sh                                     # local: rsync tree -> cluster
ssh snellius 'cd vlm-flow-probe && scripts/snellius_setup.sh'    # login node: venv + deps + tests
```

`docs/snellius.md` covers the rest — module stack, the home-vs-scratch split, and why model weights
must be pre-fetched on a login node.

## Pipeline

Each stage is a console script driven by the same YAML config
(`configs/experiments/<model>/*.yaml`, composed from `configs/fragments/` via `include:`;
`configs/frozen/` holds fully-resolved configs that reproduce published runs).

```bash
vfp-knockout  --config configs/experiments/llava15/knockout_clevr_lite.yaml  # 1. layer sweep
vfp-collect   --config <cfg> --layers 0,10,11,12,13,14 --output_dir <d> # 2. cache activations
vfp-train-sae --config <cfg> --activations_path <d>                     # 3. train SAE per layer
vfp-import-sae --config <cfg> --activations_path <d>                    # 3'. or import a pre-trained one (Gemma Scope 2)
vfp-identify  --config <cfg> --target margin --top_k 200                # 4. causal feature ID
vfp-ablate    --config <cfg> --max_samples 256                          # 5. ablation + controls
vfp-analyze   --config <cfg> --results <path>                           # 6. stats report
vfp-multilayer --config <cfg> --phases all                              # multi-layer condition matrix
vfp-analyze-multilayer --experiment_dir <d>                             # redundancy index R = A/K
vfp-distill   --root output                                             # distill raw results to summaries
vfp-gate      --all --archive-root <archive>                            # equivalence gate
vfp-verify-adapter --config <cfg>                                      # adapter contract, real model
```

Every stage writes `provenance.json` (git SHA, package versions, GPU, resolved config, seed, argv)
into its experiment directory.

The Gemma 3 replication (`configs/experiments/gemma3_4b/`) runs the same stages with
`vfp-import-sae` in place of training; `scripts/run_gemma3_layers.sh` chains import, identification
and single-layer ablation per layer, `scripts/make_multilayer_config.py` writes the multi-layer
config for a span chosen from the knockout sweep, and `scripts/gemma3_figures.py` draws the figures.

## Layout

- `src/vlmflowprobe/adapters/` — `ModelAdapter` ABC, the contract checks, registry, stub (CPU
  tests), `hf_llava`. Adding a model: `docs/adding-a-model.md`
- `src/vlmflowprobe/core/` — SAE module, config system
- `src/vlmflowprobe/knockout/` — flow grammar, teacher-forced scoring, sweep runner
- `src/vlmflowprobe/ablation/` — feature ablator, three-condition experiment, multi-layer matrix, statistics
- `src/vlmflowprobe/features/` — gradient×activation causal feature identification
- `src/vlmflowprobe/positions.py` — the single position resolver (policy-pinned)
- `src/vlmflowprobe/contracts.py` — output-artifact contract and condition-id grammar
- `src/vlmflowprobe/utils/run_context.py` — where an analysis gets its layers and model depth
- `gate/` — vendored reference artifacts + gate criteria

## Provenance

Published results in the paper were produced in the archive repo; `gate/references/SOURCES.md`
records the archive commit and artifact paths each gate reference was vendored from.
