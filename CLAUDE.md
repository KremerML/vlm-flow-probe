# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The venv is `.venv` (`PY=.venv/bin/python` is what the scripts default to).

```bash
pip install -e .[dev]
pytest -q                                  # 194 CPU tests, no weights, ~5s
pytest -q tests/test_positions.py::test_name  # one test
pytest -m gpu -q                           # 63 tests: needs CUDA + weights + the archive
pytest -m gpu -q -k "hf-llava and not next"  # one model at a time: three 7B-class models
                                           # do not fit in 24 GB together
ruff check src tests                       # line-length 110, py310
```

`gpu` and `gate` are deselected by default via `addopts` in `pyproject.toml` (the `gate` marker is
declared but currently unused — the gate runs through `vfp-gate`, not pytest). Anything needing a
GPU or model download goes behind `-m gpu`; the default suite must stay CPU-only and network-free.

`pyproject.toml` sets only `line-length` and `target-version`, so ruff's rule set moves with the
installed ruff. 0.16.4 reports ~630 findings across the tree, nearly all of them style rules
(`UP006`, `E741`) that also fire on untouched committed files — a version drift, not a regression.
Judge a change by whether it adds findings of a *new* kind, not by the total.

Pipeline stages are console scripts (`vfp-*`, see `[project.scripts]`), all driven by one YAML
config. `README.md` lists the full sequence. Two wrappers plus the adapter check:

```bash
scripts/run_full_pipeline.sh [layers...]   # collect -> train -> identify -> ablate, resumable
scripts/run_gate.sh                        # full equivalence gate (ARCHIVE=<path to archive repo>)
vfp-verify-adapter --config <cfg>          # the adapter contract against a real model
```

Every CLI takes `--override section.key=value` (repeatable, YAML-parsed) on top of `--config`.

### Remote execution (Snellius)

GPU runs go to Snellius, SURF's national cluster — `docs/snellius.md` is the standing guide.
Access is per-user (portal-registered ed25519 key plus a whitelisted workstation IP) and none of it
lives in the repo. Partitions: `gpu_h100`, `gpu_a100`, `gpu_mig` for debug-sized jobs, 5-day
walltime cap; `staging` for bulk transfer.

```bash
scripts/snellius_sync.sh                                     # local: rsync tree -> cluster
ssh snellius 'cd vlm-flow-probe && scripts/snellius_setup.sh'    # login node: venv + deps + tests
source scripts/snellius_env.sh                               # every later session, on the cluster
```

`snellius_sync.sh` exists because the repo is private and the cluster cannot clone it; it carries
uncommitted work and includes `.git`, so `provenance.json` still records a real SHA.
`snellius_env.sh` loads the 2024 module stack and Python 3.12 — the login node's system `python3`
is 3.9 and cannot install this package. `snellius_fetch_weights.sh` pre-pulls weights on a login
node: compute nodes are not assumed to reach huggingface.co, so a job that downloads its own
weights can die minutes in.

Storage split, the one non-obvious decision: checkout, venv and `HF_HOME` live in `$HOME` (200 GiB,
never purged); activation caches go to `$VFP_OUTPUT_ROOT` on `/scratch-shared` (8 TiB, purged on a
rolling 14-day window). Only regenerable things go on scratch. SAE checkpoints and
`output/experiments` stay on `$HOME` — small, and losing a run's results costs more than a purge
saves.

`scripts/snellius_submit.sh <name> "<vfp command>" [time] [partition] [sbatch args]` wraps one
command in `scripts/slurm/vfp.sbatch` (one GPU, logs in `~/logs`) and defaults to `gpu_a100`.
`gpu_h100` was entirely down on 2026-09-06, which is why the Gemma runs went to `gpu_a100`; it is
up again as of 2026-09-08. On the cluster `output/` is a real directory under `$HOME` and
`output/activations` is a symlink onto `$VFP_OUTPUT_ROOT` (scratch) — the Gemma cache moved there
2026-09-08, and LLaVA-1.6's 32 layers are ~1.3 TB.
`scripts/run_gemma3_layers.sh` is the per-layer import -> identify -> ablate chain for Gemma.

## Architecture

Two causal interventions measured at the same locus on the same forced-choice margin
`log P(true) - log P(false)`: **attention knockout** (`knockout/`) gives the model-centric ceiling,
**SAE feature ablation** (`ablation/`) tests whether that flow is mediated by sparse features. The
multi-layer matrix (`ablation/multilayer_experiments.py`) runs both plus their combination as
`Condition` objects over a shared sample set, and `vfp-analyze-multilayer` reports the redundancy
index R = A/K.

### The adapter boundary

`adapters/base.py` is the whole model-specific surface: prompt building, image preprocessing, token
geometry, forward/generate, module resolution, knockout install/remove. Nothing outside
`adapters/` may know about LLaVA. Adapters are registered lazily by config key `model.adapter`
(`adapters/registry.py`) so CPU-only work never imports torch-heavy modules.

`adapters/contract.py` holds that boundary as runnable checks, defined once and consumed twice:
`tests/test_adapter_contract.py` parametrizes them over the probes in `tests/probes.py` (stub on
CPU, real adapters marked `gpu`), and `vfp-verify-adapter` runs the same set plus an end-to-end
group against a real model. A new model has no archived numbers, so the LLaVA gate cannot apply to
it — that command is what takes its place. Adding a model: `docs/adding-a-model.md`.

**Coordinate convention:** every index an adapter exposes — `image_token_span`,
`question_token_span`, `last_token_index`, block-config pairs, hidden-state and logit indices — is a
*post-expansion language-model sequence index*. An adapter for a model that expands images
internally must translate at its own boundary.

`StubAdapter` (`adapters/stub.py`, plus `tests/stubs.py`) is how every model-facing test drives the
interface on CPU.

Three real adapters: `hf-llava` (LLaVA-1.5-7B), `hf-llava-next` (LLaVA-1.6 vicuna-7B) and
`hf-gemma3` (Gemma 3 4B IT). `hf-llava-next` subclasses `hf-llava` and changes only the vision
front-end: AnyRes tiling (1176 image tokens for a 224x224 image, against 576), `image_sizes` on
`ModelBatch.extra`, no `expand2square`, and a per-sample image-feature cache. Gemma runs on
pre-trained Gemma Scope 2 dictionaries via `vfp-import-sae` instead of `vfp-train-sae`; its SAEs
read the `o_proj` *input*, exposed as the `attn_z` site (identity tap, width 2048, `site_dim`).
Gemma's question span is every post-image text position and its `answer_prefix` is empty -- see
`docs/adding-a-model.md`. `core/sparse_autoencoder.py` holds both the L1/ReLU and the JumpReLU
class; `build_sae_from_state` picks one from the state dict.

### Load-bearing single sources of truth

- `positions.py` — the one position resolver. The archive had three drifted copies with three real
  semantic divergences that produced the published numbers; those are preserved as two explicit
  `PositionPolicy` values (`COLLECTION_POLICY` for SAE training data + causal scores,
  `ABLATION_POLICY` for the published ablation numbers). Do not "fix" the asymmetry.
- `contracts.py` — the output-artifact list and the condition-id grammar. The multilayer analyzer,
  the distiller, and the archive's figure notebooks all key on these names; changing one is a
  breaking schema change.
- `ablation/sample_cache.py` — per-sample baselines and resolved positions, computed once and reused
  across all conditions. Its list order is the ordering authority (`dataset.create_dataloader()`
  order); records carry `question_id` so misordering is detectable, and `strict_cache=True` makes a
  miss an error.
- `knockout/block_config.py` — the flow grammar (`"Image->Question"`, `"Image->Last"`) and the
  `(target, source)` pair builders.
- `utils/run_context.py` — where an analysis gets its layer set, its spans and the model depth.
  The analysis CLIs carry no layer numbers: `RunContext` reads them from the run's
  `provenance.json` (or `--config`), and `span_pairs()` derives the ablation/knockout condition-id
  pairing that `analyze_multilayer` used to spell out as a literal table.
  `tests/test_run_context.py` pins that derivation against the deleted literal.

### Config system

`core/config.py`: `DEFAULT_CONFIG` <- `include:` fragments <- the file's own keys <- `--override`.
Fragments (`configs/fragments/`) may not nest includes. `configs/experiments/` composes them;
`configs/frozen/` holds fully-resolved reproduction records (`frozen: true`, includes forbidden).

Frozen configs carry archived values verbatim, including known-bad ones — e.g.
`random_control.matched_metric: correct_mean` with `strict_matching: false`, the permissive matching
that silently produced uniform controls. Never copy frozen values into a new experiment. The
`frozen` flag is retained in the resolved config (it exempts reproduction records from the
model-identity rule below).

**Model identity.** Experiment configs live under `configs/experiments/<tag>/` and their
`experiment.name` must contain `model.tag` (`llava15` for the shipped fragment).
`validate_model_identity`, called from `setup_experiment`, refuses a name that does not — nothing in
an experiment directory records which model wrote it, so two models under one name would interleave
checkpoints and catalogs in silence. Frozen configs and `experiment.require_model_tag: false` are
the two exemptions.

### Knockout mechanism (version-sensitive)

`knockout/mask_hooks.py` rewrites the materialized 4D additive attention mask via a decoder-layer
`forward_pre_hook(with_kwargs=True)`. This requires `attn_implementation="eager"` and is verified
against the transformers 4.57.x layer-call contract — hence the `>=4.56,<4.58` pin. Blocked entries
are *set* to the dtype minimum, never added (adding overflows fp16). The hook raises if the expected
mask is absent rather than silently no-op'ing.

### Controls

Sampled matched controls are not identified for this experiment (selection is a top-k cut on the
causal score, which is near-monotone in activation). `ablation/rank_band_controls.py` replaces them
with controls disjoint by construction — rank band, activation-top, ranking-difference — and
`ablation/matching_diagnostics.py` is opt-in instrumentation
(`random_control.log_matching: true`) that makes a silent matched->uniform fallback visible. See
each module's docstring for the measurements behind this.

## Conventions

- **Fail loudly, never degrade silently.** `AdapterContractError` exists because the archive
  returned `0` image tokens on failure and corrupted every downstream position. Same for
  `layer_module`'s unknown-site fallback and the matched-sampling banner in
  `ablation/ablation_experiments.py`.
- **Artifact policy.** Large result JSONs are gitignored; their `*.summary.json` siblings from
  `vfp-distill` are what gets committed and cited. `cli/distill_results.py` must stay stdlib-only so
  it runs without the model venv. See `.gitignore` for the exact list.
- **Provenance.** Every stage writes `provenance.json` (git SHA, package versions, GPU, resolved
  config, seed, argv, and `model_geometry` when an adapter is loaded) into its experiment dir;
  `save_config` writes the fully-resolved config beside it. The analyses read both back, so pass
  `adapter=` to `write_provenance` in any new stage or depth fractions go missing downstream. CLI
  entry points import `cli/_bootstrap` before torch to set allocator and cuBLAS determinism env
  vars.
- **Do not change published numbers.** The equivalence gate (`gate/README.md`, 18/18 as of
  2026-08-25) must keep passing. Relaxing any gate criterion requires an entry in
  `docs/deviation_ledger.md`; stage-0 geometry checks are hard equality and never waivable.

`docs/next-session-scaffold.md` is the brief this scaffolding was built from; `docs/adding-a-model.md`
is the standing guide, including the two known risks for the models queued next (Gemma 3's
alternating local/global attention may not present the single 4D mask `mask_hooks` needs; Qwen3-VL's
DeepStack breaks the contiguous single-image-span assumption).
