# Scaffold session brief

Paste as the initial prompt for a session started in `~/Documents/Github/vlm-flow-probe`.

---

**Repo.** `vlm-flow-probe` — mechanistic-interpretability toolkit pairing attention knockout with
SAE feature ablation in vision-language models, behind a `ModelAdapter` interface. Ported from the
archive repo `cross-modal-information-flow-in-MLLM`; the equivalence gate (`vfp-gate`) reproduced
the archived LLaVA-1.5 results 18/18. Install with `pip install -e .[dev]`, venv at `.venv`, GPU is
a 24 GB RTX 4090. 143 CPU tests + 5 GPU tests currently green.

**State.** Exactly one real adapter (`hf-llava`) plus a `StubAdapter`. The core carries no
LLaVA-specific code, but the per-model scaffolding still assumes a single 32-layer model.

**Goal of this session.** Make adding a model a systematic operation. The Gemma 3 adapter comes
next session and should be written against this scaffolding, not alongside it.

## Work items

1. **Reusable adapter contract suite.** `tests/test_adapter_contract.py` is stub-only and not
   parameterized. Turn it into a suite any adapter can be run against (parametrize over adapter
   factories), covering: `layer_module` raises on an unknown site and an out-of-range layer;
   post-expansion coordinate invariants (`image_token_span` inside `[0, seq_len)`, question span in
   bounds, `answer_start_index == seq_len`); `forward` with `extra_input_ids` extends the sequence;
   `generate` returns new tokens only; `n_image_tokens` raises rather than returning 0; knockout
   install/remove pairing with no leaked hooks. `StubAdapter` and `hf-llava` must both pass, the
   latter under `-m gpu`.

2. **`vfp-verify-adapter` CLI.** One command that loads an adapter from a config and runs the
   contract suite against the real model plus a minimal end-to-end path: `build_inputs` → `forward`
   → `generate` → `sequence_logprob` → knockout at one layer moves the margin → hooks cleaned up.
   Prints a pass/fail table, exits nonzero on failure. This is what a new model gets in place of the
   LLaVA-specific equivalence gate, which cannot apply to a model with no archived results.

3. **Config-driven layer sets.** Remove the hardcoded layers: `cli/analyze_controls.py:27`
   (`LAYERS = (10, 11, 12, 13, 14)`) and the `for layer in (11, 14)` loops at lines 189 and 220;
   the literal span tuples in `cli/analyze_multilayer.py:182`. Layers come from config, and add
   depth-fraction reporting (`layer / n_layers`) so models of different depths are comparable.

4. **Model identity in names.** Config fragments, experiment configs and experiment output
   directories must carry the model, so two models cannot collide — e.g.
   `configs/experiments/{model}/sae_layer{N}_{site}_{position}.yaml`, and `experiment.name`
   including the adapter key.

5. **`docs/adding-a-model.md`.** The abstract methods, the post-expansion coordinate rule, and the
   eager-attention requirement of `knockout/mask_hooks` (transformers pinned `<4.58`). Record the
   two known risks explicitly: Gemma 3's alternating local/global attention may not present the
   single materialized 4D mask per layer the hook requires, and Qwen3-VL's DeepStack injects visual
   features at several layers, breaking the contiguous single-image-span assumption in
   `image_token_span`.

6. **`CLAUDE.md` for this repo** (there is none): architecture, the adapter contract, commands,
   conventions, and the artifact policy — raw result JSONs gitignored, `*.summary.json` committed.

## Constraints

- Do not change any published numbers. The LLaVA equivalence gate must still pass afterwards.
- Keep `cli/distill_results.py` importable with the standard library alone.
- Tests stay CPU-only by default; anything needing a GPU or weights goes behind `-m gpu`.

## Done when

`pytest -q` and `pytest -m gpu -q` are green; `vfp-verify-adapter` passes end to end on the LLaVA
config; no layer numbers are hardcoded in the analysis CLIs; `docs/adding-a-model.md` and
`CLAUDE.md` exist.
