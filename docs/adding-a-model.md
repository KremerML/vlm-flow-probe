# Adding a model

Everything model-specific lives in one `ModelAdapter` subclass. Nothing outside
`src/vlmflowprobe/adapters/` may know which model is loaded, and the checks in
`adapters/contract.py` are what hold that line.

The LLaVA equivalence gate (`gate/README.md`) cannot apply to a new model: it
compares against archived numbers a new model does not have. `vfp-verify-adapter`
takes its place.

## The steps

1. **Write the adapter.** `src/vlmflowprobe/adapters/<model>.py`, decorated with
   `@register("<key>")`, implementing the abstract methods below.
2. **Register it lazily.** Add `"<key>": "vlmflowprobe.adapters.<model>:<Class>"`
   to `_LAZY` in `adapters/registry.py`, so CPU-only work (tests, distillation,
   analysis) never imports a torch-heavy module it does not need.
3. **Add a model fragment.** `configs/fragments/model_<model>.yaml`, carrying
   `model.adapter`, `model.name`, `model.dtype` and a short `model.tag`.
4. **Add experiment configs** under `configs/experiments/<tag>/`, named
   `sae_layer{N}_{site}_{position}.yaml`, with `experiment.name` prefixed by the
   tag. This is enforced: `validate_model_identity` refuses a name that does not
   carry the tag, because nothing in an experiment directory records which model
   wrote it and two models would interleave artifacts in silence.
5. **Add a probe** to `tests/probes.py` and one entry to `ADAPTER_PROBES` marked
   `pytest.mark.gpu`. The whole contract suite then runs against the new adapter
   under `pytest -m gpu`.
6. **Run `vfp-verify-adapter --config configs/experiments/<tag>/<cfg>.yaml`**
   until every row passes.
7. **Set the layer set** in the config (`multilayer.layers`, `conditions.*`).
   The analysis CLIs read those, so nothing needs editing for a model of a
   different depth. Set `conditions.a0_regression_layer: null` — the A0 condition
   regresses against a published LLaVA number that means nothing here.

## What the adapter must implement

| method | contract |
|---|---|
| `load` | model + processor, on device, `eval()`, in whatever attention implementation knockout needs (**eager**, see below) |
| `model` / `tokenizer` / `d_model` / `n_layers` | the basics; `d_model` and `n_layers` are read from the loaded model, never hardcoded |
| `build_inputs(question, image)` | owns prompt templating and image preprocessing; returns a `ModelBatch` whose `prompt` is the exact string tokenized |
| `forward(batch, extra_input_ids)` | teacher-forced, `use_cache=False`, logits `[1, S(+A), V]`; the extra ids are appended — this is the sequence-scoring path |
| `generate(batch, max_new_tokens)` | greedy, normalized to `GenResult`: **new tokens only**, not HF's prompt-inclusive sequence |
| `n_image_tokens` / `image_token_span` / `question_token_span` | token geometry, in post-expansion coordinates |
| `_decoder_layers` | the decoder layers, indexable by layer number |
| `install_attention_knockout` / `remove_attention_knockout` | block `(target, source)` pairs per layer; the base class's `attention_knockout` context manager guarantees removal |

`last_token_index` and `answer_start_index` have correct defaults; override them
only if your sequence layout genuinely differs.

### Raise, never guess

`AdapterContractError` exists because the archive's geometry code returned `0`
image tokens when it could not work out the answer, corrupting every downstream
position calculation without a traceback. Every geometry method must raise when
it cannot answer. `layer_module` likewise raises on an unknown activation site
and an out-of-range layer instead of falling back to the residual stream.

## The post-expansion coordinate rule

Every index an adapter exposes — `image_token_span`, `question_token_span`,
`last_token_index`, block-config pairs, hidden-state and logit indices — is a
**post-expansion language-model sequence index**. One coordinate system, no
translation anywhere else in the package.

Adapters whose processor materializes image placeholder tokens directly in
`input_ids` (HF LLaVA, Qwen-VL, Gemma 3) satisfy this for free. A model that
expands images *inside* the forward pass must translate at its own boundary —
before returning anything from `build_inputs` or the geometry methods.

## The eager-attention requirement

`knockout/mask_hooks.py` blocks attention by rewriting the materialized 4D
additive mask in a decoder-layer `forward_pre_hook(with_kwargs=True)`. That
requires:

* `attn_implementation="eager"` at load time. Under SDPA or FlashAttention the
  mask may be `None` and the hook raises rather than silently not blocking.
* transformers `>=4.56,<4.58` — the pin in `pyproject.toml`. The mechanism is
  verified against the 4.57.x layer-call contract; 4.58 removes deprecated
  kwargs and may reshape the call.

Blocked entries are **set** to the dtype minimum, never added to it: adding to
an already-minimal entry overflows fp16.

If a model's layers do not receive a single materialized 4D mask, the adapter
must implement `install_attention_knockout` itself rather than delegating to
`install_mask_knockout`, and `vfp-verify-adapter`'s `knockout_moves_the_margin`
row is what tells you whether your implementation actually reaches the attention
computation.

## Pre-trained dictionaries instead of stage 01

A model that ships with public SAEs skips `vfp-train-sae`. `vfp-import-sae`
downloads the dictionary named by `sae.pretrained` for the config's
`model.target_layer`, converts it to `sae_checkpoint.pt`, and -- given
`--activations_path` from `vfp-collect` -- writes `reconstruction_eval.json`
measured on this task's activations. Every later stage runs unchanged: the
loaders dispatch on the state dict (`threshold` present means JumpReLU), and
the ablator only needs `encode`/`decode`/`n_features`.

Two things to check before trusting an imported dictionary. Its hook point:
Gemma Scope 2's "attn_out" SAEs read the *input* of `o_proj` (2048-wide for
4B), not the attention module's output, which is why the Gemma adapter exposes
an `attn_z` site (an identity tap in front of `o_proj`) and `site_dim` reports
its width so the loaders can refuse a mismatch. And its fit to the task:
public dictionaries are trained on text, so the explained variance, mean $L_0$
and dead fraction on the model's question-position activations are results,
not assumptions.

## The Gemma 3 adapter (`hf-gemma3`)

Three conventions differ from LLaVA and are set in the adapter rather than
downstream. The question span is every text position after the image block,
which is what the LLaVA runs used in effect (`tests/test_hf_llava_adapter.py`
records that the sublist match never hit) and what makes `Image->Question`
knockout a complete cut of cross-modal transfer at a layer. The answer prefix
is empty: the generation prompt ends in `model\n`, so an answer continues
without a leading space (`ModelAdapter.answer_prefix`, read by
`sequence_logprob`). And `token_type_ids` are passed on every forward so the
image block keeps its bidirectional attention.

**Alternating local/global attention** was the risk flagged before the
adapter existed. Under transformers 4.57 with eager attention, every decoder
layer -- sliding-window and global alike -- receives a materialized 4D mask
(`causal_mask_mapping[layer.attention_type]`), so `mask_hooks` applies
unchanged. CLEVR-Lite prompts are ~300 tokens, well inside the 1024-token
window, so the sliding mask is the causal mask. `vfp-verify-adapter --layer 0`
(sliding) and `--layer 5` (global) both show the edit reaching the attention
computation; `tests/test_hf_gemma3_adapter.py` pins the same pair.

## Known risks

**Qwen3-VL — DeepStack.** Qwen3-VL injects visual features at several layers
rather than only at the input. `image_token_span` assumes one contiguous image
block in `input_ids`, and it raises on non-contiguous placeholders; with
DeepStack the image's causal footprint is not a single span at all, so both the
span abstraction and the `Image->Question` flow grammar in
`knockout/block_config.py` need rethinking before the results mean anything.
Do not work around the contiguity check — it is reporting a real modelling
problem.

## Verifying

```bash
pytest -m gpu -q                                  # the contract suite, on the real model
vfp-verify-adapter --config configs/experiments/<tag>/<cfg>.yaml
```

`vfp-verify-adapter` runs the same checks plus the end-to-end path —
`build_inputs` → `forward` → `generate` → `sequence_logprob` on both options →
knockout at one layer moving the margin → no hooks left behind — prints a
pass/fail table and exits nonzero on failure. `--layer` picks the knockout layer
(default 0, where `Image->Question` knockout has its largest effect on LLaVA);
`--json` writes the table with provenance.

Adding a model must not change the LLaVA numbers. `pytest -q`, `pytest -m gpu -q`
and `scripts/run_gate.sh` all have to stay green.
