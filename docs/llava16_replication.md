# LLaVA-1.6 replication: decisions, deviations, run log

The LLaVA-1.5 study (`cross-modal-information-flow-in-MLLM/overleaf/main_final.tex`) re-run on
LLaVA-1.6 / LLaVA-NeXT (`llava-hf/llava-v1.6-vicuna-7b-hf`) with dictionaries trained from
scratch on all 32 layers. This file is the reviewer's record: every place the run departs from
the LLaVA-1.5 protocol, why, and where the evidence lives. The paper addition itself is
`overleaf/llava16_addition.tex` in the archive repo.

Where the Gemma 3 replication (`docs/gemma3_replication.md`) changed everything at once --
model, dictionaries, hook site, answer form -- this one changes as little as possible. Same
Vicuna-7B decoder, same vicuna_v1 prompt, same CLEVR-Lite data, same SAE recipe. The question
it asks is whether the calibration survives a doubled visual context with the *same* language
model.

## What is held fixed

The decoder (32 layers, d 4096, fp16); the vicuna_v1 prompt and single-word-answer suffix,
byte-identical (`HFLlavaAdapter.build_prompt`, one definition for both models); CLEVR-Lite and
its validation split; the forced-choice margin `log P(true) - log P(false)` with per-token
normalisation and the `" " + lowercase` answer convention; `Image->Question` and `Image->Last`
knockout over every layer on correctly answered validation items; the SAE recipe (ReLU + L1,
32,768 features on `attn_out`); gradient x activation feature selection with the dictionary
spliced into the forward pass, `k = 200` per layer; `replace`-mode ablation at question
positions; the multi-layer matrix and `vfp-analyze-multilayer`.

## What differs, and why

| Item | LLaVA-1.5-7B | LLaVA-1.6 vicuna-7B | Reason / evidence |
|---|---|---|---|
| Image tokens | 576 CLIP patches, one 336x336 crop | **1176**: 576 base + 576 unpadded high-res + 24 `image_newline` slots, from a 3-patch AnyRes tiling of the 224x224 image (`pixel_values` `(1, 3, 3, 336, 336)`, pinpoint 336x672) | The processor materialises all of them in `input_ids`, so geometry stays post-expansion for free. `output/llava16_verify_layer0.json`, `output/llava16_bringup.json`. |
| Prompt length | 636 tokens, image at [35, 611) | 1235 tokens, image at [35, 1211) | Same prompt text; only the image block grew. |
| Image preprocessing | `expand2square` (a no-op on square CLEVR-Lite images) | none | AnyRes selects its tiling from the image's own size, so padding would change the token count. `adapter_options.pad_to_square` is refused at load rather than ignored. |
| Image encoding | per forward | once per sample, features scattered into `inputs_embeds` (the model's own path) | Identical logits (max abs diff 0.0, `test_cached_features_match_the_pixel_path`), 1.10x on a full forward -- the language model dominates at 1235 tokens, so the saving is smaller than Gemma's. |
| Question span | 25 positions after the image block | 24 positions | The same tail text tokenises to 25 tokens in both checkpoints; in 1.6 the leading `▁` is absorbed after the image block (its tokenizer sets `add_prefix_space=True`). One token, not an AnyRes effect. |
| Answer tokenisation | `" blue"` -> `[▁][▁blue]` (2 tokens) | `" blue"` -> `[▁blue]` (1 token) | Same *string* convention, different *token* convention: 1.6's tokenizer sets `add_prefix_space=True`, so the prefix space is already implied. See "Margins are not comparable in absolute nats" below. |
| Answer form | model generates capitalised; scored lowercase | same | Bring-up on 32 items: generation matched the true answer on 32/32, capitalised on 32/32, and the pipeline lowercases decoded predictions, so nothing needs to change. Scoring stays lowercase for both models. |
| Dictionaries | trained here on layers 0, 10-14 | trained here on **all 32 layers**, same recipe | User decision; every layer gets a dictionary so the single-layer A-vs-K landscape covers the whole stack and any span the sweep picks can be run without a new training job. |
| Forward cost | 78 ms (RTX 4090) | 176 ms (RTX 4090) | 2.3x, from the doubled sequence under eager attention. |
| Compute | RTX 4090 | Snellius `gpu_h100` | Local bring-up only; all sweeps and training on the cluster. |

### Margins are not comparable in absolute nats across the two models

`sequence_logprob` normalises by answer-token count. Under the published convention the answer
string is `" " + option`, which is 2 tokens on LLaVA-1.5's tokenizer (`[▁][▁blue]`) and 1 token
on LLaVA-1.6's (`[▁blue]`). The leading `▁` is identical for the true and the false option, so
it cancels in the numerator, but dividing by 2 instead of 1 scales every LLaVA-1.5 margin down
by roughly a factor of two. Bring-up means on the same 32 items: LLaVA-1.5 3.157 (5.651 without
the prefix), LLaVA-1.6 7.297 (7.297 without the prefix -- a no-op there).

This affects nothing within a model: every condition of a run shares the convention, and
`R = A/K` is a ratio of margin drops, so it cancels exactly. It does mean absolute drops in nats
must not be compared across the two models without this factor stated. The protocol is not
changed to paper over it -- doing so would break the equivalence gate and the published numbers.

## Run log

| Date | Step | Where |
|---|---|---|
| 2026-09-08 | Gemma 3 work committed (`2a3ddd7`..`31c1c02`), including the first commit of `output/` summaries | this repo |
| 2026-09-08 | Adapter, fragment, 33 configs, tests; CPU suite 195 green | `34e154a` |
| 2026-09-08 | Weights fetched to the external SSD (14 GB); `protobuf` + `sentencepiece` added to the deps -- without protobuf, transformers raises `ImportError` *from the except clause meant to handle a load error*, masking the real one | `/run/media/ron/External SSD/vfp_weights/hf` |
| 2026-09-08 | `vfp-verify-adapter` 15/15 at layers 0 and 11; 1176 image tokens confirmed | `output/llava16_verify_layer{0,11}.json` |
| 2026-09-08 | GPU tests: 21 passed (`-k llava_next`); LLaVA-1.5's 15 contract checks + 4 module tests still pass, so the subclassing changed nothing | -- |
| 2026-09-08 | Bring-up on 32 val items, both models through the same script | `output/llava16_bringup.json`, `output/llava15_bringup.json` |
| 2026-09-08 | Equivalence gate **18/18** on the RTX 4090 after all Gemma + LLaVA-1.6 changes (A0 mean drop 0.21358 vs archived 0.21307, per-sample r 0.99959) | `output/gate/20260908_010458/gate_report.json` |
| 2026-09-08 | Cluster: tree synced, weights fetched, `output/activations` moved to scratch and symlinked (home 30% -> 24% of 200 GiB) | `/scratch-shared/rkremer/vlm-flow-probe/output/activations` |
| 2026-09-08 | CLEVR-Lite regenerated on the cluster (50,000 train / 2,000 val scenes; 186,638 / 7,790 questions). `val_questions.json` is byte-identical to the archive's, and all 2,000 val images match it pixel-for-pixel (only the PNG container bytes differ). The evaluation set is therefore the same items the LLaVA-1.5 and Gemma runs used. | `~/vlm-flow-probe/datasets/clevr_lite` |
| 2026-09-08 | CPU suite green on the cluster (195); bring-up job submitted to `gpu_h100` | job 26457746 |

### Bring-up numbers (32 validation items, RTX 4090)

| | LLaVA-1.5 | LLaVA-1.6 |
|---|---|---|
| Image tokens / sequence length | 576 / 636 | 1176 / 1235 |
| Forced-choice accuracy | 0.969 | 0.969 |
| Mean margin (published convention) | 3.157 | 7.297 |
| Mean `log P(true)` | -8.293 | -9.498 |
| Generation matches the true answer | 31/32 | 32/32 |
| Forward / greedy 4-token generate | 78 / 99 ms | 176 / 198 ms |

Single-sample knockout landscape (sample 0, `Image->Question`, window 1) peaks at layer 11 for
both models: 0.36 for 1.5, 2.05 for 1.6, with 1.6 also showing 0.90 at layer 10 and 1.02 at
layer 21. One sample -- the span decision comes from the full sweep, not from this.

## Incidents

**The cluster's dataset directory was deleted and regenerated (2026-09-08).** A local `datasets`
symlink into the archive checkout was rsynced to the cluster: `snellius_sync.sh` excluded
`datasets/` with a trailing slash, which matches directories only, so the symlink was transferred
and `--delete` removed the real directory behind it. Nothing unique was lost -- CLEVR-Lite is
deterministic, which is why it is gitignored -- and the regenerated val split is byte-identical in
its questions and pixel-identical in its images (verified above). The sync script's exclusions
lost their trailing slashes so a symlink cannot slip through again.

## Open items

- Row count per sample for SAE training is 24 question positions, not 25 as on LLaVA-1.5, so the
  collection will not reproduce the 4,898,438 rows/layer of the LLaVA-1.5 cache. Check
  `collection_info.json` against 24 x the train-split question count rather than against that
  number.
- Whether `replace` mode survives the pass-through gate (LLaVA-1.5: +0.00109). If it does not,
  the delta fragment applies and the reason gets recorded here.
