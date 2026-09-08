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
| 2026-09-08 01:31 | Cluster bring-up done on an H100: verify 15/15 at layers 0 and 11, same geometry and the same margin as locally (7.298 vs 7.297 on 32 items); forward 67 ms against the 4090's 176 ms | job 26457746, `output/llava16_bringup.json` |
| 2026-09-08 01:35 | **Knockout sweeps submitted**, one job per flow: full val split, `filter_correct`, window 1, all 32 layers, 249,280 steps each at ~7 steps/s (ETA ~10 h) | jobs 26457768 (Image->Question), 26457769 (Image->Last) |
| 2026-09-08 01:35 | **Activation collection submitted**: train split, all 32 layers, question positions, writing to scratch (~1.2 TB: 186,638 samples x 24 positions x 4096 x fp16 x 32 layers) | job 26457770 |
| 2026-09-08 02:1x | Span fixed from the partial `Image->Question` sweep (n = 1,546): **layers 10-14**, concentrated 11, sensitivity span drops 13 -- the same band as LLaVA-1.5 | `configs/experiments/llava16/multilayer_l10-14_attn_out_question.yaml` |
| 2026-09-08 03:45 | All 32 per-layer chains (train -> identify -> ablate) queued with `--dependency=afterok:26457770`, span layers and layer 0 first, so they start when the collection succeeds and not before | jobs 26459245-26459276 |
| 2026-09-08 07:40 | **Collection complete** (6 h 08 m): 32 layers x 4,711,800 rows x 4096, 1.2 TB on scratch. Per-layer chains released by the dependency and running | job 26457770, `output/activations/llava16_clevr_lite_question/collection_info.json` |
| 2026-09-08 10:00 | `Image->Last` sweep complete (n = 7,186 correctly answered of 7,790, 92.2% accuracy); peaks late: layer 19 (+0.558), 17 (+0.411), 14 (+0.343) | job 26457769, `.../llava16_knockout_clevr_lite_il/knockout/knockout_summary.json` |
| 2026-09-08 10:00 | All 32 dictionaries trained; 29 of 32 chains through identify + ablate | jobs 26459245-26459276, `output/experiments/llava16_sae_fit_table.json` |
| 2026-09-08 10:05 | Replace-mode pass-through gate submitted | job 26464470 |

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

## Span decision (preliminary sweep, n = 1,546 of 7,790)

Per-layer `Image->Question` margin drop, window 1, correctly answered validation items:

| layer | 0 | 8 | 10 | 11 | 12 | 13 | 14 | 17 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| drop | +1.938 | +0.344 | +0.460 | **+1.519** | +0.231 | -0.210 | **+1.318** | +0.527 | +0.640 | +0.295 |
| Cohen's d | +1.02 | +0.77 | +1.11 | +1.59 | +0.70 | -0.93 | +1.51 | +0.69 | +1.25 | +1.00 |

By the paper's rule -- the contiguous band containing the strongest `Image->Question` layers
other than layer 0, inhibitory members retained -- the span is **layers 10-14**, the concentrated
layer is **11** and the sensitivity span drops **13**.

That is the *same* band, the same strongest layer and the same dropped layer as the published
LLaVA-1.5 run, so the generated matrix config is field-for-field identical to the archived one in
its condition set (`configs/experiments/llava16/multilayer_l10-14_attn_out_question.yaml` against
`configs/frozen/multilayer_l10-14_attn_out_question.yaml`: nested spans, non-nested spans, budget
curve, downstream anchors 11 and 14, sensitivity span). The comparison is like-for-like without
having to run a second matrix at the LLaVA-1.5 span.

The top-3 layers are also identical to LLaVA-1.5's (`[0, 11, 14]`, the set the equivalence gate
pins). Doubling the visual context did not move where the image-to-question flow lives.

To be re-checked against the completed sweep; the Gemma decision taken at n = 1,896 held at
n = 7,790.

## Incidents

**The cluster's dataset directory was deleted and regenerated (2026-09-08).** A local `datasets`
symlink into the archive checkout was rsynced to the cluster: `snellius_sync.sh` excluded
`datasets/` with a trailing slash, which matches directories only, so the symlink was transferred
and `--delete` removed the real directory behind it. Nothing unique was lost -- CLEVR-Lite is
deterministic, which is why it is gitignored -- and the regenerated val split is byte-identical in
its questions and pixel-identical in its images (verified above). The sync script's exclusions
lost their trailing slashes so a symlink cannot slip through again.

## The row count confirms the tokenizer finding exactly

The 32-layer collection over the train split gives **4,711,800 rows per layer** from 186,638
samples. LLaVA-1.5's cache holds 4,898,438. The difference is 186,638 -- the sample count, to the
row: exactly one question position fewer per sample, on every sample, which is the
`add_prefix_space` difference and nothing else. The image block grew by 600 tokens and the text
side lost precisely one.

## Dictionary fit: the first precondition holds

All 32 dictionaries, trained on this task's own activations with the LLaVA recipe
(`output/experiments/llava16_sae_fit_table.json`, written by `scripts/sae_fit_table.py`):

| | explained variance | mean L0 | dead fraction |
|---|---|---|---|
| range over 32 layers | 0.99893 - 0.99995 | 784 - 1,768 | 0.0002 - 0.79 |
| span layers 10-14 | 0.99893 - 0.99957 | 1,221 - 1,499 | 0.0002 - 0.0016 |
| layer 0 | 0.99987 | 1,468 | 0.79 |

The dead fraction is high only at layers 0 and 1 (0.79, 0.51), as it was on LLaVA-1.5 (74% at
layer 0); everywhere the analysis uses it is below 1.5%. Compare Gemma's pre-trained dictionaries
on the same task: explained variance 0.74-0.82 with 48-79% dead. The precondition that broke the
Gemma replication is comfortably satisfied here, which is the point of training dictionaries on
the model and task being studied.

## Open items

- Whether `replace` mode survives the pass-through gate (LLaVA-1.5: +0.00109). If it does not,
  the delta fragment applies and the reason gets recorded here.
