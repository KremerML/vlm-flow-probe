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
| 2026-09-08 10:00 | All 32 dictionaries trained; 29 of 32 chains through identify + ablate | jobs 26459245-26459276, `output/experiments/llava16/llava16_sae_fit_table.json` |
| 2026-09-08 10:05 | `Image->Question` sweep complete (n = 7,186). The n = 1,546 span decision holds unchanged: same ranking, layers 11 (+1.512) and 14 (+1.313) dominating, 13 inhibitory (-0.203) | job 26457768 |
| 2026-09-08 10:52 | **Replace-mode pass-through gate passes**: worst span drop +0.00777 against a 0.02 threshold. The run stays in the published replace-mode protocol | job 26464470 |
| 2026-09-08 11:00 | Full matrix submitted (`--phases all`, includes the isolation phase), 32/32 per-layer chains complete | job 26465465 |
| 2026-09-08 13:00 | **Matrix complete** (53 conditions, ~2 h); redundancy analysis, decomposition, single-layer A-vs-K table and six figures generated; summaries synced back | job 26465465, `analysis/multilayer_summary.md`, `output/paper_figures/llava16/` |

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
(`output/experiments/llava16/llava16_sae_fit_table.json`, written by `scripts/sae_fit_table.py`):

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

## Pass-through: the second precondition holds, and replace mode stays

With nothing ablated and the dictionary's reconstruction substituted at every span layer:

| condition | margin drop |
|---|---|
| `gate_none` (harness check) | +0.00000 |
| passthrough L14 | +0.00481 |
| passthrough L13-14 | +0.00626 |
| passthrough L12-14 | +0.00671 |
| passthrough L11-14 | +0.00612 |
| passthrough L10-14 | +0.00777 |
| delta passthrough L10-14 | +0.00000 |

The worst case is +0.008 against a 0.02 threshold, on a baseline margin of 7.63 -- a tenth of a
percent. LLaVA-1.5's was +0.00109; the difference is partly the ~2x margin scale from the
tokenization above. Gemma's, for contrast, ran to +1.56 and changed sign with the span, which is
why that run had to leave the protocol. **This run stays in replace mode**, so the ablation
numbers are directly comparable to the published ones.

## Full sweep confirms the span

At n = 7,186 the `Image->Question` ranking is unchanged from n = 1,546: 0 (+1.944), 11 (+1.512),
14 (+1.313), 19 (+0.639), 17 (+0.510), 10 (+0.464), with 13 inhibitory (-0.203). Span 10-14
stands, as the Gemma decision did.

## Results

53 conditions, 256 evaluation items, replace mode, span 10-14.
Artifacts: `output/experiments/llava16/llava16_multilayer_clevr_lite_l10-14_attn_out_question/analysis/`
(`multilayer_summary.json`, `.md`), `output/experiments/llava16/llava16_single_layer_ak.json`,
`output/experiments/llava16/llava16_metric_decomposition.json`, figures under
`output/paper_figures/llava16/`.

### The headline: recovery is much higher than on LLaVA-1.5, and not constant

**R = A/K over the full span 10-14 is 93.6% (95% CI 89.0-98.5%)**, from A = 3.6422 and
K = 3.8899. The published LLaVA-1.5 figure is 72.6%.

| kind | span | size | A | K | R | 95% CI |
|---|---|---|---|---|---|---|
| nested | `{14}` | 1 | 1.7021 | 1.2329 | 138.1% | 131.3 - 145.4 |
| nested | `{13,14}` | 2 | 2.1778 | 1.1197 | 194.5% | 185.1 - 205.4 |
| nested | `{12,13,14}` | 3 | 2.8235 | 1.5376 | 183.6% | 175.6 - 192.4 |
| nested | `{11-14}` | 4 | 3.3639 | 3.5895 | 93.7% | 89.1 - 98.7 |
| nested | `{10-14}` | 5 | 3.6422 | 3.8899 | 93.6% | 89.0 - 98.5 |
| non-nested | `{10,11,12}` | 3 | 1.9796 | 3.4513 | 57.4% | 53.5 - 61.5 |
| non-nested | `{10,12,14}` | 3 | 2.4192 | 2.0060 | 120.6% | 116.1 - 125.3 |
| sensitivity | `{10,11,12,14}` | 4 | 3.1682 | 3.6939 | 85.8% | 81.2 - 90.6 |

R is **not** constant across spans -- no value lies inside every interval -- and it trends with
span size at -0.19 per layer (95% CI -0.212 to -0.169, excluding zero). At short spans ablation
*exceeds* the knockout ceiling (up to 194.5%); by four and five layers it sits just below it.
The LLaVA-1.5 per-span ratios are also not individually constant (65-88%), so what changes with
the vision front-end is the level, not the fact of dispersion.

A reading consistent with the numbers, though this run does not test it directly: a single-layer
knockout is easy for the model to route around -- it can re-read the image at any other layer --
so K is small at short spans, while the ablation removes content that later layers cannot restore
by re-reading. Severing five layers at once is much harder to route around, and there the two
interventions come back into line.

### The features do act within the pathway

The isolation phase severs `Image->Question` at *every* layer and then ablates on top:

| condition | margin drop |
|---|---|
| `full_knockout_L0-31` (the no-image floor) | +6.9215 |
| `isolated_joint_L10-14` (ablation on top of it) | +6.9508 |
| `isolated_ablate_L11` | +6.6058 |

With no image information reaching the text positions anywhere in the stack, ablating the span's
1,000 selected features moves the margin by +0.03, and single-layer isolation moves it the wrong
way. What the ablation still does in that state is lower *both* options together -- the joint
condition takes the true option from -8.70 to -12.76 and the false from -8.57 to -12.60 -- a
general loss of confidence rather than a separation of the two, which is why the margin does not
move. The features have nothing left to take *from the margin* once the pathway is cut, which is
what the calibration requires. This is the precondition Gemma failed, and it holds here.

### Spreading still beats concentrating, by 2x rather than 3x

Same 200-feature budget: spread 40 per layer over 10-14 gives +1.3590; concentrated at layer 11
gives +0.6949. A ratio of 1.96, against roughly 3x on LLaVA-1.5. Neither the ablation nor the
knockout curve saturates over spans of 1-5 layers (both fits collapse to lines, slopes 1.766 and
0.762 per layer), so the saturation claim cannot be evaluated on this range; what is identified is
the slope ratio, 2.318.

### The redundancy signature replicates

Leave-one-out marginal contributions sit far below the standalone single-layer effects -- layer 14
contributes +0.944 in context against +1.702 standalone (0.555), layer 11 +0.596 against +0.695
(0.858). The rest of the span already carries most of what any one layer contributes.

### Both arms move the margin the same way

Every effect decomposes into true-option loss and false-option rise. On this model both arms are
dominated by the false-option rise, with the true option unchanged or slightly improved:

| condition | margin | true drop | false rise |
|---|---|---|---|
| `joint_L10-14` (ablation) | +3.642 | -0.364 | +4.006 |
| `span_knockout_L10-14` | +3.890 | +0.303 | +3.587 |
| `nested_L14` (ablation) | +1.702 | -0.371 | +2.073 |
| `knockout_L14` | +1.233 | -0.167 | +1.400 |
| `full_knockout_L0-31` | +6.922 | -1.038 | +7.959 |

That the *knockout* behaves this way too -- including with the image entirely severed -- makes it a
property of the metric on this task, not a symptom of the ablation. A and K are therefore
measuring the same kind of change, which is what the ratio requires. (Gemma's problem was
different: there the margin was saturated, true near 0 nats and false near -33.)

### Single layers, same 256 samples

| layer | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|
| A | 0.389 | 0.695 | 0.480 | 0.751 | 1.702 |
| K | 0.448 | 1.433 | 0.217 | -0.183 | 1.233 |
| A/K | 0.867 | 0.485 | 2.214 | -- | 1.380 |

Layer 13's knockout is inhibitory, so its ratio is undefined; the span retains it because the
paper's rule does.

## Paper addition

`cross-modal-information-flow-in-MLLM/overleaf/llava16_addition.tex` (committed there as
`a73b9f4`), with the six figures as PNG and PDF under `overleaf/paper_figures/` and the
LLaVA-1.5 and LLaVA-NeXT entries appended to `refs.bib`. It compiles standalone (8 pages, no
errors); the undefined references that remain are the cross-links into `main_final.tex` and the
Gemma appendix, which resolve when it is `\input`. Suggested abstract, Limitations and
Availability edits are comments at the end of the file -- the Limitations sentence is a combined
replacement that also carries the Gemma addition's suggestion, since both edit the same sentence.

## What this means for the calibration

The three preconditions all hold: the dictionaries fit the site (EV >= 0.9989), the ablation acts
within the pathway (isolation), and the margin is unsaturated (baseline 7.63, log P(true) -9.5).
The qualitative account survives too -- strong redundancy, spreading beats concentrating, the same
span, the same top-3 layers.

What does not transfer is the number. Recovery at the full span is 93.6% where LLaVA-1.5 gives
72.6%, and at short spans ablation exceeds the knockout ceiling outright. Holding the language
model, the prompt, the data and the recipe fixed and changing only the vision front-end moves the
calibration by twenty points. Whatever R measures, it is not a property of the language model
alone.
