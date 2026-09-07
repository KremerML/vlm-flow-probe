# Gemma 3 replication: decisions, deviations, run log

The LLaVA-1.5 study (`cross-modal-information-flow-in-MLLM/overleaf/main_final.tex`)
re-run on Gemma 3 4B IT with Gemma Scope 2's pre-trained attention dictionaries. This file
is the reviewer's record: every place the Gemma run departs from the LLaVA protocol, why,
and where the evidence for each decision lives. The paper addition itself is
`overleaf/gemma3_addition.tex` in the archive repo.

## What is held fixed

CLEVR-Lite (same generated data, same validation split, same first-256 evaluation set);
the forced-choice margin `log P(true) - log P(false)` with per-token normalisation;
`Image->Question` and `Image->Last` knockout over every layer on the correctly answered
validation items; gradient x activation feature selection with the dictionary spliced
into the forward pass, `k = 200` per layer; `replace`-mode ablation at question positions;
activation-matched random controls (15 sets on the primary conditions); the 47-condition
multi-layer matrix and its analysis (`vfp-analyze-multilayer`, paired bootstrap over
questions, `R = A/K`).

## What differs, and why

| Item | LLaVA-1.5-7B | Gemma 3 4B IT | Reason / evidence |
|---|---|---|---|
| Decoder | 32 layers, d 4096, fp16 | 34 layers, d 2560, bf16; global attention every 6th layer (5, 11, 17, 23, 29), 1024-token sliding window elsewhere | Gemma 3 is unstable in fp16. Prompts are 286 tokens, inside the window, so every layer sees the full sequence. |
| Image tokens | 576 CLIP patches | 256 pooled SigLIP tokens, bidirectional among themselves (`token_type_ids` passed on every forward) | HF processor materialises them in `input_ids`; geometry is post-expansion for free. |
| Dictionaries | trained here, ReLU + L1, 32,768 features on `attn_out` (post-`o_proj`), L0 1100-1600 | Gemma Scope 2 `attn_out_all/layer_N_width_16k_l0_big`: JumpReLU, 16,384 features, hooked at the `o_proj` **input** (2048-wide), nominal L0 60-120, trained on text | User asked for pre-trained SAEs. The hook point is Gemma Scope's choice (`config.json: hf_hook_point_in = model.layers.N.self_attn.o_proj.input`); the repo exposes it as site `attn_z` through an identity tap. Fit to CLEVR-Lite activations is measured, not assumed (`reconstruction_eval.json` per layer). |
| Question span | intended: question tokens; effective: every post-image text position (sublist match never hit) | every post-image text position, by construction | Same effective span; makes the knockout a complete cut. Documented in `adapters/hf_gemma3.py`. |
| Answer form | `" " + lowercase` after `ASSISTANT:` | capitalised, no space, after `model\n` | Bring-up on 32 items: the model generated `Blue`, `Square`, ... on every item; `Blue` scores 0.00 nats, `blue` -16.5, `" blue"` -19.9 (`output/gemma3_bringup.json`). `ModelAdapter.format_answer` / `answer_prefix`. |
| Logit precision | fp16 logits | bf16 decoder, float32 untied `lm_head` | bf16 quantises a logit of magnitude 30 to steps of 0.125-0.25; every bring-up margin was such a multiple before the change. |
| Image encoding | per forward | once per sample, features scattered into `inputs_embeds` (the model's own path) | 138 forwards per sweep sample share one image; verified identical logits (`test_cached_features_match_the_pixel_path`). |
| Knockout mechanism | 4D mask edit | same; verified at layer 0 (sliding) and 5 (global) | `vfp-verify-adapter` 15/15 at both; the sliding/global risk in `docs/adding-a-model.md` did not materialise under transformers 4.57 eager attention. |
| Ablation mode | `replace` (pass-through drop +0.001) | `residual` (delta): only the selected features' decoded contribution is subtracted | Dictionary fit: explained variance 0.79 / 0.82 / 0.81 / 0.74 at layers 0 / 9 / 17 / 26 on 203,245 question-position rows, task mean L0 121-169 against the nominal 60-120, 48-79% of features never firing on the task. Replace mode would inject that error at every hooked position; the multi-layer gate measures the replace-mode pass-through drop so the avoided damage is on record. `configs/fragments/ablation_gemma_delta.yaml`. |
| Feature scoring | dictionary spliced in, forward runs on the reconstruction | same, plus the SAE error term added back (Marks et al. 2024): forward pass exact, gradients still reach `z` | `feature_identification.error_term: true`; default False keeps the archived LLaVA catalogs reproducible. |
| Compute | RTX 4090 | Snellius `gpu_a100` (H100 partition down on 2026-09-06) | ~100 ms per language-model forward. |

## Run log

| Date | Step | Where |
|---|---|---|
| 2026-09-06 | Adapter, JumpReLU import path, configs, tests written; CPU suite green (188) | this repo |
| 2026-09-06 | Weights + 34 SAEs fetched on a login node; CLEVR-Lite val synced | `$HOME/.cache/huggingface`, `~/vlm-flow-probe/datasets` |
| 2026-09-06 | Bring-up: verify 15/15 at layers 0 and 5; answer-form finding; fp32 head + feature cache added | jobs 26429169, 26429315, 26429387; `output/gemma3_bringup.json` |
| 2026-09-06 | Knockout sweeps submitted, one job per flow (ETA ~17 h each) | jobs 26429421 (Image->Question), 26429422 (Image->Last) |
| 2026-09-06 | Activation collection (val, 34 layers, `attn_z`) and per-layer import -> identify -> ablate chains submitted | jobs 26429437, 26429438-41 |
| 2026-09-06 | Collection done (203,245 rows x 34 layers); dictionary fit measured; per-layer jobs restarted after switching to delta mode + error-term scoring | jobs 26429638/40/41/43 |

| 2026-09-07 01:40 | Span fixed from the partial Image->Question sweep (n = 1,896 of ~7,760; the top-8 ranking is identical on the first 1,000): **layers 11-17**, concentrated/anchor layer 17, second anchor 12, sensitivity span drops the inhibitory layer 16. Multi-layer matrix queued behind the per-layer jobs. | `configs/experiments/gemma3_4b/multilayer_l11-17_attn_z_question.yaml` |

## Span decision (preliminary sweep, n = 1,896)

Image->Question mean margin drop per layer, top of the ranking: 17 (5.61), 0 (2.77), 12 (2.51),
11 (2.26), 26 (1.62), 15 (1.42), 14 (0.89), 19 (0.49), 30 (0.48), 13 (0.34). Inhibitory
(knockout raises the margin): 9, 16, 18, 20-25, 27, 31-33, with 33 at -1.68 and 21/27 near -1.3.
Image->Last is weak everywhere except layer 26 (+1.24) and is inhibitory at most late layers.

Rule applied, as in the LLaVA study: the contiguous band that contains the strongest transfer
layers other than layer 0, inhibitory members retained. Five of the six strongest non-zero
layers (11, 12, 14, 15, 17) sit in 11-17; layer 26 is isolated and left to the per-layer
single-site results. The band is seven layers rather than five, so the equal-budget point of
the spread-vs-concentrated comparison is 40 x 7 = 280 features and k = 280 is added to the
concentrated curve. Layer 0 is the second-strongest site and, unlike LLaVA's, its dictionary is
usable (explained variance 0.79); it is analysed as a single site, not folded into the span.

| 2026-09-07 08:17 | Replace-mode pass-through gate (nothing ablated, reconstruction substituted): margin drop +1.56 over L11-17, +1.52 L12-17, +1.39 L13-17, +0.61 L14-17, +0.50 L15-17, **-1.58** L16-17, **-0.90** L17 alone; relative perturbation at the site 0.037-0.042. Delta-mode pass-through 0.0000 exactly. Confirms delta mode was necessary: the reconstruction error alone is the size of a single-layer feature effect and changes sign with the span. | `output/experiments/gemma3_4b_multilayer_clevr_lite_l11-17_attn_z_question_replace_gate` (job 26431513) |

| 2026-09-07 10:24 | 57-condition matrix complete (2 h 39 m); analysis, decomposition and figures run; isolation phase (ablation with Image->Question severed at every layer) added and run | `output/experiments/gemma3_4b_multilayer_clevr_lite_l11-17_attn_z_question/analysis/`, `gemma3_4b_metric_decomposition.json`, `output/paper_figures/gemma3/` |

## Results in one place

Numbers are in `overleaf/gemma3_addition.tex` (archive repo) with their tables; the headline
readings, so a reviewer can check the text against the artifacts:

- **Dictionary fit** (`*/reconstruction_eval.json`, `gemma3_4b_sae_fit_table.json`): explained
  variance 0.67-0.95 (median 0.79), task L0 75-169, dead 46-86%. Replace-mode pass-through
  drop +1.56 (L11-17) / -1.58 (L16-17). Delta mode used throughout.
- **Landscape** (final, n = 7,790: the model answers every validation item correctly;
  `gemma3_4b_knockout_landscape_final.json`): Image->Question 17: 5.58, 0: 2.72, 12: 2.49,
  11: 2.09, 26: 1.60, 15: 1.42, 14: 0.86; negative at 17 of 34 layers (1, 4, 6, 8, 9, 16, 18,
  20-25, 27, 31-33), 33: -1.67. Image->Last: 26: +1.23, everything else |x| < 0.35 except the
  same inhibitory tail. The span decision made on n = 1,896 stands unchanged. True-option loss
  under single-layer knockout only at layer 11 (0.69 nats).
- **Single-layer A vs K, all 34 layers**: Spearman 0.03; A > K at 23 layers. Span layers
  (A / K): 11: 6.88/2.46, 12: 0.04/2.55, 13: 0.82/0.29, 14: 2.66/0.89, 15: 5.02/1.36,
  16: 9.45/-0.40, 17: 2.18/5.43 (R 0.40 [0.34, 0.46]). Controls <= 0.34.
- **Decomposition** (`gemma3_4b_metric_decomposition.json`): every single-layer ablation and
  every single-layer knockout except L11 leaves the true option within 0.02 nats; margin drops
  are false-option rises. Baseline true -0.05, false -32.66.
- **Matrix**: joint A(11-17) 28.61 (true 10.02 + false 18.59; gen acc 0.000; emits "okay" 174/256)
  vs span K 23.55 (true 0.95 + false 22.60; gen acc 0.773); R 1.215 [1.176, 1.256]; nested R
  0.40, 2.13, 2.28, 1.63, 1.73, 1.80, 1.22; slope +0.044/layer [+0.031, +0.056]; pooled 1.538.
  Budget at L17: 0.68, 1.06, 2.17, 2.99, 3.98, 6.19 for k = 40..800 (true option unmoved);
  spread 40x7 = 23.82 (gen acc 0.008; control 0.69). Downstream anchor 17: 2.17 + 14.94 ->
  14.92 combined (excess -2.19). Full knockout at all 34 layers: margin -0.20 (chance), true
  -6.09.
- **Isolation** (image severed at all 34 layers, then ablate): floor margin -0.20 (chance,
  fc 0.473, true -6.09); single-layer ablations move the margin <= 0.32 but L16 costs the true
  option another 1.30 nats and turns 113/256 generations into "it"; the joint ablation takes
  the true option to -9.52 and every generation to "this". Off-pathway damage is real.
- **Reading**: the calibration's preconditions (dictionary fits the site; ablation acts within
  the pathway; metric unsaturated) all fail on this model + dictionary pair; the addition reports
  it as a boundary-conditions result with the diagnostics that detect each failure.

| 2026-09-07 15:05 | Both sweeps complete (Image->Last 16 h 20 m, Image->Question 17 h 45 m); final distill, decomposition, figures; landscape numbers filled into the addition | `output/experiments/gemma3_4b_knockout_clevr_lite_{iq,il}/knockout/knockout_summary.json`, `overleaf/gemma3_addition.tex`, `overleaf/paper_figures/gemma3_4b_*` |

## Deliverables

- Paper addition (separate file, for manual merge): `cross-modal-information-flow-in-MLLM/overleaf/gemma3_addition.tex`;
  figures `overleaf/paper_figures/gemma3_4b_fig{1_knockout_landscape,_decomposition,6_budget_distribution,8a_ablation_vs_knockout,8b_recovered_share,_single_layer}.png`;
  bib entries and suggested abstract/limitations edits as comments at the end of the tex.
- Code: `adapters/hf_gemma3.py`, `core/sparse_autoencoder.py` (JumpReLU), `cli/import_sae.py`,
  the `isolation` phase and `error_term` option, Gemma configs, Slurm wrappers, analysis scripts.
- Artifacts (summaries, committed-size): `output/experiments/gemma3_4b_*`; raw per-sample files
  and dictionaries remain on Snellius scratch (`/scratch-shared/rkremer/vlm-flow-probe/output`,
  14-day purge).
