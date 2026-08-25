# Gate reference provenance

Vendored 2026-08-25 from the archive repo
`cross-modal-information-flow-in-MLLM` at commit `457875c8d93d6d08bf400c103746bc7ad3628d28`
(branch `multilayer-ablation`, 39 uncommitted paths at vendoring time — all docs/figures,
none under the referenced outputs).

| file | archive path | what it pins |
|---|---|---|
| sample_cache.json | output/sae_experiments/multilayer_clevr_lite_l10-14_attn_out_question/sample_cache.json | 256 val samples: question ids, post-expansion positions, per-sample baselines (Stage 0/1) |
| A0_regression_L11.summary.json | .../conditions/A0_regression_L11/summary.json | layer-11 200-feature replace-mode ablation: mean drop 0.21307, per-sample drops, 15 archived control feature sets (Stage 3) |
| knockout_summary_n7084.json | output/sae_experiments/exp_default/knockout/knockout_summary.json | full CLEVR-Lite knockout sweep, n=7084 per layer (Stage 2 reference) |
| knockout_summary_n10.json | output/test_clevr_lite_pipeline/knockout_summary.json | 10-sample smoke sweep (fast Stage 2 variant) |

Not vendored (too large; read from --archive-root):
- output/sae_experiments/sae_clevr_lite_layer11_attn_out_question/sae_checkpoint.pt (1.07 GB)
- output/sae_experiments/sae_clevr_lite_layer11_attn_out_question_causal/causal_feature_catalog.json
- datasets/clevr_lite/ (regenerable: seed 32)
